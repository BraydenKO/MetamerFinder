import torch
import torch.nn as nn
import torch.fx
import operator
import builtins
import functools
from typing import Dict, Set, Any, List

# --- CAPTURE ORIGINALS AT STARTUP ---
# We keep these to avoid recursion when monkeypatching builtins and torch functions.
_orig_ones = torch.ones
_orig_zeros = torch.zeros
_orig_arange = torch.arange
_orig_triu = torch.triu
_orig_tril = torch.tril
_orig_tensor = torch.tensor
_orig_cat = torch.cat
_orig_stack = torch.stack
_orig_topk = torch.topk
_orig_len = builtins.len
_orig_str = builtins.str
_orig_int = builtins.int
_orig_float = builtins.float
_orig_isinst = builtins.isinstance

# Global reference to the active tracer for resilient wrappers
_ACTIVE_TRACER = None

def has_proxy(obj):
    """Recursively check if an object contains a Proxy."""
    if isinstance(obj, torch.fx.Proxy): return True
    if isinstance(obj, (list, tuple)): return any(has_proxy(i) for i in obj)
    if isinstance(obj, dict): return any(has_proxy(v) for v in obj.values())
    return False

# --- RESILIENT WRAPPERS ---
# These wrappers return a Proxy if any argument is a Proxy, otherwise they call the original.
# This prevents tracing crashes due to dynamic shapes or device attributes.

def _fx_resilient_ones(*args, **kwargs):
    if _ACTIVE_TRACER is not None and has_proxy((args, kwargs)):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_ones, args, kwargs)
    return _orig_ones(*args, **kwargs)

def _fx_resilient_zeros(*args, **kwargs):
    if _ACTIVE_TRACER is not None and has_proxy((args, kwargs)):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_zeros, args, kwargs)
    return _orig_zeros(*args, **kwargs)

def _fx_resilient_arange(*args, **kwargs):
    if _ACTIVE_TRACER is not None and has_proxy((args, kwargs)):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_arange, args, kwargs)
    return _orig_arange(*args, **kwargs)

def _fx_resilient_triu(*args, **kwargs):
    if _ACTIVE_TRACER is not None and has_proxy((args, kwargs)):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_triu, args, kwargs)
    return _orig_triu(*args, **kwargs)

def _fx_resilient_tril(*args, **kwargs):
    if _ACTIVE_TRACER is not None and has_proxy((args, kwargs)):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_tril, args, kwargs)
    return _orig_tril(*args, **kwargs)

def _fx_resilient_tensor(*args, **kwargs):
    if _ACTIVE_TRACER is not None and has_proxy((args, kwargs)):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_tensor, args, kwargs)
    return _orig_tensor(*args, **kwargs)

def _fx_resilient_cat(*args, **kwargs):
    if _ACTIVE_TRACER is not None and has_proxy((args, kwargs)):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_cat, args, kwargs)
    return _orig_cat(*args, **kwargs)

def _fx_resilient_stack(*args, **kwargs):
    if _ACTIVE_TRACER is not None and has_proxy((args, kwargs)):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_stack, args, kwargs)
    return _orig_stack(*args, **kwargs)

def _fx_resilient_topk(*args, **kwargs):
    if _ACTIVE_TRACER is not None and has_proxy((args, kwargs)):
        res = _ACTIVE_TRACER.create_proxy('call_function', _orig_topk, args, kwargs)
        return res[0], res[1] # topk returns (values, indices)
    return _orig_topk(*args, **kwargs)

def _fx_resilient_len(obj):
    if _ACTIVE_TRACER is not None and isinstance(obj, torch.fx.Proxy):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_len, (obj,), {})
    return _orig_len(obj)

def _fx_resilient_str(obj):
    if _ACTIVE_TRACER is not None and isinstance(obj, torch.fx.Proxy):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_str, (obj,), {})
    return _orig_str(obj)

def _fx_resilient_int(obj):
    if _ACTIVE_TRACER is not None and isinstance(obj, torch.fx.Proxy):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_int, (obj,), {})
    return _orig_int(obj)

def _fx_resilient_float(obj):
    if _ACTIVE_TRACER is not None and isinstance(obj, torch.fx.Proxy):
        return _ACTIVE_TRACER.create_proxy('call_function', _orig_float, (obj,), {})
    return _orig_float(obj)

class ResilientDict(dict):
    """
    A dictionary that handles Proxy keys and attribute access by recording 
    the access in the FX graph instead of failing.
    """
    def __getitem__(self, key):
        if _ACTIVE_TRACER is not None and isinstance(key, torch.fx.Proxy):
            return _ACTIVE_TRACER.create_proxy('call_function', operator.getitem, (self, key), {})
        try:
            val = super().__getitem__(key)
            if isinstance(val, dict) and not isinstance(val, ResilientDict):
                val = ResilientDict(val)
                self[key] = val
            return val
        except KeyError:
            if _ACTIVE_TRACER is not None:
                return _ACTIVE_TRACER.create_proxy('call_function', operator.getitem, (self, key), {})
            raise
            
    def __contains__(self, key):
        if _ACTIVE_TRACER is not None and isinstance(key, torch.fx.Proxy):
            return True
        return super().__contains__(key)
        
    def __getattr__(self, name):
        if name in self:
            val = self[name]
            if isinstance(val, dict) and not isinstance(val, ResilientDict):
                val = ResilientDict(val)
                self[name] = val
            return val
        if name.startswith('_'): raise AttributeError(name)
        try: return super().__getattribute__(name)
        except AttributeError:
            if _ACTIVE_TRACER is not None:
                return _ACTIVE_TRACER.create_proxy('call_function', getattr, (self, name), {})
            raise

class ResilientTracer(torch.fx.Tracer):
    """
    A robust tracer that handles complex models by pre-wrapping dictionaries,
    monkeypatching problematic functions, and handling Proxy iteration/unpacking.
    """
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._wrap_cache = {}

    def is_leaf_module(self, m: nn.Module, module_qualified_name: str) -> bool:
        # Standard leaf modules
        standard_leaves = (nn.Linear, nn.Conv2d, nn.BatchNorm2d, nn.LayerNorm, nn.Dropout)
        if isinstance(m, standard_leaves): return True
        return super().is_leaf_module(m, module_qualified_name)

    def getattr(self, *args, **kwargs) -> Any:
        res = super().getattr(*args, **kwargs)
        if isinstance(res, dict) and not isinstance(res, ResilientDict):
            obj_id = id(res)
            if obj_id not in self._wrap_cache: self._wrap_cache[obj_id] = ResilientDict(res)
            return self._wrap_cache[obj_id]
        return res

    def call_module(self, m: nn.Module, forward: Any, args: Any, kwargs: Any) -> Any:
        res = super().call_module(m, forward, args, kwargs)
        # Handle tuple unpacking for multi-return modules
        if isinstance(res, torch.fx.Proxy):
            name = m.__class__.__name__.lower()
            if "attention" in name: return res[0], res[1]
            if "transformerencoder" in name and "layer" not in name: return res[0], res[1], res[2], res[3]
            if "inferenceadapter" in name or "neuraldatatransformer" in name: return res[0], res[1], res[2], res[3]
        return res

    def trace(self, root: nn.Module, concrete_args: Dict[str, Any] = None):
        global _ACTIVE_TRACER
        _ACTIVE_TRACER = self
        self._wrap_cache = {}
        
        # 1. Pre-wrap dictionaries in the model hierarchy
        dict_patches = []
        for _, m in root.named_modules():
            for attr_name in dir(m):
                if attr_name.startswith('_'): continue
                try:
                    attr = getattr(m, attr_name)
                    if isinstance(attr, dict) and not isinstance(attr, ResilientDict):
                        dict_patches.append((m, attr_name, attr))
                        setattr(m, attr_name, ResilientDict(attr))
                except: continue

        # 2. Patch problematic torch and builtin functions
        patches = {
            torch: {
                'ones': _fx_resilient_ones, 'zeros': _fx_resilient_zeros,
                'arange': _fx_resilient_arange, 'triu': _fx_resilient_triu,
                'tril': _fx_resilient_tril, 'tensor': _fx_resilient_tensor,
                'cat': _fx_resilient_cat, 'stack': _fx_resilient_stack,
                'topk': _fx_resilient_topk
            },
            builtins: {
                'len': _fx_resilient_len, 'str': _fx_resilient_str,
                'int': _fx_resilient_int, 'float': _fx_resilient_float
            }
        }
        originals = {}
        for mod, funcs in patches.items():
            for name, wrapper in funcs.items():
                try:
                    orig = getattr(mod, name)
                    originals[(mod, name)] = orig
                    setattr(mod, name, wrapper)
                except: pass
        
        # 3. Safe isinstance wrapper
        def resilient_isinstance(obj, types):
            try: return _orig_isinst(obj, types)
            except TypeError:
                if _ACTIVE_TRACER is not None: return True # Trace-time fallback
                raise
        builtins.isinstance = resilient_isinstance

        # 4. Proxy methods
        proxy_originals = {}
        proxy_originals['__iter__'] = torch.fx.Proxy.__iter__
        def resilient_iter(self):
            # yield enough proxies to satisfy common small unpackings or varargs
            for i in range(10): yield self[i]
        torch.fx.Proxy.__iter__ = resilient_iter
        
        proxy_originals['__bool__'] = getattr(torch.fx.Proxy, "__bool__", None)
        torch.fx.Proxy.__bool__ = lambda self: True

        try:
            return super().trace(root, concrete_args)
        finally:
            _ACTIVE_TRACER = None
            # Rigorous restoration of all monkeypatches
            for (mod, name), orig in originals.items(): setattr(mod, name, orig)
            builtins.isinstance = _orig_isinst
            torch.fx.Proxy.__iter__ = proxy_originals['__iter__']
            if proxy_originals['__bool__'] is not None:
                torch.fx.Proxy.__bool__ = proxy_originals['__bool__']
            else:
                try: delattr(torch.fx.Proxy, "__bool__")
                except: pass
            for m, attr_name, orig in dict_patches:
                try: setattr(m, attr_name, orig)
                except: pass

def analyze_skip_connections(model: nn.Module) -> Dict[str, bool]:
    """
    Identifies modules that are inside residual skip connections using torch.fx.
    
    Returns:
        Dict[str, bool]: Mapping of module names to a boolean indicating if they are inside a residual.
    """
    is_residual = {name: False for name, _ in model.named_modules()}
    
    try:
        # Use our resilient tracer instead of standard symbolic_trace
        tracer = ResilientTracer()
        graph = tracer.trace(model)
        
        # Helper to find all ancestors of a node
        def get_ancestors(node: torch.fx.Node) -> Set[torch.fx.Node]:
            ancestors = set()
            stack = list(node.all_input_nodes)
            while stack:
                n = stack.pop()
                if n not in ancestors:
                    ancestors.add(n)
                    stack.extend(n.all_input_nodes)
            return ancestors

        for node in graph.nodes:
            # Detect addition operations (residual joins)
            is_add = (node.op == 'call_function' and node.target in [operator.add, torch.add]) or \
                     (node.op == 'call_method' and node.target == 'add')
            
            if is_add:
                inputs = node.all_input_nodes
                if len(inputs) < 2: continue
                
                # Find common ancestors to identify the divergence point
                ancestor_sets = [get_ancestors(inp) | {inp} for inp in inputs]
                common_ancestors = set.intersection(*ancestor_sets)
                if not common_ancestors: continue
                
                divergence_point = max(common_ancestors, key=lambda n: list(graph.nodes).index(n))
                
                # Flag call_module nodes on the transformation path between divergence and addition
                for inp in inputs:
                    path_nodes = {inp} | get_ancestors(inp)
                    for n in path_nodes:
                        if n == divergence_point: continue
                        n_ancestors = get_ancestors(n)
                        if divergence_point in n_ancestors or n == divergence_point:
                            if n.op == 'call_module':
                                is_residual[str(n.target)] = True

    except Exception as e:
        print(f"\nWarning: torch.fx tracing failed for this model. Skip connection analysis unavailable.")
        print(f"Reason: {e}")
        return {}
        
    return is_residual
