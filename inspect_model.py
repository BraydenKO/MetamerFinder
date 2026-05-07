"""
Utility script to inspect a PyTorch model's structure and list its layer names.
Includes detection of layers inside residual skip connections.
"""

import argparse
import torch
import torch.nn as nn
import torch.fx
import operator
from typing import Dict, Set
from metamer_finder import get_device, load_model

def analyze_skip_connections(model: nn.Module) -> Dict[str, bool]:
    """
    Identifies modules that are inside residual skip connections using torch.fx.
    
    Returns:
        Dict[str, bool]: Mapping of module names to a boolean indicating if they are inside a residual.
    """
    is_residual = {name: False for name, _ in model.named_modules()}
    
    try:
        # Trace the model to get the computational graph
        traced = torch.fx.symbolic_trace(model)
        graph = traced.graph
        
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
                if len(inputs) < 2:
                    continue
                
                # Find common ancestors
                ancestor_sets = [get_ancestors(inp) | {inp} for inp in inputs]
                common_ancestors = set.intersection(*ancestor_sets)
                
                if not common_ancestors:
                    continue
                
                # The divergence point is the "latest" common ancestor (no other common ancestor is its descendant)
                # However, for simplicity and to meet the "strictly on transformation path" requirement:
                # A node is on a transformation path if it is an ancestor of the 'add' node 
                # but NOT an ancestor of ALL inputs (i.e., it's specific to some branches).
                
                # Actually, the user says: "Trace the inputs backward to find their divergence point. 
                # Flag call_module nodes strictly on the transformation path between divergence and addition."
                
                # Let's find the divergence point: the node in common_ancestors that has the highest index in graph.nodes
                divergence_point = max(common_ancestors, key=lambda n: list(graph.nodes).index(n))
                
                # Nodes on transformation paths are descendants of divergence_point and ancestors of 'node' (the add)
                # strictly means not the divergence point itself.
                
                def is_descendant(target: torch.fx.Node, start: torch.fx.Node) -> bool:
                    # Check if target is a descendant of start
                    stack = [start]
                    visited = set()
                    while stack:
                        n = stack.pop()
                        if n == target: return True
                        if n not in visited:
                            visited.add(n)
                            # This is inefficient but works for small graphs. 
                            # Better to check if start is in target's ancestors.
                            pass
                    return False
                
                # Efficient descendant check: start is in target's ancestors
                transformation_nodes = set()
                for inp in inputs:
                    path_nodes = {inp} | get_ancestors(inp)
                    # Filter: must be descendant of divergence_point
                    # A node n is a descendant of divergence_point if divergence_point is in n's ancestors or is n
                    for n in path_nodes:
                        if n == divergence_point:
                            continue
                        n_ancestors = get_ancestors(n)
                        if divergence_point in n_ancestors or n == divergence_point:
                            if n.op == 'call_module':
                                is_residual[str(n.target)] = True

    except Exception as e:
        print(f"\nWarning: torch.fx tracing failed for this model. Skip connection analysis unavailable.")
        print(f"Reason: {e}")
        return {}
        
    return is_residual

def list_layers(model: nn.Module, show_types: bool = False, skip_analysis: Dict[str, bool] = None):
    """Prints all layer names in the model with optional residual warnings."""
    skip_analysis = skip_analysis or {}
    
    print(f"\n{'Layer Name':<45} {'Type' if show_types else ''}")
    print("-" * (70 if show_types else 45))
    
    for name, module in model.named_modules():
        # Skip the root module
        if name == "":
            continue
            
        type_str = str(type(module)).split("'")[1].split(".")[-1] if show_types else ""
        
        # Check if inside residual
        warning = ""
        if skip_analysis.get(name):
            warning = " [WARNING: Inside Residual]"
            
        print(f"{name:<45} {type_str}{warning}")

def parse_args():
    parser = argparse.ArgumentParser(description="Inspect a PyTorch model and list its layer names.")
    parser.add_argument("-m", "--model", type=str, default="vgg16", 
                        help="Torchvision model name OR path to a saved .pth model file.")
    parser.add_argument("-t", "--types", action="store_true", 
                        help="Show the type of each layer (e.g., Conv2d, ReLU).")
    return parser.parse_args()

def main():
    args = parse_args()
    device = torch.device("cpu")  # Inspection is fast on CPU
    
    print(f"Loading model '{args.model}' for inspection...")
    model = load_model(args.model, device)
    
    # Analyze skip connections
    print("Analyzing model architecture for residual connections...")
    skip_analysis = analyze_skip_connections(model)
    
    list_layers(model, args.types, skip_analysis)
    print("\nNote: You can use any of the names above with the --layers argument in main.py.")

if __name__ == "__main__":
    main()
