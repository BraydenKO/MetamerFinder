"""
Utility script to inspect a PyTorch model's structure and list its layer names.
Includes detection of layers inside residual skip connections.
"""

import argparse
import torch
import torch.nn as nn
import sys
import os
from typing import Dict, Optional

from metamer_finder import load_model, analyze_skip_connections

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
    try:
        model = load_model(args.model, device)
    except Exception as e:
        print(f"Error: {e}")
        return

    if isinstance(model, dict):
        print("\nNote: The loaded object is a dictionary (likely a state dict or checkpoint).")
        
        # Check if it looks like a checkpoint
        checkpoint_keys = ["state_dict", "model_state_dict", "params", "weights"]
        found_state_dict = next((k for k in checkpoint_keys if k in model and isinstance(model[k], dict)), None)
        
        target_dict = model[found_state_dict] if found_state_dict else model
        print(f"Listing keys within {found_state_dict if found_state_dict else 'dictionary'}:")
        print("-" * 45)
        for key in target_dict.keys():
            print(key)
        
        print("\nNote: Since this is just a state dict, residual analysis is unavailable.")
        print("To see full architecture analysis, provide a file containing the full nn.Module.")
        return

    # Analyze skip connections
    print("Analyzing model architecture for residual connections...")
    skip_analysis = analyze_skip_connections(model)
    
    list_layers(model, args.types, skip_analysis)
    print("\nNote: You can use any of the names above with the --layers argument in main.py.")
    print("Layers flagged as [Inside Residual] are part of a skip connection path.")

if __name__ == "__main__":
    main()
