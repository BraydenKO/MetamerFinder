"""
Metamer Finder: A CLI tool to generate images that match internal neural activations.
"""

import argparse
import torch
import torchvision.models as models
from PIL import Image
import sys
import os
from typing import List, Optional

# Import core components from the package
from metamer_finder import (
    FeatureExtractor, 
    MetamerOptimizer, 
    preprocess_image, 
    postprocess_tensor,
    concat_images,
    get_device,
    load_model,
    is_image_file,
    load_data,
    save_data
)

def parse_args():
    parser = argparse.ArgumentParser(
        description="Metamer Finder: Generate images or tensors with matching internal neural representations.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )
    
    # Input/Output Arguments
    parser.add_argument("-i", "--image", type=str, required=True, 
                        help="Path to the base/target data (image or .pt tensor).")
    parser.add_argument("-o", "--output", type=str, default=None, 
                        help="Custom filename for the generated metamer. If None, uses 'metamer_{input_name}.ext'.")
    parser.add_argument("--out_dir", type=str, default="results",
                        help="Directory to save the generated metamer.")
    
    # Model Arguments
    parser.add_argument("-m", "--model", type=str, default="vgg16", 
                        help="Torchvision model name OR path to a saved .pth model file.")
    parser.add_argument("-l", "--layers", type=str, nargs="+", default=["features.1"], 
                        help="One or more target layer names to match.")
    
    # Optimization Hyperparameters
    parser.add_argument("--lr", type=float, default=0.1, 
                        help="Learning rate for the Adam optimizer.")
    parser.add_argument("--iters", type=int, default=300, 
                        help="Maximum number of optimization iterations.")
    parser.add_argument("--loss_type", type=str, choices=['mse', 'gram'], default='mse',
                        help="Matching strategy: 'mse' for exact spatial, 'gram' for texture/style.")
    parser.add_argument("--tv_weight", type=float, default=0.0,
                        help="Total Variation penalty to smooth the generated output.")
    parser.add_argument("--seed", type=int, default=None, 
                        help="Random seed for reproducibility.")
    parser.add_argument("--loss_hook", type=str, default=None,
                        help="Path to a .py file containing a custom_penalty(synthetic_spikes, synthetic_features, target_features) function.")

    return parser.parse_args()

def load_custom_loss(path: str):
    """Dynamically loads a custom_penalty function from a .py file."""
    import importlib.util
    spec = importlib.util.spec_from_file_location("custom_loss", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    if hasattr(module, "custom_penalty"):
        return module.custom_penalty
    raise AttributeError(f"Module at {path} does not have a 'custom_penalty' function.")

def main():
    args = parse_args()
    
    if args.seed is not None:
        torch.manual_seed(args.seed)

    device = get_device()
    print(f"Using device: {device}")

    # 1. Resolve Image Path
    image_path = args.image
    if not os.path.exists(image_path):
        # Check in inputs/ folder as a fallback
        alt_path = os.path.join("inputs", image_path)
        if os.path.exists(alt_path):
            image_path = alt_path
        else:
            print(f"Error: Image '{args.image}' not found.")
            sys.exit(1)

    # 2. Load Model
    model_id = args.model
    if not os.path.exists(model_id):
        alt_model_path = os.path.join("models", model_id)
        if os.path.exists(alt_model_path):
            model_id = alt_model_path
            
    model = load_model(model_id, device)
    
    # 3. Prepare the Target Data
    is_vision = is_image_file(image_path)
    print(f"Processing target {'image' if is_vision else 'data'}: {image_path}")
    
    try:
        if is_vision:
            img = Image.open(image_path).convert('RGB')
            target_tensor = preprocess_image(img, target_size=(224, 224)).to(device)
        else:
            target_tensor = load_data(image_path, device)
            if target_tensor is None:
                raise ValueError(f"Could not load data from {image_path}. Expected an image or .pt/.pth file.")
    except Exception as e:
        print(f"Error processing input: {e}")
        sys.exit(1)
    
    # 4. Extract Target Features
    print(f"Targeting layers: {args.layers}")
    try:
        extractor = FeatureExtractor(model, args.layers)
    except ValueError as e:
        print(f"Error initializing extractor: {e}")
        sys.exit(1)
    
    with torch.no_grad():
        target_features = extractor(target_tensor)
        target_features = {k: v.clone().detach() for k, v in target_features.items()}
    
    # 5. Setup Optimization
    print(f"Optimizing for {args.iters} iterations (LR: {args.lr})...")
    noise_tensor = torch.randn_like(target_tensor) * 0.01
    
    custom_loss_hook = None
    if args.loss_hook:
        print(f"Loading custom loss hook from: {args.loss_hook}")
        custom_loss_hook = load_custom_loss(args.loss_hook)

    optimizer = MetamerOptimizer(
        extractor=extractor, 
        target_features=target_features, 
        lr=args.lr,
        max_iterations=args.iters,
        loss_type=args.loss_type,
        tv_weight=args.tv_weight,
        custom_loss_hook=custom_loss_hook
    )
    
    # 6. Generate the Metamer
    try:
        metamer_tensor = optimizer.generate(starting_image=noise_tensor)
    finally:
        extractor.remove_hooks()
    
    # 7. Save the Result
    os.makedirs(args.out_dir, exist_ok=True)
    input_filename = os.path.basename(image_path)
    base_name = os.path.splitext(input_filename)[0]
    
    if args.output:
        metamer_filename = args.output
    else:
        ext = ".jpg" if is_vision else ".pt"
        metamer_filename = f"metamer_{base_name}{ext}"
    
    output_path = os.path.join(args.out_dir, metamer_filename)
    
    print(f"Saving metamer to {output_path}...")
    try:
        if is_vision:
            result_img = postprocess_tensor(metamer_tensor)
            result_img.save(output_path)
            
            compare_filename = f"metamer_{base_name}_compare.jpg"
            compare_path = os.path.join(args.out_dir, compare_filename)
            print(f"Saving comparison image to {compare_path}...")
            
            target_img_for_compare = postprocess_tensor(target_tensor)
            comparison_img = concat_images(target_img_for_compare, result_img)
            comparison_img.save(compare_path)
        else:
            save_data(metamer_tensor, output_path)
        
        print("Success!")
    except Exception as e:
        print(f"Error saving result: {e}")

if __name__ == "__main__":
    main()
