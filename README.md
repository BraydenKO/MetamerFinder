# Metamer Finder

Metamer Finder is a specialized PyTorch research tool designed to generate neural metamers—physically distinct inputs that elicit nearly identical internal representations within a deep neural network.

By optimizing an input image (starting from noise or another image) to minimize the distance between its activations and a target image's activations, we can "see" what information is preserved at different stages of the neural hierarchy.

---

## Features

- **Dual-Interface System**:
  - **CLI**: High-throughput metamer generation for large-scale experiments.
  - **Interactive GUI**: A PyQt6-based application for real-time visualization, dynamic model switching, and selective pixel masking.
- **Advanced Optimization Strategies**:
  - **Exact Spatial (MSE)**: Matches the precise spatial activations of the target.
  - **Texture/Style (Gram Matrix)**: Matches the global statistics/texture of the target while discarding spatial constraints.
  - **Total Variation (TV) Smoothing**: Penalizes high-frequency noise for more "natural" looking visualizations.
- **Data-Agnostic Engine**: Optimized for vision models but compatible with any PyTorch nn.Module and raw continuous tensors of any dimension (e.g., audio, images, 1D time-series).
- **Architectural Awareness**: Automated analysis of residual skip connections to identify robust optimization targets.
- **Selective Optimization**: Interactive drawing canvas with Brush and Rectangle tools to "freeze" specific image regions during generation.
- **Live Visual Feedback**: Watch the metamer evolve in real-time as the optimizer matches the target activations.

---

## The Neuroscience Connection

In biological vision, metamers are stimuli that are perceptually indistinguishable despite physical differences. In the context of Artificial Neural Networks (ANNs), we use metamers to probe the representational hierarchy:

1. **Early Layers (e.g., VGG16 features.1)**: These representations are analogous to the Primary Visual Cortex (V1). Metamers matching these layers preserve local spatial edges, orientation, and color, resulting in an image that is often visually similar to the target.
2. **Late Layers (e.g., VGG16 features.25)**: These representations are analogous to the Inferotemporal (IT) Cortex. At this stage, the network has discarded precise spatial coordinates in favor of semantic content. Metamers generated here look like abstract "semantic textures" or "neural noise"—highly complex patterns that look nothing like the original to a human, yet are "identical" to the model.

---

## Architectural Insights: Safe Choke Points

When working with modern architectures like ResNet, not all layers are equal. Metamer Finder includes a specialized inspection tool (inspect_model.py) that uses torch.fx symbolic tracing to identify Residual Skip Connections:

- **Inside Residual**: Layers within a residual block only contribute a delta to the main identity path. Targeting these can lead to unstable optimization as the network can "bypass" your changes via the skip connection.
- **Safe Choke Point**: These are layers where the entire signal must pass through a single module (e.g., the output of a residual block). Targeting these ensures that your generated metamer truly captures the full state of the network at that depth.

---

## Example Results

Below is a metamer generated using VGG16 matching features.25 layer using MSE.

![VGG16 Dog Metamer](examples/vgg16_dog_metamer.jpg)  
*Left: Original Target Image | Right: Generated Metamer (Neural Noise)*

VGG 16 matching featues.30 using MSE and TV=0.001  
![VGG16 Dog Metamer](examples/vgg16_dog_layer30_tv_0-001.jpg)

VGG 16 matching featues.20 using Gram Matrix  
![VGG16 Dog Metamer](examples/vgg16_dog_layer20_gram.jpg)

---

## Installation

```bash
# Clone the repository
git clone https://github.com/your-repo/MetamerFinder.git
cd MetamerFinder

# Install dependencies
pip install -r requirements.txt
```

---

## Usage

### 1. Interactive GUI
The most user-friendly way to explore metamers.
```bash
python gui_app.py
```
- Load an image, select a model, and draw a mask on the canvas to freeze pixels.
- Choose between Exact Spatial (MSE) or Texture/Style (Gram Matrix) strategies.
- Adjust TV Smoothing Weight to reduce high-frequency adversarial artifacts.
- Use the Undo and Rectangle tools for precise masking.

### 2. Command Line Interface
For automated or large-scale generation.
```bash
# Basic image metamer (MSE)
python main.py --image inputs/dog.png --layers features.15 --iters 500

# Texture matching with TV smoothing
python main.py --image inputs/dog.png --layers features.20 --loss_type gram --tv_weight 0.05

# Non-vision tensor metamer
python main.py --image inputs/data.pt --model models/custom.pth --layers layer1 --output metamer.pt
```

### 3. Model Inspection
Identify layers and check for residual bottlenecks.
```bash
python inspect_model.py --model resnet50 --types
```

---

## Inspiration & References

This tool is heavily inspired by the work of Jenelle Feather and the McDermott Lab at MIT, whose research utilizes model metamers as a primary tool to probe the divergence between human perceptual systems and artificial neural networks.

- **Metamers of neural networks reveal divergence from human perceptual systems (Feather et al., NeurIPS 2019)**: This seminal work established the core methodology of using gradient descent to match intermediate CNN representations, demonstrating that deeper layers produce metamers that are indistinguishable to the model yet completely unrecognizable to human observers.
- **Model metamers reveal divergent invariances between biological and artificial neural networks (Feather et al., Nature Neuroscience 2023)**: This paper expanded the scope to a wide range of architectures (including Transformers and robustly trained models), quantifying how artificial invariances—such as a bias toward local texture—diverge significantly from primate visual processing.

---

## Project Structure

- `metamer_finder/`: Core package containing extraction, optimization, and GUI logic.
- `inputs/`: Directory for target images and tensors.
- `results/`: Default output directory for generated metamers and comparisons.
- `examples/`: Permanent assets for documentation.
- `gui_app.py`: Entry point for the graphical application.
- `main.py`: Entry point for the CLI.
- `inspect_model.py`: Structural analysis utility.
