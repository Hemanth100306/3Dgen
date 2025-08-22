# 2D Image to 3D Model Converter

A Python project that converts 2D images (photos, sketches) into 3D models using deep learning depth estimation and 3D reconstruction.

## Features

- **Depth Estimation**: Uses pre-trained MiDaS models (DPT_Large, DPT_Hybrid, MiDaS_small) for monocular depth estimation
- **3D Reconstruction**: Converts depth maps to point clouds and meshes using Open3D
- **Multiple Output Formats**: Saves results in PLY (point cloud) and OBJ/PLY (mesh) formats
- **GPU Acceleration**: Automatically uses CUDA if available, falls back to CPU
- **Easy to Use**: Simple Python API and command-line interface

## Requirements

- Python 3.8+
- CUDA-compatible GPU (optional, but recommended for faster processing)

## Installation

1. **Clone or download this project**
2. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## Quick Start

### Command Line Usage

```bash
# Basic usage
python image_to_3d.py input_image.jpg

# Specify output path
python image_to_3d.py input_image.jpg -o my_3d_model

# Use different model (faster but lower quality)
python image_to_3d.py input_image.jpg -m MiDaS_small

# Generate only point cloud (skip mesh)
python image_to_3d.py input_image.jpg --no-mesh

# Limit point cloud size
python image_to_3d.py input_image.jpg -p 5000
```

### Python API Usage

```python
from image_to_3d import ImageTo3DConverter

# Initialize converter
converter = ImageTo3DConverter(model_type="DPT_Large")

# Load image
image = converter.load_image("input_image.jpg")

# Estimate depth
depth = converter.estimate_depth(image)

# Generate point cloud
pcd = converter.generate_pointcloud(image, depth, max_points=10000)

# Generate mesh
mesh = converter.generate_mesh(pcd)

# Save results
converter.save_3d_model(pcd, mesh, "output")
```

### Example Script

Run the included example:
```bash
python example.py
```

## Available Models

- **DPT_Large**: Highest quality, slower processing (default)
- **DPT_Hybrid**: Good quality, medium speed
- **MiDaS_small**: Lower quality, fastest processing

## Output Files

The converter generates several output files:

- **`.ply`**: Point cloud with colors
- **`.obj`**: Triangle mesh (can be opened in Blender, MeshLab)
- **`_mesh.ply`**: Triangle mesh in PLY format

## How It Works

1. **Image Loading**: Loads and preprocesses the input image using OpenCV
2. **Depth Estimation**: Uses MiDaS deep learning model to predict depth from the 2D image
3. **Point Cloud Generation**: Converts depth map to 3D point cloud with color information
4. **Mesh Generation**: Applies Poisson surface reconstruction to create a watertight mesh
5. **Output**: Saves results in standard 3D formats

## Tips for Best Results

- **High Contrast Images**: Images with clear depth cues work best
- **Good Lighting**: Well-lit images produce more accurate depth estimates
- **Simple Scenes**: Complex scenes may have artifacts in depth estimation
- **Resolution**: Higher resolution images generally produce better results

## Troubleshooting

### Common Issues

1. **CUDA Out of Memory**: Use `MiDaS_small` model or reduce image resolution
2. **Poor Depth Quality**: Try different models or preprocess the image for better contrast
3. **Mesh Generation Fails**: Reduce point cloud size or use `--no-mesh` flag

### Performance Tips

- Use GPU acceleration when possible
- Choose appropriate model size for your needs
- Adjust `max_points` parameter based on your hardware capabilities

## Dependencies

- **OpenCV**: Image processing and loading
- **PyTorch**: Deep learning framework for MiDaS models
- **Open3D**: 3D data processing and visualization
- **NumPy**: Numerical computations
- **PIL**: Image handling

## License

This project uses open-source libraries. Please refer to individual library licenses.

## Contributing

Feel free to submit issues, feature requests, or pull requests to improve the project!
