#!/usr/bin/env python3
"""
Example usage of the ImageTo3DConverter class.
This script demonstrates how to convert a 2D image to a 3D model.
"""

from image_to_3d import ImageTo3DConverter
import os


def main():
    """Example usage of the ImageTo3DConverter."""
    
    # Check if we have an input image
    input_image = "sample_image.jpg"  # Change this to your image path
    
    if not os.path.exists(input_image):
        print(f"Please place an image file named '{input_image}' in the current directory")
        print("Or modify the 'input_image' variable in this script to point to your image")
        return
    
    print("=== 2D to 3D Conversion Example ===")
    print(f"Input image: {input_image}")
    
    try:
        # Initialize converter with DPT_Large model (best quality)
        print("\n1. Initializing converter...")
        converter = ImageTo3DConverter(model_type="DPT_Large")
        
        # Load and preprocess image
        print("\n2. Loading image...")
        image = converter.load_image(input_image)
        
        # Estimate depth
        print("\n3. Estimating depth...")
        depth = converter.estimate_depth(image)
        
        # Generate point cloud
        print("\n4. Generating point cloud...")
        pcd = converter.generate_pointcloud(image, depth, max_points=15000)
        
        # Generate mesh
        print("\n5. Generating mesh...")
        mesh = converter.generate_mesh(pcd)
        
        # Save results
        print("\n6. Saving 3D model...")
        output_path = "example_output"
        converter.save_3d_model(pcd, mesh, output_path)
        
        print(f"\n✅ Conversion completed successfully!")
        print(f"📁 Output files:")
        print(f"   - Point cloud: {output_path}.ply")
        print(f"   - Mesh (OBJ): {output_path}.obj")
        print(f"   - Mesh (PLY): {output_path}_mesh.ply")
        print(f"\n💡 You can now open these files in Blender, MeshLab, or other 3D viewers!")
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        print("Make sure you have all dependencies installed:")
        print("pip install -r requirements.txt")


if __name__ == "__main__":
    main()
