#!/usr/bin/env python3
"""
Demo script for the 2D to 3D converter.
This script demonstrates the complete workflow with better error handling.
"""

import os
import sys
from image_to_3d import ImageTo3DConverter


def demo_conversion(image_path: str = None):
    """Demonstrate the 2D to 3D conversion process."""
    
    print("🎯 2D to 3D Conversion Demo")
    print("=" * 40)
    
    # If no image path provided, look for common image files
    if not image_path:
        common_images = ['sample.jpg', 'test.png', 'image.jpg', 'photo.jpg']
        for img in common_images:
            if os.path.exists(img):
                image_path = img
                break
    
    if not image_path or not os.path.exists(image_path):
        print("❌ No image file found!")
        print("\nPlease place an image file in the current directory or specify a path:")
        print("python demo.py path/to/your/image.jpg")
        print("\nSupported formats: JPG, PNG, BMP, TIFF")
        return False
    
    print(f"📸 Input image: {image_path}")
    print(f"📁 Current directory: {os.getcwd()}")
    
    try:
        # Step 1: Initialize converter
        print("\n🚀 Step 1: Initializing converter...")
        converter = ImageTo3DConverter(model_type="DPT_Large")
        print("✅ Converter initialized successfully")
        
        # Step 2: Load image
        print("\n📥 Step 2: Loading image...")
        image = converter.load_image(image_path)
        print(f"✅ Image loaded: {image.shape[1]}x{image.shape[0]} pixels")
        
        # Step 3: Estimate depth
        print("\n🔍 Step 3: Estimating depth...")
        depth = converter.estimate_depth(image)
        print(f"✅ Depth map generated: {depth.shape}")
        
        # Step 4: Generate point cloud
        print("\n☁️ Step 4: Generating point cloud...")
        pcd = converter.generate_pointcloud(image, depth, max_points=12000)
        print(f"✅ Point cloud created with {len(pcd.points)} points")
        
        # Step 5: Generate mesh
        print("\n🔲 Step 5: Generating mesh...")
        mesh = converter.generate_mesh(pcd)
        print(f"✅ Mesh created with {len(mesh.vertices)} vertices")
        
        # Step 6: Save results
        print("\n💾 Step 6: Saving 3D model...")
        output_name = f"demo_{os.path.splitext(os.path.basename(image_path))[0]}"
        converter.save_3d_model(pcd, mesh, output_name)
        
        print("\n🎉 Demo completed successfully!")
        print(f"\n📁 Generated files:")
        print(f"   • Point cloud: {output_name}.ply")
        print(f"   • Mesh (OBJ): {output_name}.obj")
        print(f"   • Mesh (PLY): {output_name}_mesh.ply")
        
        print(f"\n💡 You can now:")
        print(f"   • Open .obj files in Blender, MeshLab, or other 3D viewers")
        print(f"   • View .ply files in CloudCompare or MeshLab")
        print(f"   • Use the generated models for 3D printing or visualization")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during conversion: {e}")
        print("\n🔧 Troubleshooting tips:")
        print("   • Make sure all dependencies are installed: pip install -r requirements.txt")
        print("   • Check if the image file is corrupted or unsupported")
        print("   • Try with a different image or use MiDaS_small model for faster processing")
        return False


def main():
    """Main function."""
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        success = demo_conversion(image_path)
    else:
        success = demo_conversion()
    
    if success:
        print("\n✨ Demo completed! Check the generated 3D files above.")
    else:
        print("\n💡 Try running: python test_installation.py to check your setup")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
