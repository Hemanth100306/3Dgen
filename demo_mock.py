#!/usr/bin/env python3
"""
Mock demo script that demonstrates the 2D to 3D conversion project structure
without requiring MiDaS model download (which is blocked by GitHub rate limits).
This shows how the project works and what it would produce.
"""

import os
import numpy as np
import cv2
from PIL import Image


def create_mock_depth_map(image):
    """Create a mock depth map for demonstration purposes."""
    height, width = image.shape[:2]
    
    # Create a simple depth map based on image intensity
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Create depth based on distance from center (closer to center = closer in 3D)
    depth = np.zeros((height, width), dtype=np.float32)
    center_y, center_x = height // 2, width // 2
    
    for y in range(height):
        for x in range(width):
            # Distance from center
            dist = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_dist = np.sqrt(center_x**2 + center_y**2)
            
            # Normalize distance to 0-1
            normalized_dist = dist / max_dist
            
            # Invert so center is closer (smaller depth value)
            depth[y, x] = 1.0 - normalized_dist
    
    # Add some noise to make it more realistic
    noise = np.random.normal(0, 0.1, depth.shape)
    depth = np.clip(depth + noise, 0, 1)
    
    return depth


def create_mock_point_cloud(image, depth, max_points=5000):
    """Create a mock point cloud for demonstration."""
    height, width = depth.shape
    
    # Sample points
    step = max(1, int(np.sqrt(height * width / max_points)))
    
    points = []
    colors = []
    
    for y in range(0, height, step):
        for x in range(0, width, step):
            # Normalize coordinates to [-1, 1]
            x_norm = (x - width/2) / (width/2)
            y_norm = (y - height/2) / (height/2)
            z_norm = depth[y, x]
            
            points.append([x_norm, y_norm, z_norm])
            colors.append(image[y, x] / 255.0)
    
    return np.array(points), np.array(colors)


def save_mock_3d_files(points, colors, output_path):
    """Save mock 3D files in PLY format."""
    base_path = os.path.splitext(output_path)[0]
    
    # Save point cloud as PLY
    ply_path = f"{base_path}.ply"
    
    with open(ply_path, 'w') as f:
        f.write("ply\n")
        f.write("format ascii 1.0\n")
        f.write(f"element vertex {len(points)}\n")
        f.write("property float x\n")
        f.write("property float y\n")
        f.write("property float z\n")
        f.write("property uchar red\n")
        f.write("property uchar green\n")
        f.write("property uchar blue\n")
        f.write("end_header\n")
        
        for i in range(len(points)):
            x, y, z = points[i]
            r, g, b = colors[i] * 255
            f.write(f"{x:.6f} {y:.6f} {z:.6f} {int(r)} {int(g)} {int(b)}\n")
    
    print(f"✅ Saved mock point cloud: {ply_path}")
    
    # Create a simple mesh file (just for demonstration)
    obj_path = f"{base_path}.obj"
    with open(obj_path, 'w') as f:
        f.write("# Mock 3D mesh file\n")
        f.write("# This is a demonstration file\n")
        f.write(f"# Generated from {len(points)} points\n")
        f.write("\n")
        
        # Add some vertices
        for i in range(min(100, len(points))):
            x, y, z = points[i]
            f.write(f"v {x:.6f} {y:.6f} {z:.6f}\n")
        
        # Add some simple faces (triangles)
        for i in range(0, min(98, len(points)-2), 3):
            f.write(f"f {i+1} {i+2} {i+3}\n")
    
    print(f"✅ Saved mock mesh: {obj_path}")
    
    return ply_path, obj_path


def mock_demo_conversion(image_path: str = None):
    """Demonstrate the 2D to 3D conversion process with mock implementation."""
    
    print("🎯 2D to 3D Conversion Demo (Mock Version)")
    print("=" * 50)
    print("Note: This is a demonstration using mock depth estimation")
    print("      due to GitHub rate limits for MiDaS model download.")
    print("=" * 50)
    
    # If no image path provided, look for common image files
    if not image_path:
        common_images = ['test_depth_image.jpg', 'sample.jpg', 'test.png', 'image.jpg']
        for img in common_images:
            if os.path.exists(img):
                image_path = img
                break
    
    if not image_path or not os.path.exists(image_path):
        print("❌ No image file found!")
        print("\nPlease place an image file in the current directory or specify a path:")
        print("python demo_mock.py path/to/your/image.jpg")
        return False
    
    print(f"📸 Input image: {image_path}")
    print(f"📁 Current directory: {os.getcwd()}")
    
    try:
        # Step 1: Load image
        print("\n📥 Step 1: Loading image...")
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        print(f"✅ Image loaded: {image_rgb.shape[1]}x{image_rgb.shape[0]} pixels")
        
        # Step 2: Mock depth estimation
        print("\n🔍 Step 2: Estimating depth (mock)...")
        depth = create_mock_depth_map(image_rgb)
        print(f"✅ Mock depth map generated: {depth.shape}")
        
        # Step 3: Generate mock point cloud
        print("\n☁️ Step 3: Generating point cloud (mock)...")
        points, colors = create_mock_point_cloud(image_rgb, depth, max_points=8000)
        print(f"✅ Mock point cloud created with {len(points)} points")
        
        # Step 4: Save results
        print("\n💾 Step 4: Saving mock 3D model...")
        output_name = f"mock_demo_{os.path.splitext(os.path.basename(image_path))[0]}"
        ply_path, obj_path = save_mock_3d_files(points, colors, output_name)
        
        print("\n🎉 Mock demo completed successfully!")
        print(f"\n📁 Generated files:")
        print(f"   • Point cloud: {ply_path}")
        print(f"   • Mesh (OBJ): {obj_path}")
        
        print(f"\n💡 You can now:")
        print(f"   • Open .obj files in Blender, MeshLab, or other 3D viewers")
        print(f"   • View .ply files in CloudCompare or MeshLab")
        print(f"   • The real project would use MiDaS for accurate depth estimation")
        
        return True
        
    except Exception as e:
        print(f"\n❌ Error during mock conversion: {e}")
        return False


def main():
    """Main function."""
    import sys
    
    if len(sys.argv) > 1:
        image_path = sys.argv[1]
        success = mock_demo_conversion(image_path)
    else:
        success = mock_demo_conversion()
    
    if success:
        print("\n✨ Mock demo completed! Check the generated 3D files above.")
        print("\n🔧 To use the real MiDaS-based conversion:")
        print("   1. Wait for GitHub rate limits to reset")
        print("   2. Run: python demo.py your_image.jpg")
        print("   3. Or use: python image_to_3d.py your_image.jpg")
    else:
        print("\n💡 Try running: python test_installation.py to check your setup")
    
    return 0 if success else 1


if __name__ == "__main__":
    exit(main())
