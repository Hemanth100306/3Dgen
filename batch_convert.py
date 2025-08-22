#!/usr/bin/env python3
"""
Batch conversion script for multiple images.
Converts all images in a directory to 3D models.
"""

import os
import glob
from pathlib import Path
from image_to_3d import ImageTo3DConverter
import argparse


def get_image_files(input_dir: str) -> list:
    """Get all image files from a directory."""
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp', '*.tiff', '*.tif']
    image_files = []
    
    for ext in image_extensions:
        image_files.extend(glob.glob(os.path.join(input_dir, ext)))
        image_files.extend(glob.glob(os.path.join(input_dir, ext.upper())))
    
    return sorted(image_files)


def batch_convert(input_dir: str, output_dir: str, model_type: str = "DPT_Large", 
                  max_points: int = 10000, skip_mesh: bool = False):
    """Convert all images in a directory to 3D models."""
    
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)
    
    # Get all image files
    image_files = get_image_files(input_dir)
    
    if not image_files:
        print(f"No image files found in {input_dir}")
        return
    
    print(f"Found {len(image_files)} image files to convert")
    print(f"Output directory: {output_dir}")
    print(f"Model: {model_type}")
    print(f"Max points: {max_points}")
    print(f"Skip mesh: {skip_mesh}")
    print("-" * 50)
    
    # Initialize converter
    converter = ImageTo3DConverter(model_type=model_type)
    
    successful = 0
    failed = 0
    
    for i, image_path in enumerate(image_files, 1):
        try:
            print(f"\n[{i}/{len(image_files)}] Processing: {os.path.basename(image_path)}")
            
            # Generate output filename
            base_name = Path(image_path).stem
            output_path = os.path.join(output_dir, base_name)
            
            # Load image
            image = converter.load_image(image_path)
            
            # Estimate depth
            depth = converter.estimate_depth(image)
            
            # Generate point cloud
            pcd = converter.generate_pointcloud(image, depth, max_points)
            
            # Generate mesh (optional)
            mesh = None
            if not skip_mesh:
                mesh = converter.generate_mesh(pcd)
            
            # Save results
            converter.save_3d_model(pcd, mesh, output_path, save_mesh=not skip_mesh)
            
            print(f"✅ Successfully converted: {base_name}")
            successful += 1
            
        except Exception as e:
            print(f"❌ Failed to convert {os.path.basename(image_path)}: {e}")
            failed += 1
    
    print("\n" + "=" * 50)
    print(f"Batch conversion completed!")
    print(f"Successful: {successful}")
    print(f"Failed: {failed}")
    print(f"Total: {len(image_files)}")
    
    if successful > 0:
        print(f"\nOutput files saved in: {output_dir}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Batch convert multiple images to 3D models")
    parser.add_argument("input_dir", help="Directory containing input images")
    parser.add_argument("--output", "-o", default="batch_output", help="Output directory")
    parser.add_argument("--model", "-m", default="DPT_Large", 
                       choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                       help="MiDaS model to use")
    parser.add_argument("--max-points", "-p", type=int, default=10000,
                       help="Maximum number of points in point cloud")
    parser.add_argument("--skip-mesh", action="store_true", help="Skip mesh generation")
    
    args = parser.parse_args()
    
    if not os.path.isdir(args.input_dir):
        print(f"Error: Input directory '{args.input_dir}' does not exist")
        return 1
    
    try:
        batch_convert(
            input_dir=args.input_dir,
            output_dir=args.output,
            model_type=args.model,
            max_points=args.max_points,
            skip_mesh=args.skip_mesh
        )
    except KeyboardInterrupt:
        print("\nBatch conversion interrupted by user")
        return 1
    except Exception as e:
        print(f"Error during batch conversion: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
