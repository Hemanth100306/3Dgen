import cv2
import torch
import torchvision.transforms as transforms
import numpy as np
import open3d as o3d
from PIL import Image
import os
import argparse
from typing import Tuple, Optional


class ImageTo3DConverter:
    def __init__(self, model_type: str = "DPT_Large"):
        """
        Initialize the ImageTo3DConverter with a pre-trained depth estimation model.
        
        Args:
            model_type: Type of MiDaS model to use (DPT_Large, DPT_Hybrid, MiDaS_small)
        """
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")
        
        # Load MiDaS model
        self.midas = torch.hub.load("intel-isl/MiDaS", model_type)
        self.midas.to(self.device)
        self.midas.eval()
        
        # Define transforms
        self.transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        # Reverse transform for depth map
        self.reverse_transform = transforms.Compose([
            transforms.Resize((384, 384)),
            transforms.ToTensor()
        ])

    def load_image(self, image_path: str) -> np.ndarray:
        """
        Load and preprocess an image for depth estimation.
        
        Args:
            image_path: Path to the input image
            
        Returns:
            Preprocessed image as numpy array
        """
        if not os.path.exists(image_path):
            raise FileNotFoundError(f"Image not found: {image_path}")
        
        # Load image
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"Could not load image: {image_path}")
        
        # Convert BGR to RGB
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        print(f"Loaded image: {image.shape[1]}x{image.shape[0]}")
        return image_rgb

    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from 2D image using MiDaS.
        
        Args:
            image: Input RGB image as numpy array
            
        Returns:
            Depth map as numpy array
        """
        # Convert to PIL Image
        pil_image = Image.fromarray(image)
        
        # Apply transforms
        input_batch = self.transform(pil_image).unsqueeze(0).to(self.device)
        
        # Predict depth
        with torch.no_grad():
            prediction = self.midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=image.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()
            
            depth = prediction.cpu().numpy()
        
        # Normalize depth to 0-1 range
        depth = (depth - depth.min()) / (depth.max() - depth.min())
        
        print(f"Generated depth map: {depth.shape}")
        return depth

    def generate_pointcloud(self, image: np.ndarray, depth: np.ndarray, 
                           max_points: int = 10000) -> o3d.geometry.PointCloud:
        """
        Generate 3D point cloud from image and depth map.
        
        Args:
            image: Input RGB image
            depth: Depth map
            max_points: Maximum number of points to generate
            
        Returns:
            Open3D PointCloud object
        """
        height, width = depth.shape
        
        # Create coordinate grids
        x_coords, y_coords = np.meshgrid(np.arange(width), np.arange(height))
        
        # Normalize coordinates to [-1, 1]
        x_norm = (x_coords - width/2) / (width/2)
        y_norm = (y_coords - height/2) / (height/2)
        
        # Use depth as Z coordinate (inverted for better visualization)
        z_norm = 1.0 - depth
        
        # Stack coordinates
        coords = np.stack([x_norm, y_norm, z_norm], axis=-1)
        
        # Sample points if too many
        if height * width > max_points:
            step = int(np.sqrt(height * width / max_points))
            coords = coords[::step, ::step]
            image_sampled = image[::step, ::step]
        else:
            image_sampled = image
        
        # Reshape for point cloud
        points = coords.reshape(-1, 3)
        colors = image_sampled.reshape(-1, 3) / 255.0
        
        # Create Open3D point cloud
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        pcd.colors = o3d.utility.Vector3dVector(colors)
        
        # Remove outliers
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        print(f"Generated point cloud with {len(pcd.points)} points")
        return pcd

    def generate_mesh(self, pcd: o3d.geometry.PointCloud) -> o3d.geometry.TriangleMesh:
        """
        Generate mesh from point cloud using Poisson surface reconstruction.
        
        Args:
            pcd: Input point cloud
            
        Returns:
            Open3D TriangleMesh object
        """
        # Estimate normals
        pcd.estimate_normals(search_param=o3d.geometry.KDTreeSearchParamHybrid(radius=0.1, max_nn=30))
        
        # Poisson surface reconstruction
        mesh, densities = o3d.geometry.TriangleMesh.create_from_point_cloud_poisson(pcd, depth=8)
        
        # Remove low density vertices
        vertices_to_remove = densities < np.quantile(densities, 0.1)
        mesh.remove_vertices_by_mask(vertices_to_remove)
        
        print(f"Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
        return mesh

    def save_3d_model(self, pcd: o3d.geometry.PointCloud, mesh: o3d.geometry.TriangleMesh, 
                      output_path: str, save_mesh: bool = True):
        """
        Save 3D model in multiple formats.
        
        Args:
            pcd: Point cloud to save
            mesh: Mesh to save
            output_path: Base path for output files
            save_mesh: Whether to save mesh files
        """
        base_path = os.path.splitext(output_path)[0]
        
        # Save point cloud
        pcd_path = f"{base_path}.ply"
        o3d.io.write_point_cloud(pcd_path, pcd)
        print(f"Saved point cloud: {pcd_path}")
        
        if save_mesh:
            # Save mesh
            mesh_path = f"{base_path}.obj"
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"Saved mesh: {mesh_path}")
            
            # Also save as PLY for compatibility
            mesh_ply_path = f"{base_path}_mesh.ply"
            o3d.io.write_triangle_mesh(mesh_ply_path, mesh)
            print(f"Saved mesh (PLY): {mesh_ply_path}")


def main():
    """Main function for command line usage."""
    parser = argparse.ArgumentParser(description="Convert 2D image to 3D model")
    parser.add_argument("input_image", help="Path to input image")
    parser.add_argument("--output", "-o", default="output", help="Output file path (without extension)")
    parser.add_argument("--model", "-m", default="DPT_Large", 
                       choices=["DPT_Large", "DPT_Hybrid", "MiDaS_small"],
                       help="MiDaS model to use")
    parser.add_argument("--max-points", "-p", type=int, default=10000,
                       help="Maximum number of points in point cloud")
    parser.add_argument("--no-mesh", action="store_true", help="Skip mesh generation")
    
    args = parser.parse_args()
    
    try:
        # Initialize converter
        converter = ImageTo3DConverter(model_type=args.model)
        
        # Load image
        image = converter.load_image(args.input_image)
        
        # Estimate depth
        depth = converter.estimate_depth(image)
        
        # Generate point cloud
        pcd = converter.generate_pointcloud(image, depth, args.max_points)
        
        # Generate mesh (optional)
        mesh = None
        if not args.no_mesh:
            mesh = converter.generate_mesh(pcd)
        
        # Save results
        converter.save_3d_model(pcd, mesh, args.output, save_mesh=not args.no_mesh)
        
        print("Conversion completed successfully!")
        
    except Exception as e:
        print(f"Error: {e}")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
