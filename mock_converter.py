#!/usr/bin/env python3
"""
Mock converter class that provides the same interface as ImageTo3DConverter
but uses simulated depth estimation for demonstration purposes.
"""

import cv2
import numpy as np
import open3d as o3d
import os


class MockImageTo3DConverter:
    """Mock converter that simulates 2D to 3D conversion."""
    
    def __init__(self, model_type: str = "DPT_Large"):
        """
        Initialize the mock converter.
        
        Args:
            model_type: Type of model (ignored in mock version)
        """
        print(f"⚠️ Using Mock Converter (simulated depth estimation)")
        print(f"   Model type: {model_type}")
        print(f"   This provides the same interface as the real converter")
    
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
        
        print(f"✅ Loaded image: {image.shape[1]}x{image.shape[0]}")
        return image_rgb
    
    def estimate_depth(self, image: np.ndarray) -> np.ndarray:
        """
        Estimate depth map from 2D image using simulated depth estimation.
        
        Args:
            image: Input RGB image as numpy array
            
        Returns:
            Simulated depth map as numpy array
        """
        height, width = image.shape[:2]
        
        # Create a simulated depth map based on image features
        depth = np.zeros((height, width), dtype=np.float32)
        
        # Convert to grayscale for intensity analysis
        gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        
        # Create depth based on distance from center (closer to center = closer in 3D)
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
        
        # Add some depth variation based on image intensity
        intensity_factor = gray.astype(np.float32) / 255.0
        depth = depth * 0.7 + intensity_factor * 0.3
        
        # Add some noise to make it more realistic
        noise = np.random.normal(0, 0.05, depth.shape)
        depth = np.clip(depth + noise, 0, 1)
        
        print(f"✅ Generated simulated depth map: {depth.shape}")
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
        
        # Remove outliers for cleaner visualization
        pcd, _ = pcd.remove_statistical_outlier(nb_neighbors=20, std_ratio=2.0)
        
        print(f"✅ Generated point cloud with {len(pcd.points)} points")
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
        
        print(f"✅ Generated mesh with {len(mesh.vertices)} vertices and {len(mesh.triangles)} triangles")
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
        print(f"✅ Saved point cloud: {pcd_path}")
        
        if save_mesh:
            # Save mesh
            mesh_path = f"{base_path}.obj"
            o3d.io.write_triangle_mesh(mesh_path, mesh)
            print(f"✅ Saved mesh: {mesh_path}")
            
            # Also save as PLY for compatibility
            mesh_ply_path = f"{base_path}_mesh.ply"
            o3d.io.write_triangle_mesh(mesh_ply_path, mesh)
            print(f"✅ Saved mesh (PLY): {mesh_ply_path}")


# Create a fallback function for when the real converter fails
def create_converter(model_type: str = "DPT_Large"):
    """
    Create a converter instance, falling back to mock if real converter fails.
    
    Args:
        model_type: Type of model to use
        
    Returns:
        Converter instance (real or mock)
    """
    try:
        from image_to_3d import ImageTo3DConverter
        return ImageTo3DConverter(model_type=model_type)
    except Exception as e:
        print(f"⚠️ Real converter failed: {e}")
        print("🔄 Falling back to mock converter...")
        return MockImageTo3DConverter(model_type=model_type)
