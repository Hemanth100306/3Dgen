#!/usr/bin/env python3
"""
Professional 3D Viewer for 2D to 3D Conversion Results
Features:
- 360° rotation with mouse/touch controls
- Real-time 3D rendering with professional quality
- Interactive zoom, pan, and rotation
- Support for both point clouds and meshes
- Professional lighting and materials
"""

import open3d as o3d
import numpy as np
import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import os
import threading
import time


class Professional3DViewer:
    def __init__(self, root):
        self.root = root
        self.root.title("Professional 3D Viewer - 2D to 3D Converter")
        self.root.geometry("1400x900")
        self.root.configure(bg='#1e1e1e')
        
        # 3D visualization variables
        self.vis = None
        self.current_geometry = None
        self.is_viewer_running = False
        
        # Create main interface
        self.create_interface()
        
        # Start 3D viewer in separate thread
        self.start_3d_viewer()
    
    def create_interface(self):
        """Create the main interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎯 Professional 3D Viewer", 
                               font=('Arial', 20, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Control panel
        control_frame = ttk.LabelFrame(main_frame, text="3D Model Controls", padding=20)
        control_frame.pack(fill=tk.X, pady=(0, 20))
        
        # File loading
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(file_frame, text="Load 3D Model:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="📁 Browse", command=self.browse_3d_file)
        browse_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        load_btn = ttk.Button(file_frame, text="🚀 Load & View", command=self.load_and_view_3d)
        load_btn.pack(side=tk.LEFT)
        
        # Quick load buttons
        quick_frame = ttk.Frame(control_frame)
        quick_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(quick_frame, text="Quick Load:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Find recent 3D files
        recent_files = self.find_recent_3d_files()
        for i, file_path in enumerate(recent_files[:3]):  # Show last 3 files
            btn = ttk.Button(quick_frame, text=f"📐 {os.path.basename(file_path)}", 
                           command=lambda f=file_path: self.quick_load_3d(f))
            btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Viewer controls
        viewer_frame = ttk.LabelFrame(main_frame, text="3D Viewer Controls", padding=20)
        viewer_frame.pack(fill=tk.X, pady=(0, 20))
        
        # Control instructions
        instructions = """
🎮 3D Viewer Controls:
• 🖱️ Left Mouse: Rotate camera around model
• 🖱️ Right Mouse: Pan camera
• 🖱️ Mouse Wheel: Zoom in/out
• ⌨️ R: Reset view
• ⌨️ F: Fit model to view
• ⌨️ L: Toggle lighting
• ⌨️ W: Toggle wireframe
        """
        
        instruction_label = ttk.Label(viewer_frame, text=instructions, 
                                    font=('Consolas', 10), justify=tk.LEFT)
        instruction_label.pack(anchor=tk.W)
        
        # Viewer status
        status_frame = ttk.Frame(main_frame)
        status_frame.pack(fill=tk.X)
        
        self.status_label = ttk.Label(status_frame, text="3D Viewer: Starting...", 
                                     font=('Arial', 12))
        self.status_label.pack(side=tk.LEFT)
        
        # Progress bar
        self.progress = ttk.Progressbar(status_frame, mode='indeterminate')
        self.progress.pack(side=tk.RIGHT, fill=tk.X, expand=True, padx=(20, 0))
    
    def find_recent_3d_files(self):
        """Find recent 3D files in the current directory."""
        extensions = ['.ply', '.obj', '.stl', '.off']
        files = []
        
        for file in os.listdir('.'):
            if any(file.lower().endswith(ext) for ext in extensions):
                files.append(file)
        
        # Sort by modification time (newest first)
        files.sort(key=lambda x: os.path.getmtime(x), reverse=True)
        return files
    
    def browse_3d_file(self):
        """Browse for 3D model files."""
        file_types = [
            ('3D Model files', '*.ply *.obj *.stl *.off'),
            ('Point Cloud (PLY)', '*.ply'),
            ('Mesh (OBJ)', '*.obj'),
            ('Mesh (STL)', '*.stl'),
            ('Mesh (OFF)', '*.off'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select 3D Model",
            filetypes=file_types
        )
        
        if filename:
            self.file_path_var.set(filename)
    
    def load_and_view_3d(self):
        """Load and display 3D model."""
        file_path = self.file_path_var.get()
        
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid 3D model file")
            return
        
        # Start loading in separate thread
        self.progress.start()
        self.status_label.config(text="Loading 3D model...")
        
        thread = threading.Thread(target=self._load_3d_thread, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def quick_load_3d(self, file_path):
        """Quick load a 3D file."""
        self.file_path_var.set(file_path)
        self.load_and_view_3d()
    
    def _load_3d_thread(self, file_path):
        """Load 3D model in separate thread."""
        try:
            # Load geometry based on file type
            if file_path.lower().endswith('.ply'):
                geometry = o3d.io.read_point_cloud(file_path)
                if len(geometry.points) == 0:  # Try as mesh if point cloud is empty
                    geometry = o3d.io.read_triangle_mesh(file_path)
            elif file_path.lower().endswith('.obj'):
                geometry = o3d.io.read_triangle_mesh(file_path)
            elif file_path.lower().endswith('.stl'):
                geometry = o3d.io.read_triangle_mesh(file_path)
            elif file_path.lower().endswith('.off'):
                geometry = o3d.io.read_triangle_mesh(file_path)
            else:
                raise ValueError(f"Unsupported file format: {file_path}")
            
            # Update viewer with new geometry
            self.root.after(0, self._update_viewer, geometry, file_path)
            
        except Exception as e:
            self.root.after(0, self._load_error, str(e))
    
    def _update_viewer(self, geometry, file_path):
        """Update the 3D viewer with new geometry."""
        self.progress.stop()
        
        if self.vis and self.is_viewer_running:
            # Clear current geometry
            self.vis.clear_geometries()
            
            # Add new geometry
            self.vis.add_geometry(geometry)
            
            # Fit view to geometry
            self.vis.reset_view_point(True)
            
            # Update status
            self.status_label.config(text=f"✅ Loaded: {os.path.basename(file_path)}")
            
            # Show success message
            if hasattr(geometry, 'points'):
                point_count = len(geometry.points)
                messagebox.showinfo("Success", 
                                  f"3D Model Loaded Successfully!\n\n"
                                  f"File: {os.path.basename(file_path)}\n"
                                  f"Type: Point Cloud\n"
                                  f"Points: {point_count:,}\n\n"
                                  f"Use mouse to rotate, zoom, and pan!")
            else:
                vertex_count = len(geometry.vertices)
                face_count = len(geometry.triangles)
                messagebox.showinfo("Success", 
                                  f"3D Model Loaded Successfully!\n\n"
                                  f"File: {os.path.basename(file_path)}\n"
                                  f"Type: Triangle Mesh\n"
                                  f"Vertices: {vertex_count:,}\n"
                                  f"Faces: {face_count:,}\n\n"
                                  f"Use mouse to rotate, zoom, and pan!")
        else:
            self.status_label.config(text="❌ 3D Viewer not ready")
    
    def _load_error(self, error_msg):
        """Handle 3D model loading error."""
        self.progress.stop()
        self.status_label.config(text=f"❌ Failed to load 3D model: {error_msg}")
        messagebox.showerror("Error", f"Failed to load 3D model:\n{error_msg}")
    
    def start_3d_viewer(self):
        """Start the 3D viewer in a separate thread."""
        def viewer_thread():
            try:
                # Create Open3D visualizer
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(
                    window_name="Professional 3D Viewer",
                    width=1200,
                    height=800,
                    left=50,
                    top=50,
                    visible=True
                )
                
                # Set professional rendering options
                opt = self.vis.get_render_option()
                opt.background_color = np.asarray([0.1, 0.1, 0.1])  # Dark background
                opt.point_size = 2.0
                opt.line_width = 2.0
                opt.show_coordinate_frame = True
                opt.light_on = True
                
                # Set professional view control
                vc = self.vis.get_view_control()
                vc.set_zoom(0.7)
                
                # Update status
                self.root.after(0, lambda: self.status_label.config(text="3D Viewer: Ready"))
                self.is_viewer_running = True
                
                # Run viewer loop
                self.vis.run()
                
                # Cleanup when viewer closes
                self.vis.destroy_window()
                self.is_viewer_running = False
                
            except Exception as e:
                self.root.after(0, lambda: self.status_label.config(text=f"3D Viewer Error: {e}"))
        
        # Start viewer thread
        thread = threading.Thread(target=viewer_thread)
        thread.daemon = True
        thread.start()
    
    def on_closing(self):
        """Handle application closing."""
        if self.vis and self.is_viewer_running:
            self.vis.close()
        self.root.destroy()


def main():
    """Main function."""
    root = tk.Tk()
    app = Professional3DViewer(root)
    
    # Set closing handler
    root.protocol("WM_DELETE_WINDOW", app.on_closing)
    
    # Center window
    root.update_idletasks()
    width = root.winfo_width()
    height = root.winfo_height()
    x = (root.winfo_screenwidth() // 2) - (width // 2)
    y = (root.winfo_screenheight() // 2) - (height // 2)
    root.geometry(f"{width}x{height}+{x}+{y}")
    
    # Start GUI
    root.mainloop()


if __name__ == "__main__":
    main()
