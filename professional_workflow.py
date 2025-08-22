#!/usr/bin/env python3
"""
Professional Integrated Workflow: 2D to 3D Conversion + Real-time 3D Viewer
Features:
- Upload 2D images or capture from camera
- Convert to 3D models in real-time
- Professional 3D viewer with 360° rotation
- Seamless workflow from image to 3D visualization
- Professional quality rendering and controls
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import threading
from PIL import Image, ImageTk
import time
import open3d as o3d
from mock_converter import create_converter


class ProfessionalWorkflow:
    def __init__(self, root):
        self.root = root
        self.root.title("🎯 Professional 2D to 3D Workflow")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')
        
        # Initialize variables
        self.camera = None
        self.is_camera_on = False
        self.current_frame = None
        self.converter = None
        self.vis = None
        self.is_viewer_running = False
        
        # Initialize converter
        self.initialize_converter()
        
        # Create interface
        self.create_interface()
        
        # Start 3D viewer
        self.start_3d_viewer()
    
    def initialize_converter(self):
        """Initialize the 3D converter with fallback."""
        try:
            self.converter = create_converter("DPT_Large")
            print("✅ Converter initialized successfully")
        except Exception as e:
            print(f"⚠️ Using mock converter due to: {e}")
            self.converter = None
    
    def create_interface(self):
        """Create the main interface."""
        # Main container
        main_frame = ttk.Frame(self.root)
        main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Title
        title_label = ttk.Label(main_frame, text="🎯 Professional 2D to 3D Workflow", 
                               font=('Arial', 24, 'bold'))
        title_label.pack(pady=(0, 20))
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_upload_tab()
        self.create_camera_tab()
        self.create_3d_viewer_tab()
        
        # Status bar
        self.create_status_bar(main_frame)
    
    def create_upload_tab(self):
        """Create the file upload tab."""
        upload_frame = ttk.Frame(self.notebook)
        self.notebook.add(upload_frame, text="📁 Upload & Convert")
        
        # Title
        title_label = ttk.Label(upload_frame, text="Upload 2D Image and Convert to 3D", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(pady=20)
        
        # File selection frame
        file_frame = ttk.Frame(upload_frame)
        file_frame.pack(pady=20, padx=20, fill=tk.X)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="📁 Browse", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        # Image preview frame
        preview_frame = ttk.LabelFrame(upload_frame, text="Image Preview", padding=20)
        preview_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        self.preview_label = ttk.Label(preview_frame, text="No image selected")
        self.preview_label.pack(expand=True)
        
        # Control buttons
        button_frame = ttk.Frame(upload_frame)
        button_frame.pack(pady=20)
        
        convert_btn = ttk.Button(button_frame, text="🔄 Convert to 3D", 
                                command=self.convert_uploaded_image, style='Accent.TButton')
        convert_btn.pack(side=tk.LEFT, padx=10)
        
        view_3d_btn = ttk.Button(button_frame, text="👁️ View in 3D", 
                                 command=self.view_latest_3d, state=tk.DISABLED)
        view_3d_btn.pack(side=tk.LEFT, padx=10)
        
        clear_btn = ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_upload)
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Progress bar
        self.upload_progress = ttk.Progressbar(upload_frame, mode='indeterminate')
        self.upload_progress.pack(pady=10, fill=tk.X, padx=20)
        
        # Status label
        self.upload_status = ttk.Label(upload_frame, text="Ready to convert images", 
                                      font=('Arial', 12))
        self.upload_status.pack(pady=10)
    
    def create_camera_tab(self):
        """Create the live camera tab."""
        camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(camera_frame, text="📷 Live Camera")
        
        # Title
        title_label = ttk.Label(camera_frame, text="Live Camera Capture for 3D Generation", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(pady=20)
        
        # Camera controls
        control_frame = ttk.Frame(camera_frame)
        control_frame.pack(pady=20)
        
        self.camera_btn = ttk.Button(control_frame, text="📹 Start Camera", 
                                     command=self.toggle_camera)
        self.camera_btn.pack(side=tk.LEFT, padx=10)
        
        self.capture_btn = ttk.Button(control_frame, text="📸 Capture & Convert", 
                                      command=self.capture_and_convert, state=tk.DISABLED)
        self.capture_btn.pack(side=tk.LEFT, padx=10)
        
        self.view_3d_camera_btn = ttk.Button(control_frame, text="👁️ View in 3D", 
                                             command=self.view_latest_3d, state=tk.DISABLED)
        self.view_3d_camera_btn.pack(side=tk.LEFT, padx=10)
        
        # Camera preview frame
        camera_preview_frame = ttk.LabelFrame(camera_frame, text="Camera Preview", padding=20)
        camera_preview_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        self.camera_label = ttk.Label(camera_preview_frame, text="Camera not started")
        self.camera_label.pack(expand=True)
        
        # Camera status
        self.camera_status = ttk.Label(camera_frame, text="Camera: Off", 
                                     font=('Arial', 12))
        self.camera_status.pack(pady=10)
        
        # Progress bar for camera
        self.camera_progress = ttk.Progressbar(camera_frame, mode='indeterminate')
        self.camera_progress.pack(pady=10, fill=tk.X, padx=20)
    
    def create_3d_viewer_tab(self):
        """Create the 3D viewer tab."""
        viewer_frame = ttk.Frame(self.notebook)
        self.notebook.add(viewer_frame, text="👁️ 3D Viewer")
        
        # Title
        title_label = ttk.Label(viewer_frame, text="Professional 3D Model Viewer", 
                               font=('Arial', 18, 'bold'))
        title_label.pack(pady=20)
        
        # 3D viewer controls
        control_frame = ttk.LabelFrame(viewer_frame, text="3D Model Controls", padding=20)
        control_frame.pack(fill=tk.X, pady=20, padx=20)
        
        # File loading
        file_frame = ttk.Frame(control_frame)
        file_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(file_frame, text="Load 3D Model:").pack(side=tk.LEFT, padx=(0, 10))
        
        self.viewer_file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.viewer_file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="📁 Browse", command=self.browse_3d_file)
        browse_btn.pack(side=tk.LEFT, padx=(0, 10))
        
        load_btn = ttk.Button(file_frame, text="🚀 Load & View", command=self.load_and_view_3d)
        load_btn.pack(side=tk.LEFT)
        
        # Quick load buttons for recent files
        quick_frame = ttk.Frame(control_frame)
        quick_frame.pack(fill=tk.X, pady=(0, 15))
        
        ttk.Label(quick_frame, text="Quick Load:").pack(side=tk.LEFT, padx=(0, 10))
        
        # Find recent 3D files
        recent_files = self.find_recent_3d_files()
        for i, file_path in enumerate(recent_files[:3]):  # Show last 3 files
            btn = ttk.Button(quick_frame, text=f"📐 {os.path.basename(file_path)}", 
                           command=lambda f=file_path: self.quick_load_3d(f))
            btn.pack(side=tk.LEFT, padx=(0, 5))
        
        # Viewer instructions
        instruction_frame = ttk.LabelFrame(viewer_frame, text="3D Viewer Controls", padding=20)
        instruction_frame.pack(fill=tk.X, pady=20, padx=20)
        
        instructions = """
🎮 Professional 3D Viewer Controls:
• 🖱️ Left Mouse: Rotate camera around model (360° view)
• 🖱️ Right Mouse: Pan camera
• 🖱️ Mouse Wheel: Zoom in/out
• ⌨️ R: Reset view
• ⌨️ F: Fit model to view
• ⌨️ L: Toggle professional lighting
• ⌨️ W: Toggle wireframe mode
• ⌨️ C: Toggle coordinate frame
        """
        
        instruction_label = ttk.Label(instruction_frame, text=instructions, 
                                    font=('Consolas', 10), justify=tk.LEFT)
        instruction_label.pack(anchor=tk.W)
        
        # Viewer status
        self.viewer_status = ttk.Label(viewer_frame, text="3D Viewer: Starting...", 
                                      font=('Arial', 12))
        self.viewer_status.pack(pady=10)
        
        # Progress bar
        self.viewer_progress = ttk.Progressbar(viewer_frame, mode='indeterminate')
        self.viewer_progress.pack(pady=10, fill=tk.X, padx=20)
    
    def create_status_bar(self, parent):
        """Create status bar."""
        status_frame = ttk.Frame(parent)
        status_frame.pack(fill=tk.X, pady=(20, 0))
        
        self.main_status = ttk.Label(status_frame, text="Ready", font=('Arial', 10))
        self.main_status.pack(side=tk.LEFT)
        
        # Workflow status
        workflow_status = ttk.Label(status_frame, text="Workflow: 2D → 3D → View", 
                                   font=('Arial', 10, 'bold'))
        workflow_status.pack(side=tk.RIGHT)
    
    def browse_file(self):
        """Browse for image files."""
        file_types = [
            ('Image files', '*.jpg *.jpeg *.png *.bmp *.tiff *.tif'),
            ('JPEG files', '*.jpg *.jpeg'),
            ('PNG files', '*.png'),
            ('All files', '*.*')
        ]
        
        filename = filedialog.askopenfilename(
            title="Select 2D Image",
            filetypes=file_types
        )
        
        if filename:
            self.file_path_var.set(filename)
            self.preview_image(filename)
    
    def preview_image(self, image_path):
        """Preview the selected image."""
        try:
            image = cv2.imread(image_path)
            if image is not None:
                # Resize for preview
                height, width = image.shape[:2]
                max_size = 400
                if height > max_size or width > max_size:
                    scale = max_size / max(height, width)
                    new_width = int(width * scale)
                    new_height = int(height * scale)
                    image = cv2.resize(image, (new_width, new_height))
                
                # Convert BGR to RGB
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                
                # Convert to PIL Image and then to PhotoImage
                pil_image = Image.fromarray(image_rgb)
                photo = ImageTk.PhotoImage(pil_image)
                
                # Update preview
                self.preview_label.configure(image=photo, text="")
                self.preview_label.image = photo
                
                self.upload_status.config(text=f"Image loaded: {os.path.basename(image_path)}")
            else:
                self.preview_label.config(text="Failed to load image")
                self.upload_status.config(text="Error: Could not load image")
        except Exception as e:
            self.preview_label.config(text=f"Error: {str(e)}")
            self.upload_status.config(text="Error loading image")
    
    def convert_uploaded_image(self):
        """Convert the uploaded image to 3D."""
        image_path = self.file_path_var.get()
        
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
        
        # Start conversion
        self.upload_progress.start()
        self.upload_status.config(text="Converting image to 3D...")
        
        thread = threading.Thread(target=self._convert_image_thread, args=(image_path,))
        thread.daemon = True
        thread.start()
    
    def _convert_image_thread(self, image_path):
        """Convert image in separate thread."""
        try:
            if self.converter:
                # Use converter
                image = self.converter.load_image(image_path)
                depth = self.converter.estimate_depth(image)
                pcd = self.converter.generate_pointcloud(image, depth, max_points=15000)
                mesh = self.converter.generate_mesh(pcd)
                
                # Save results
                output_name = f"uploaded_{os.path.splitext(os.path.basename(image_path))[0]}"
                self.converter.save_3d_model(pcd, mesh, output_name)
                
                self.root.after(0, self._conversion_complete, output_name)
            else:
                self.root.after(0, self._conversion_error, "No converter available")
                
        except Exception as e:
            self.root.after(0, self._conversion_error, str(e))
    
    def _conversion_complete(self, output_name):
        """Handle conversion completion."""
        self.upload_progress.stop()
        self.upload_status.config(text=f"✅ Conversion complete! Files saved as {output_name}")
        self.main_status.config(text=f"3D model generated: {output_name}")
        
        # Enable view 3D button
        for widget in self.root.winfo_children():
            if hasattr(widget, 'winfo_children'):
                for child in widget.winfo_children():
                    if hasattr(child, 'winfo_children'):
                        for grandchild in child.winfo_children():
                            if isinstance(grandchild, ttk.Button) and "View in 3D" in grandchild.cget('text'):
                                grandchild.config(state=tk.NORMAL)
        
        messagebox.showinfo("Success", 
                          f"3D conversion completed!\n\n"
                          f"Generated files:\n"
                          f"• {output_name}.ply (Point cloud)\n"
                          f"• {output_name}.obj (Mesh)\n"
                          f"• {output_name}_mesh.ply (Mesh PLY)\n\n"
                          f"Click 'View in 3D' to see your model!")
    
    def _conversion_error(self, error_msg):
        """Handle conversion error."""
        self.upload_progress.stop()
        self.upload_status.config(text=f"❌ Conversion failed: {error_msg}")
        self.main_status.config(text="Conversion failed")
        messagebox.showerror("Error", f"Conversion failed:\n{error_msg}")
    
    def clear_upload(self):
        """Clear the upload tab."""
        self.file_path_var.set("")
        self.preview_label.config(image="", text="No image selected")
        self.upload_status.config(text="Ready to convert images")
    
    def toggle_camera(self):
        """Toggle camera on/off."""
        if not self.is_camera_on:
            self.start_camera()
        else:
            self.stop_camera()
    
    def start_camera(self):
        """Start the camera."""
        try:
            self.camera = cv2.VideoCapture(0)
            if not self.camera.isOpened():
                messagebox.showerror("Error", "Could not open camera")
                return
            
            self.is_camera_on = True
            self.camera_btn.config(text="📹 Stop Camera")
            self.capture_btn.config(state=tk.NORMAL)
            self.camera_status.config(text="Camera: On")
            
            # Start camera thread
            self.camera_thread = threading.Thread(target=self._camera_loop)
            self.camera_thread.daemon = True
            self.camera_thread.start()
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to start camera: {str(e)}")
    
    def stop_camera(self):
        """Stop the camera."""
        self.is_camera_on = False
        if self.camera:
            self.camera.release()
            self.camera = None
        
        self.camera_btn.config(text="📹 Start Camera")
        self.capture_btn.config(state=tk.DISABLED)
        self.camera_status.config(text="Camera: Off")
        self.camera_label.config(image="", text="Camera not started")
    
    def _camera_loop(self):
        """Camera loop in separate thread."""
        while self.is_camera_on:
            if self.camera and self.camera.isOpened():
                ret, frame = self.camera.read()
                if ret:
                    self.current_frame = frame.copy()
                    
                    # Resize for preview
                    height, width = frame.shape[:2]
                    max_size = 400
                    if height > max_size or width > max_size:
                        scale = max_size / max(height, width)
                        new_width = int(width * scale)
                        new_height = int(height * scale)
                        frame = cv2.resize(frame, (new_width, new_height))
                    
                    # Convert BGR to RGB
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    
                    # Convert to PIL Image and then to PhotoImage
                    pil_image = Image.fromarray(frame_rgb)
                    photo = ImageTk.PhotoImage(pil_image)
                    
                    # Update camera preview
                    self.root.after(0, self._update_camera_preview, photo)
                    
                    time.sleep(0.03)  # ~30 FPS
    
    def _update_camera_preview(self, photo):
        """Update camera preview (called from main thread)."""
        if self.is_camera_on:
            self.camera_label.configure(image=photo, text="")
            self.camera_label.image = photo
    
    def capture_and_convert(self):
        """Capture current frame and convert to 3D."""
        if self.current_frame is None:
            messagebox.showerror("Error", "No frame captured")
            return
        
        # Save captured frame
        timestamp = int(time.time())
        capture_path = f"captured_{timestamp}.jpg"
        cv2.imwrite(capture_path, self.current_frame)
        
        # Start conversion
        self.camera_progress.start()
        self.camera_status.config(text="Converting captured image to 3D...")
        
        # Convert in separate thread
        thread = threading.Thread(target=self._convert_captured_image, args=(capture_path,))
        thread.daemon = True
        thread.start()
    
    def _convert_captured_image(self, capture_path):
        """Convert captured image in separate thread."""
        try:
            if self.converter:
                # Use converter
                image = self.converter.load_image(capture_path)
                depth = self.converter.estimate_depth(image)
                pcd = self.converter.generate_pointcloud(image, depth, max_points=15000)
                mesh = self.converter.generate_mesh(pcd)
                
                # Save results
                output_name = f"captured_{os.path.splitext(os.path.basename(capture_path))[0]}"
                self.converter.save_3d_model(pcd, mesh, output_name)
                
                self.root.after(0, self._capture_conversion_complete, output_name)
            else:
                self.root.after(0, self._capture_conversion_error, "No converter available")
                
        except Exception as e:
            self.root.after(0, self._capture_conversion_error, str(e))
    
    def _capture_conversion_complete(self, output_name):
        """Handle capture conversion completion."""
        self.camera_progress.stop()
        self.camera_status.config(text=f"✅ Capture conversion complete! Files saved as {output_name}")
        self.main_status.config(text=f"3D model generated: {output_name}")
        
        # Enable view 3D button
        self.view_3d_camera_btn.config(state=tk.NORMAL)
        
        messagebox.showinfo("Success", 
                          f"3D conversion completed!\n\n"
                          f"Generated files:\n"
                          f"• {output_name}.ply (Point cloud)\n"
                          f"• {output_name}.obj (Mesh)\n"
                          f"• {output_name}_mesh.ply (Mesh PLY)\n\n"
                          f"Click 'View in 3D' to see your model!")
    
    def _capture_conversion_error(self, error_msg):
        """Handle capture conversion error."""
        self.camera_progress.stop()
        self.camera_status.config(text=f"❌ Capture conversion failed: {error_msg}")
        self.main_status.config(text="Capture conversion failed")
        messagebox.showerror("Error", f"Capture conversion failed:\n{error_msg}")
    
    def view_latest_3d(self):
        """View the latest generated 3D model."""
        # Find the most recent 3D file
        recent_files = self.find_recent_3d_files()
        if recent_files:
            latest_file = recent_files[0]
            self.viewer_file_path_var.set(latest_file)
            self.load_and_view_3d()
        else:
            messagebox.showwarning("Warning", "No 3D files found. Please convert an image first.")
    
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
            self.viewer_file_path_var.set(filename)
    
    def load_and_view_3d(self):
        """Load and display 3D model."""
        file_path = self.viewer_file_path_var.get()
        
        if not file_path or not os.path.exists(file_path):
            messagebox.showerror("Error", "Please select a valid 3D model file")
            return
        
        # Start loading
        self.viewer_progress.start()
        self.viewer_status.config(text="Loading 3D model...")
        
        thread = threading.Thread(target=self._load_3d_thread, args=(file_path,))
        thread.daemon = True
        thread.start()
    
    def quick_load_3d(self, file_path):
        """Quick load a 3D file."""
        self.viewer_file_path_var.set(file_path)
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
        self.viewer_progress.stop()
        
        if self.vis and self.is_viewer_running:
            # Clear current geometry
            self.vis.clear_geometries()
            
            # Add new geometry
            self.vis.add_geometry(geometry)
            
            # Fit view to geometry
            self.vis.reset_view_point(True)
            
            # Update status
            self.viewer_status.config(text=f"✅ Loaded: {os.path.basename(file_path)}")
            self.main_status.config(text=f"3D model loaded: {os.path.basename(file_path)}")
            
            # Show success message
            if hasattr(geometry, 'points'):
                point_count = len(geometry.points)
                messagebox.showinfo("Success", 
                                  f"3D Model Loaded Successfully!\n\n"
                                  f"File: {os.path.basename(file_path)}\n"
                                  f"Type: Point Cloud\n"
                                  f"Points: {point_count:,}\n\n"
                                  f"🎮 Use mouse to rotate 360°, zoom, and pan!")
            else:
                vertex_count = len(geometry.vertices)
                face_count = len(geometry.triangles)
                messagebox.showinfo("Success", 
                                  f"3D Model Loaded Successfully!\n\n"
                                  f"File: {os.path.basename(file_path)}\n"
                                  f"Type: Triangle Mesh\n"
                                  f"Vertices: {vertex_count:,}\n"
                                  f"Faces: {face_count:,}\n\n"
                                  f"🎮 Use mouse to rotate 360°, zoom, and pan!")
        else:
            self.viewer_status.config(text="❌ 3D Viewer not ready")
    
    def _load_error(self, error_msg):
        """Handle 3D model loading error."""
        self.viewer_progress.stop()
        self.viewer_status.config(text=f"❌ Failed to load 3D model: {error_msg}")
        self.main_status.config(text="3D model loading failed")
        messagebox.showerror("Error", f"Failed to load 3D model:\n{error_msg}")
    
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
    
    def start_3d_viewer(self):
        """Start the 3D viewer in a separate thread."""
        def viewer_thread():
            try:
                # Create Open3D visualizer
                self.vis = o3d.visualization.Visualizer()
                self.vis.create_window(
                    window_name="Professional 3D Viewer - 2D to 3D Workflow",
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
                self.root.after(0, lambda: self.viewer_status.config(text="3D Viewer: Ready"))
                self.is_viewer_running = True
                
                # Run viewer loop
                self.vis.run()
                
                # Cleanup when viewer closes
                self.vis.destroy_window()
                self.is_viewer_running = False
                
            except Exception as e:
                self.root.after(0, lambda: self.viewer_status.config(text=f"3D Viewer Error: {e}"))
        
        # Start viewer thread
        thread = threading.Thread(target=viewer_thread)
        thread.daemon = True
        thread.start()
    
    def on_closing(self):
        """Handle application closing."""
        self.stop_camera()
        if self.vis and self.is_viewer_running:
            self.vis.close()
        self.root.destroy()


def main():
    """Main function."""
    root = tk.Tk()
    app = ProfessionalWorkflow(root)
    
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
