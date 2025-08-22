#!/usr/bin/env python3
"""
Modern GUI Application for 2D to 3D Conversion
Features:
1. Upload 2D image from system and convert to 3D
2. Live camera capture with capture button for real-time 3D generation
"""

import tkinter as tk
from tkinter import ttk, filedialog, messagebox
import cv2
import numpy as np
import os
import threading
from PIL import Image, ImageTk
import time
from image_to_3d import ImageTo3DConverter


class ImageTo3DGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("2D to 3D Converter")
        self.root.geometry("1200x800")
        self.root.configure(bg='#2c3e50')
        
        # Initialize variables
        self.camera = None
        self.is_camera_on = False
        self.current_frame = None
        self.converter = None
        
        # Create main container
        self.main_frame = ttk.Frame(root)
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=20, pady=20)
        
        # Create notebook for tabs
        self.notebook = ttk.Notebook(self.main_frame)
        self.notebook.pack(fill=tk.BOTH, expand=True)
        
        # Create tabs
        self.create_upload_tab()
        self.create_camera_tab()
        
        # Initialize converter with fallback
        self.initialize_converter()
        
        # Style configuration
        style = ttk.Style()
        style.theme_use('clam')
        
    def initialize_converter(self):
        """Initialize the 3D converter with fallback."""
        try:
            from mock_converter import create_converter
            self.converter = create_converter("DPT_Large")
            print("✅ Converter initialized successfully")
        except Exception as e:
            print(f"⚠️ Using mock converter due to: {e}")
            self.converter = None
    
    def create_upload_tab(self):
        """Create the file upload tab."""
        upload_frame = ttk.Frame(self.notebook)
        self.notebook.add(upload_frame, text="📁 Upload Image")
        
        # Title
        title_label = ttk.Label(upload_frame, text="Upload 2D Image and Convert to 3D", 
                               font=('Arial', 16, 'bold'))
        title_label.pack(pady=20)
        
        # File selection frame
        file_frame = ttk.Frame(upload_frame)
        file_frame.pack(pady=20, padx=20, fill=tk.X)
        
        self.file_path_var = tk.StringVar()
        file_entry = ttk.Entry(file_frame, textvariable=self.file_path_var, width=50)
        file_entry.pack(side=tk.LEFT, padx=(0, 10))
        
        browse_btn = ttk.Button(file_frame, text="Browse", command=self.browse_file)
        browse_btn.pack(side=tk.LEFT)
        
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
        
        clear_btn = ttk.Button(button_frame, text="🗑️ Clear", command=self.clear_upload)
        clear_btn.pack(side=tk.LEFT, padx=10)
        
        # Progress bar
        self.progress = ttk.Progressbar(upload_frame, mode='indeterminate')
        self.progress.pack(pady=10, fill=tk.X, padx=20)
        
        # Status label
        self.status_label = ttk.Label(upload_frame, text="Ready to convert images", 
                                     font=('Arial', 10))
        self.status_label.pack(pady=10)
    
    def create_camera_tab(self):
        """Create the live camera tab."""
        camera_frame = ttk.Frame(self.notebook)
        self.notebook.add(camera_frame, text="📷 Live Camera")
        
        # Title
        title_label = ttk.Label(camera_frame, text="Live Camera Capture for 3D Generation", 
                               font=('Arial', 16, 'bold'))
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
        
        # Camera preview frame
        camera_preview_frame = ttk.LabelFrame(camera_frame, text="Camera Preview", padding=20)
        camera_preview_frame.pack(pady=20, padx=20, fill=tk.BOTH, expand=True)
        
        self.camera_label = ttk.Label(camera_preview_frame, text="Camera not started")
        self.camera_label.pack(expand=True)
        
        # Camera status
        self.camera_status = ttk.Label(camera_frame, text="Camera: Off", 
                                     font=('Arial', 10))
        self.camera_status.pack(pady=10)
        
        # Progress bar for camera
        self.camera_progress = ttk.Progressbar(camera_frame, mode='indeterminate')
        self.camera_progress.pack(pady=10, fill=tk.X, padx=20)
    
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
            # Load and resize image for preview
            image = cv2.imread(image_path)
            if image is not None:
                # Resize for preview (max 400x400)
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
                self.preview_label.image = photo  # Keep a reference
                
                self.status_label.config(text=f"Image loaded: {os.path.basename(image_path)}")
            else:
                self.preview_label.config(text="Failed to load image")
                self.status_label.config(text="Error: Could not load image")
        except Exception as e:
            self.preview_label.config(text=f"Error: {str(e)}")
            self.status_label.config(text="Error loading image")
    
    def convert_uploaded_image(self):
        """Convert the uploaded image to 3D."""
        image_path = self.file_path_var.get()
        
        if not image_path or not os.path.exists(image_path):
            messagebox.showerror("Error", "Please select a valid image file")
            return
        
        # Start conversion in separate thread
        self.progress.start()
        self.status_label.config(text="Converting image to 3D...")
        
        thread = threading.Thread(target=self._convert_image_thread, args=(image_path,))
        thread.daemon = True
        thread.start()
    
    def _convert_image_thread(self, image_path):
        """Convert image in separate thread."""
        try:
            if self.converter:
                # Use real converter
                image = self.converter.load_image(image_path)
                depth = self.converter.estimate_depth(image)
                pcd = self.converter.generate_pointcloud(image, depth, max_points=15000)
                mesh = self.converter.generate_mesh(pcd)
                
                # Save results
                output_name = f"uploaded_{os.path.splitext(os.path.basename(image_path))[0]}"
                self.converter.save_3d_model(pcd, mesh, output_name)
                
                self.root.after(0, self._conversion_complete, output_name, True)
            else:
                # Use mock converter
                self.root.after(0, self._conversion_complete, "mock_output", False)
                
        except Exception as e:
            self.root.after(0, self._conversion_error, str(e))
    
    def _conversion_complete(self, output_name, is_real):
        """Handle conversion completion."""
        self.progress.stop()
        if is_real:
            self.status_label.config(text=f"✅ Conversion complete! Files saved as {output_name}")
            messagebox.showinfo("Success", 
                              f"3D conversion completed!\n\n"
                              f"Generated files:\n"
                              f"• {output_name}.ply (Point cloud)\n"
                              f"• {output_name}.obj (Mesh)\n"
                              f"• {output_name}_mesh.ply (Mesh PLY)")
        else:
            self.status_label.config(text="✅ Mock conversion complete!")
            messagebox.showinfo("Success", 
                              "Mock 3D conversion completed!\n\n"
                              "This demonstrates the workflow.\n"
                              "Real MiDaS conversion will work when GitHub allows.")
    
    def _conversion_error(self, error_msg):
        """Handle conversion error."""
        self.progress.stop()
        self.status_label.config(text=f"❌ Conversion failed: {error_msg}")
        messagebox.showerror("Error", f"Conversion failed:\n{error_msg}")
    
    def clear_upload(self):
        """Clear the upload tab."""
        self.file_path_var.set("")
        self.preview_label.config(image="", text="No image selected")
        self.status_label.config(text="Ready to convert images")
    
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
                # Use real converter
                image = self.converter.load_image(capture_path)
                depth = self.converter.estimate_depth(image)
                pcd = self.converter.generate_pointcloud(image, depth, max_points=15000)
                mesh = self.converter.generate_mesh(pcd)
                
                # Save results
                output_name = f"captured_{os.path.splitext(os.path.basename(capture_path))[0]}"
                self.converter.save_3d_model(pcd, mesh, output_name)
                
                self.root.after(0, self._capture_conversion_complete, output_name, True)
            else:
                # Use mock converter
                self.root.after(0, self._capture_conversion_complete, "mock_captured", False)
                
        except Exception as e:
            self.root.after(0, self._capture_conversion_error, str(e))
    
    def _capture_conversion_complete(self, output_name, is_real):
        """Handle capture conversion completion."""
        self.camera_progress.stop()
        if is_real:
            self.camera_status.config(text=f"✅ Capture conversion complete! Files saved as {output_name}")
            messagebox.showinfo("Success", 
                              f"3D conversion completed!\n\n"
                              f"Generated files:\n"
                              f"• {output_name}.ply (Point cloud)\n"
                              f"• {output_name}.obj (Mesh)\n"
                              f"• {output_name}_mesh.ply (Mesh PLY)")
        else:
            self.camera_status.config(text="✅ Mock capture conversion complete!")
            messagebox.showinfo("Success", 
                              "Mock 3D conversion completed!\n\n"
                              "This demonstrates the workflow.\n"
                              "Real MiDaS conversion will work when GitHub allows.")
    
    def _capture_conversion_error(self, error_msg):
        """Handle capture conversion error."""
        self.camera_progress.stop()
        self.camera_status.config(text=f"❌ Capture conversion failed: {error_msg}")
        messagebox.showerror("Error", f"Capture conversion failed:\n{error_msg}")
    
    def on_closing(self):
        """Handle application closing."""
        self.stop_camera()
        self.root.destroy()


def main():
    """Main function."""
    root = tk.Tk()
    app = ImageTo3DGUI(root)
    
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
