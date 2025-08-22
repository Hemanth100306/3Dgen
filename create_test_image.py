#!/usr/bin/env python3
"""
Create a simple test image with depth cues for testing the 2D to 3D converter.
"""

import numpy as np
import cv2
from PIL import Image, ImageDraw


def create_depth_test_image():
    """Create a simple test image with depth cues."""
    
    # Create a 512x512 image
    width, height = 512, 512
    
    # Create base image with gradient background
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Add gradient background (darker = closer)
    for y in range(height):
        for x in range(width):
            # Create a radial gradient from center
            center_x, center_y = width // 2, height // 2
            distance = np.sqrt((x - center_x)**2 + (y - center_y)**2)
            max_distance = np.sqrt(center_x**2 + center_y**2)
            
            # Normalize distance to 0-1
            normalized_distance = distance / max_distance
            
            # Create depth effect (closer = brighter)
            depth_value = int(255 * (1 - normalized_distance))
            
            # Add some color variation
            r = max(50, depth_value)
            g = max(30, int(depth_value * 0.8))
            b = max(20, int(depth_value * 0.6))
            
            image[y, x] = [r, g, b]
    
    # Add some geometric shapes with different depths
    # Circle in the center (closest)
    cv2.circle(image, (width//2, height//2), 80, (255, 200, 150), -1)
    
    # Rectangle on the left (medium depth)
    cv2.rectangle(image, (100, 150), (200, 350), (150, 255, 200), -1)
    
    # Triangle on the right (farthest)
    pts = np.array([[400, 100], [350, 300], [450, 300]], np.int32)
    cv2.fillPoly(image, [pts], (200, 150, 255))
    
    # Add some text
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(image, '3D Test', (50, 50), font, 1, (255, 255, 255), 2)
    cv2.putText(image, 'Depth Cues', (50, 480), font, 1, (200, 200, 200), 2)
    
    # Save the image
    cv2.imwrite('test_depth_image.jpg', image)
    print("✅ Created test image: test_depth_image.jpg")
    print("📏 Image size: 512x512 pixels")
    print("🎨 Contains: gradient background, circle, rectangle, triangle")
    print("🔍 Depth cues: center objects appear closer, edges appear farther")
    
    return 'test_depth_image.jpg'


if __name__ == "__main__":
    create_depth_test_image()
