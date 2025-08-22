#!/usr/bin/env python3
"""
Test script to verify that all dependencies are properly installed.
Run this script to check if your environment is ready for 2D to 3D conversion.
"""

import sys
import importlib


def test_import(module_name, package_name=None):
    """Test if a module can be imported."""
    try:
        if package_name:
            importlib.import_module(module_name)
            print(f"✅ {package_name or module_name}: OK")
            return True
        else:
            importlib.import_module(module_name)
            print(f"✅ {module_name}: OK")
            return True
    except ImportError as e:
        print(f"❌ {package_name or module_name}: FAILED - {e}")
        return False


def test_torch_cuda():
    """Test PyTorch CUDA availability."""
    try:
        import torch
        if torch.cuda.is_available():
            print(f"✅ PyTorch CUDA: Available ({torch.cuda.get_device_name(0)})")
            return True
        else:
            print("⚠️  PyTorch CUDA: Not available (will use CPU)")
            return True
    except Exception as e:
        print(f"❌ PyTorch CUDA test failed: {e}")
        return False


def main():
    """Run all tests."""
    print("=== Testing 2D to 3D Converter Dependencies ===\n")
    
    # Test core dependencies
    core_deps = [
        ("cv2", "OpenCV"),
        ("torch", "PyTorch"),
        ("torchvision", "TorchVision"),
        ("numpy", "NumPy"),
        ("PIL", "Pillow"),
        ("open3d", "Open3D"),
    ]
    
    all_passed = True
    
    for module, name in core_deps:
        if not test_import(module, name):
            all_passed = False
    
    print()
    
    # Test PyTorch CUDA
    test_torch_cuda()
    
    print()
    
    # Test MiDaS model loading
    print("Testing MiDaS model loading...")
    try:
        import torch
        midas = torch.hub.load("intel-isl/MiDaS", "DPT_Large", pretrained=True)
        print("✅ MiDaS model: OK")
    except Exception as e:
        print(f"❌ MiDaS model: FAILED - {e}")
        all_passed = False
    
    print()
    
    # Test our converter class
    print("Testing ImageTo3DConverter class...")
    try:
        from image_to_3d import ImageTo3DConverter
        print("✅ ImageTo3DConverter: OK")
    except Exception as e:
        print(f"❌ ImageTo3DConverter: FAILED - {e}")
        all_passed = False
    
    print()
    
    # Summary
    if all_passed:
        print("🎉 All tests passed! Your environment is ready for 2D to 3D conversion.")
        print("\nNext steps:")
        print("1. Place an image file in the current directory")
        print("2. Run: python example.py")
        print("3. Or use command line: python image_to_3d.py your_image.jpg")
    else:
        print("❌ Some tests failed. Please install missing dependencies:")
        print("pip install -r requirements.txt")
        return 1
    
    return 0


if __name__ == "__main__":
    exit(main())
