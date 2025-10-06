import os
import sys
import ctypes

# Method 1: Add DLL directory to PATH before importing (for 64-bit Python)
dll_path = r"C:\Program Files\Intel RealSense SDK 2.0\bin\x64"  # 64-bit DLLs
os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']

# For Python 3.8+, you can also use add_dll_directory
if hasattr(os, 'add_dll_directory'):
    os.add_dll_directory(dll_path)

# Now import pyrealsense2
import pyrealsense2 as rs

# Method 2: Direct DLL loading (alternative approach)
def load_realsense_dll(dll_path):
    """
    Manually load RealSense DLL before importing pyrealsense2
    """
    try:
        # Load realsense2.dll
        realsense_dll = os.path.join(dll_path, "realsense2.dll")
        if os.path.exists(realsense_dll):
            ctypes.CDLL(realsense_dll)
            print(f"Successfully loaded DLL from: {realsense_dll}")
            return True
        else:
            print(f"DLL not found at: {realsense_dll}")
            return False
    except Exception as e:
        print(f"Error loading DLL: {e}")
        return False

# Example usage
def main():
    # Configuration for 64-bit Python
    DLL_PATH = r"C:\Program Files\Intel RealSense SDK 2.0\bin\x64"  # 64-bit DLLs
    
    # Load the DLL
    load_realsense_dll(DLL_PATH)
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Configure streams
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    
    try:
        # Start streaming
        print("Starting RealSense pipeline...")
        pipeline.start(config)
        print("Pipeline started successfully!")
        
        # Get a few frames
        for i in range(10):
            frames = pipeline.wait_for_frames()
            depth_frame = frames.get_depth_frame()
            color_frame = frames.get_color_frame()
            
            if depth_frame and color_frame:
                print(f"Frame {i}: Depth and Color frames received")
                print(f"  Depth dimensions: {depth_frame.get_width()}x{depth_frame.get_height()}")
                print(f"  Color dimensions: {color_frame.get_width()}x{color_frame.get_height()}")
        
    except Exception as e:
        print(f"Error: {e}")
    finally:
        pipeline.stop()
        print("Pipeline stopped")

# Method 3: Using environment variable
def setup_realsense_env():
    """
    Set up environment for RealSense using environment variables (64-bit)
    """
    # Set the path to 64-bit RealSense DLLs
    realsense_path = r"C:\Program Files\Intel RealSense SDK 2.0\bin\x64"
    
    # Add to PATH
    if realsense_path not in os.environ['PATH']:
        os.environ['PATH'] = realsense_path + os.pathsep + os.environ['PATH']
    
    # Set REALSENSE2_SDK_PATH (optional, for some configurations)
    os.environ['REALSENSE2_SDK_PATH'] = realsense_path

if __name__ == "__main__":
    # Setup environment first
    setup_realsense_env()
    
    # Run main program
    main()