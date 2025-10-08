import os
import sys
import ctypes
import numpy as np
import cv2

# Method 1: Add DLL directory to PATH before importing (for 64-bit Python)
dll_path = r"C:\Users\ag4077\Documents\RealSense SDK 2.0\bin\x64"  # 64-bit DLLs
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
    DLL_PATH = r"C:\Users\ag4077\Documents\RealSense SDK 2.0\bin\x64"  # 64-bit DLLs
    
    # Load the DLL
    load_realsense_dll(DLL_PATH)
    
    # Check for connected devices
    ctx = rs.context()
    devices = ctx.query_devices()
    
    if len(devices) == 0:
        print("No RealSense devices connected!")
        print("Please check:")
        print("  - Camera is plugged into USB 3.0 port")
        print("  - Camera drivers are installed")
        print("  - Camera is not being used by another application")
        return
    
    print(f"Found {len(devices)} RealSense device(s)")
    
    # Get detailed device info
    device = devices[0]
    print(f"\nDevice: {device.get_info(rs.camera_info.name)}")
    print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
    print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
    
    # List all available sensors and their capabilities
    print("\nAvailable sensors:")
    for sensor in device.query_sensors():
        print(f"  - {sensor.get_info(rs.camera_info.name)}")
        
        # Get supported stream profiles
        profiles = sensor.get_stream_profiles()
        print(f"    Supported profiles: {len(profiles)}")
        for profile in profiles[:5]:  # Show first 5
            if profile.stream_type() == rs.stream.depth:
                vp = profile.as_video_stream_profile()
                print(f"      Depth: {vp.width()}x{vp.height()} @ {vp.fps()}fps, format: {vp.format()}")
            elif profile.stream_type() == rs.stream.color:
                vp = profile.as_video_stream_profile()
                print(f"      Color: {vp.width()}x{vp.height()} @ {vp.fps()}fps, format: {vp.format()}")
    
    # Create pipeline
    pipeline = rs.pipeline()
    config = rs.config()
    
    # Try simple configuration first
    print("\nAttempting to start pipeline...")
    
    # D555 specific configuration - use supported resolutions
    try:
        # D555 supports 1280x720 for depth and 1280x800 for color
        config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
        config.enable_stream(rs.stream.color, 1280, 800, rs.format.rgb8, 30)
        print("Trying: Depth 1280x720 @ 30fps + Color 1280x800 @ 30fps (D555 native)")
    except Exception as e:
        print(f"Configuration error: {e}")
        
    # Alternative: Try depth only if color+depth fails
    # config = rs.config()
    # config.enable_stream(rs.stream.depth, 1280, 720, rs.format.z16, 30)
    # print("Trying: Depth only 1280x720 @ 30fps")
    
    pipeline_started = False
    try:
        # Start streaming
        print("Starting RealSense pipeline...")
        pipeline.start(config)
        pipeline_started = True
        print("Pipeline started successfully!")
        print("Press 'q' to quit the display window")
        
        # Create alignment object to align depth to color
        align_to = rs.stream.color
        align = rs.align(align_to)
        
        # Get frames and display
        while True:
            # Wait for frames
            frames = pipeline.wait_for_frames()
            
            # Align depth frame to color frame
            aligned_frames = align.process(frames)
            depth_frame = aligned_frames.get_depth_frame()
            color_frame = aligned_frames.get_color_frame()
            
            if not depth_frame or not color_frame:
                continue
            
            # Convert images to numpy arrays
            depth_image = np.asanyarray(depth_frame.get_data())
            color_image = np.asanyarray(color_frame.get_data())
            
            # Convert RGB to BGR for OpenCV
            color_image = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
            
            # Apply colormap to depth image for visualization
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03), 
                cv2.COLORMAP_JET
            )
            
            # Resize depth to match color height for side-by-side display
            depth_colormap_resized = cv2.resize(
                depth_colormap, 
                (color_image.shape[1], color_image.shape[0])
            )
            
            # Stack images horizontally (side by side)
            images = np.hstack((color_image, depth_colormap_resized))
            
            # Add labels
            cv2.putText(images, 'Color', (10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(images, 'Depth', (color_image.shape[1] + 10, 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            # Display images
            cv2.imshow('RealSense D555 - Color and Depth', images)
            
            # Break loop on 'q' key press
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
        
        cv2.destroyAllWindows()
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
    finally:
        if pipeline_started:
            pipeline.stop()
            print("Pipeline stopped")

# Method 3: Using environment variable
def setup_realsense_env():
    """
    Set up environment for RealSense using environment variables (64-bit)
    """
    # Set the path to 64-bit RealSense DLLs
    realsense_path = r"C:\Users\ag4077\Documents\RealSense SDK 2.0\bin\x64"
    
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