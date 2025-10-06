import os
import pyrealsense2 as rs
import numpy as np
import cv2

# --- Adjust DLL path if needed ---
# Only needed if PATH isn't set; comment out if DLL already found
os.add_dll_directory(r"C:\Program Files (Intel RealSense SDK 2.0)\bin")

def main():
    ctx = rs.context()
    devices = ctx.devices
    if len(devices) == 0:
        print("No RealSense device found!")
        return
    else:
        print(f"Found {len(devices)} device(s):")
        for dev in devices:
            print(f"  - {dev.get_info(rs.camera_info.name)}")

    # Configure streams
    pipeline = rs.pipeline()
    config = rs.config()
    config.enable_stream(rs.stream.color, 640, 480, rs.format.bgr8, 30)
    config.enable_stream(rs.stream.depth, 640, 480, rs.format.z16, 30)

    # Start streaming
    pipeline.start(config)
    print("Streaming... Press 'q' to quit.")

    try:
        while True:
            frames = pipeline.wait_for_frames()
            color_frame = frames.get_color_frame()
            depth_frame = frames.get_depth_frame()

            if not color_frame or not depth_frame:
                continue

            color_image = np.asanyarray(color_frame.get_data())
            depth_image = np.asanyarray(depth_frame.get_data())
            depth_colormap = cv2.applyColorMap(
                cv2.convertScaleAbs(depth_image, alpha=0.03),
                cv2.COLORMAP_JET
            )

            images = np.hstack((color_image, depth_colormap))
            cv2.imshow("RealSense D555 - Color | Depth", images)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        pipeline.stop()
        cv2.destroyAllWindows()
        print("Streaming stopped.")

if __name__ == "__main__":
    main()
