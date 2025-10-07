import os
import sys
import ctypes
import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple
from pylsl import StreamOutlet, StreamInfo, local_clock
from config_manager import CameraConfig


# Setup RealSense DLL path before importing pyrealsense2
def setup_realsense_dll():
    """Load RealSense DLL from custom path"""
    dll_path = r"C:\Users\ag4077\Documents\RealSense SDK 2.0\bin\x64"
    
    # Add to PATH
    os.environ['PATH'] = dll_path + os.pathsep + os.environ['PATH']
    
    # For Python 3.8+
    if hasattr(os, 'add_dll_directory'):
        os.add_dll_directory(dll_path)
    
    # Load DLL explicitly
    try:
        realsense_dll = os.path.join(dll_path, "realsense2.dll")
        if os.path.exists(realsense_dll):
            ctypes.CDLL(realsense_dll)
            print(f"Successfully loaded RealSense DLL from: {realsense_dll}")
            return True
    except Exception as e:
        print(f"Error loading RealSense DLL: {e}")
        return False


# Initialize DLL before importing pyrealsense2
setup_realsense_dll()
import pyrealsense2 as rs


class CameraHandler:
    def __init__(self, camera_config, participant_id: str):
        self.config = camera_config
        self.participant_id = participant_id
        self.cap = None
        self.outlet = None
        self.audio_outlet = None
        self.running = False
        self.thread = None
        self.lock = threading.Lock()

        # Stream names
        self.video_stream_name = f"{camera_config.name}_{participant_id}"
        self.audio_stream_name = f"Audio_{camera_config.name}_{participant_id}"

    def initialize(self) -> bool:
        """Initialize camera and LSL outlets"""
        try:
            self.cap = cv2.VideoCapture(self.config.id)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.config.id}")
                return False

            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)

            # Setup LSL outlet
            if self.config.stream_type == "Video":
                video_info = StreamInfo(
                    self.video_stream_name, 'Video',
                    channel_count=1,
                    nominal_srate=self.config.fps,
                    channel_format='string',
                    source_id=f"{self.video_stream_name}_{self.config.id}"
                )
            elif self.config.stream_type == "Depth":
                video_info = StreamInfo(
                    self.video_stream_name, 'Depth',
                    channel_count=self.config.resolution[0] * self.config.resolution[1],
                    nominal_srate=self.config.fps,
                    channel_format='float32',
                    source_id=f"{self.video_stream_name}_{self.config.id}"
                )

            self.outlet = StreamOutlet(video_info)
            print(f"Initialized camera {self.config.name} (ID: {self.config.id})")
            return True

        except Exception as e:
            print(f"Error initializing camera {self.config.id}: {e}")
            return False

    def start_streaming(self):
        """Start the camera streaming in a separate thread"""
        if self.running:
            print(f"Camera {self.config.name} is already streaming")
            return

        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        print(f"Started streaming for camera {self.config.name}")

    def stop_streaming(self):
        """Stop camera streaming"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        print(f"Stopped streaming for camera {self.config.name}")

    def _stream_loop(self):
        target_interval = 1.0 / self.config.fps

        while self.running:
            start_time = time.time()
            ret, frame = self.cap.read()

            if not ret:
                print(f"Failed to read frame from camera {self.config.id}")
                time.sleep(0.05)
                continue

            try:
                timestamp = local_clock()

                if self.config.stream_type == "Video":
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    jpeg_data = buffer.tobytes()
                    self.outlet.push_sample([jpeg_data.hex()], timestamp)

                elif self.config.stream_type == "Depth":
                    if len(frame.shape) == 3:
                        depth_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        depth_frame = frame

                    depth_data = depth_frame.astype(np.float32).flatten()
                    self.outlet.push_sample(depth_data, timestamp)

            except Exception as e:
                print(f"Error processing frame from camera {self.config.id}: {e}")

            elapsed = time.time() - start_time
            sleep_time = max(0, target_interval - elapsed)
            time.sleep(sleep_time)

    def is_running(self) -> bool:
        return self.running and self.thread and self.thread.is_alive()

    def cleanup(self):
        """Clean up resources"""
        self.stop_streaming()
        if self.cap:
            self.cap.release()
        print(f"Camera {self.config.name} cleaned up")


class DepthCameraHandler(CameraHandler):
    """Specialized handler for Intel RealSense depth cameras with RGB and Depth streaming"""
    
    def __init__(self, camera_config: CameraConfig, participant_id: str):
        super().__init__(camera_config, participant_id)
        self.pipeline = None
        self.align = None
        self.pipeline_started = False
        
        # Create separate outlets for RGB and Depth
        self.rgb_outlet = None
        self.depth_outlet = None
        
        # Stream names
        self.rgb_stream_name = f"{camera_config.name}_RGB_{participant_id}"
        self.depth_stream_name = f"{camera_config.name}_Depth_{participant_id}"
        
    def initialize(self) -> bool:
        """Initialize RealSense camera with both RGB and Depth streams"""
        try:
            # Check for connected devices
            ctx = rs.context()
            devices = ctx.query_devices()
            
            if len(devices) == 0:
                print("No RealSense devices connected!")
                return False
            
            device = devices[0]
            print(f"Found RealSense device: {device.get_info(rs.camera_info.name)}")
            print(f"Serial: {device.get_info(rs.camera_info.serial_number)}")
            print(f"Firmware: {device.get_info(rs.camera_info.firmware_version)}")
            
            # Configure streams for D555
            self.pipeline = rs.pipeline()
            config = rs.config()
            
            # D555 supports 1280x720 depth and 1280x800 color
            # Adjust based on your config or use native resolutions
            depth_width = 1280
            depth_height = 720
            color_width = 1280
            color_height = 800
            fps = self.config.fps if self.config.fps <= 30 else 30
            
            config.enable_stream(rs.stream.depth, depth_width, depth_height, rs.format.z16, fps)
            config.enable_stream(rs.stream.color, color_width, color_height, rs.format.rgb8, fps)
            
            # Start pipeline
            self.pipeline.start(config)
            self.pipeline_started = True
            
            # Create alignment object to align depth to color
            self.align = rs.align(rs.stream.color)
            
            # Create LSL outlet for RGB data (as JPEG string)
            rgb_info = StreamInfo(
                self.rgb_stream_name,
                'Video',
                channel_count=1,
                nominal_srate=fps,
                channel_format='string',
                source_id=f"{self.rgb_stream_name}_{self.config.id}"
            )
            self.rgb_outlet = StreamOutlet(rgb_info)
            
            # Create LSL outlet for Depth data (as float array)
            depth_info = StreamInfo(
                self.depth_stream_name,
                'Depth',
                channel_count=depth_width * depth_height,
                nominal_srate=fps,
                channel_format='float32',
                source_id=f"{self.depth_stream_name}_{self.config.id}"
            )
            self.depth_outlet = StreamOutlet(depth_info)
            
            print(f"Initialized RealSense camera {self.config.name}")
            print(f"  RGB Stream: {color_width}x{color_height} @ {fps}fps")
            print(f"  Depth Stream: {depth_width}x{depth_height} @ {fps}fps")
            return True
            
        except Exception as e:
            print(f"Error initializing RealSense camera: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def _stream_loop(self):
        """RealSense specific streaming loop - streams both RGB and Depth"""
        if not self.pipeline or not self.pipeline_started:
            print("Pipeline not started, cannot stream")
            return
        
        target_interval = 1.0 / self.config.fps
        frame_count = 0
        start_time = time.time()
        
        print(f"Starting RealSense streaming loop for {self.config.name}")
        
        while self.running:
            loop_start = time.time()
            
            try:
                # Wait for frames with timeout
                frames = self.pipeline.wait_for_frames(timeout_ms=5000)
                
                # Align depth to color
                aligned_frames = self.align.process(frames)
                
                depth_frame = aligned_frames.get_depth_frame()
                color_frame = aligned_frames.get_color_frame()
                
                if not depth_frame or not color_frame:
                    print("Missing frames, skipping...")
                    continue
                
                timestamp = local_clock()
                
                # Process RGB frame
                color_image = np.asanyarray(color_frame.get_data())
                # Convert RGB to BGR for OpenCV
                color_bgr = cv2.cvtColor(color_image, cv2.COLOR_RGB2BGR)
                
                _, buffer = cv2.imencode('.jpg', color_bgr, [cv2.IMWRITE_JPEG_QUALITY, 85])
                jpeg_data = buffer.tobytes()
                self.rgb_outlet.push_sample([jpeg_data.hex()], timestamp)
                
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_data = depth_image.astype(np.float32).flatten()
                self.depth_outlet.push_sample(depth_data, timestamp)
                
                frame_count += 1
                
                if frame_count % 100 == 0:
                    elapsed = time.time() - start_time
                    actual_fps = frame_count / elapsed
                    print(f"[{self.config.name}] Frames: {frame_count}, "
                          f"Actual FPS: {actual_fps:.2f}, Target FPS: {self.config.fps}")
                
            except Exception as e:
                print(f"Error processing RealSense frame: {e}")
                import traceback
                traceback.print_exc()
                time.sleep(0.1)
                continue
            
            elapsed = time.time() - loop_start
            sleep_time = max(0, target_interval - elapsed)
            if sleep_time > 0:
                time.sleep(sleep_time)
    
    def cleanup(self):
        """Clean up RealSense resources"""
        print(f"Cleaning up RealSense camera {self.config.name}")
        self.stop_streaming()
        
        if self.pipeline and self.pipeline_started:
            try:
                self.pipeline.stop()
                self.pipeline_started = False
                print("Pipeline stopped successfully")
            except Exception as e:
                print(f"Error stopping pipeline: {e}")
        
        print(f"RealSense camera {self.config.name} cleaned up")