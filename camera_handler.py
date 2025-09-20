import cv2
import numpy as np
import threading
import time
from typing import Optional, Tuple, Callable
from pylsl import StreamOutlet, StreamInfo, local_clock
from config_manager import CameraConfig

class CameraHandler:
    def __init__(self, camera_config: CameraConfig, participant_id: str):
        self.config = camera_config
        self.participant_id = participant_id
        self.cap = None
        self.outlet = None
        self.audio_outlet = None
        self.running = False
        self.thread = None
        
        # Stream names
        self.video_stream_name = f"{camera_config.name}_{participant_id}"
        self.audio_stream_name = f"Audio_{camera_config.name}_{participant_id}"
        
    def initialize(self) -> bool:
        """Initialize camera and LSL outlets"""
        try:
            # Initialize camera
            self.cap = cv2.VideoCapture(self.config.id)
            if not self.cap.isOpened():
                print(f"Error: Could not open camera {self.config.id}")
                return False
            
            # Set camera properties
            self.cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.config.resolution[0])
            self.cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.config.resolution[1])
            self.cap.set(cv2.CAP_PROP_FPS, self.config.fps)
            
            # Create LSL video outlet
            if self.config.stream_type == "Video":
                # For RGB video, we'll send compressed JPEG data
                video_info = StreamInfo(
                    self.video_stream_name,
                    'Video',
                    channel_count=1,  # Single channel for JPEG data
                    nominal_srate=self.config.fps,
                    channel_format='string',
                    source_id=f"{self.video_stream_name}_{self.config.id}"
                )
            elif self.config.stream_type == "Depth":
                # For depth, send raw depth values
                video_info = StreamInfo(
                    self.video_stream_name,
                    'Depth',
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
    
    def create_audio_outlet(self, sample_rate: int = 44100, channels: int = 1):
        """Create LSL audio outlet for microphone data"""
        try:
            audio_info = StreamInfo(
                self.audio_stream_name,
                'Audio',
                channel_count=channels,
                nominal_srate=sample_rate,
                channel_format='float32',
                source_id=f"{self.audio_stream_name}_{self.config.id}"
            )
            self.audio_outlet = StreamOutlet(audio_info)
            print(f"Audio outlet created for {self.config.name}")
        except Exception as e:
            print(f"Error creating audio outlet: {e}")
    
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
        """Main streaming loop"""
        target_interval = 1.0 / self.config.fps
        
        while self.running:
            start_time = time.time()
            
            ret, frame = self.cap.read()
            if not ret:
                print(f"Failed to read frame from camera {self.config.id}")
                continue
            
            try:
                timestamp = local_clock()
                
                if self.config.stream_type == "Video":
                    # Compress frame to JPEG and send as string
                    _, buffer = cv2.imencode('.jpg', frame, [cv2.IMWRITE_JPEG_QUALITY, 80])
                    jpeg_data = buffer.tobytes()
                    # Convert to string for LSL transmission
                    self.outlet.push_sample([jpeg_data.hex()], timestamp)
                    
                elif self.config.stream_type == "Depth":
                    # For depth cameras, convert to depth values
                    # This is a placeholder - actual depth camera implementation would differ
                    if len(frame.shape) == 3:
                        depth_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                    else:
                        depth_frame = frame
                    
                    # Flatten depth data
                    depth_data = depth_frame.astype(np.float32).flatten()
                    self.outlet.push_sample(depth_data, timestamp)
                
            except Exception as e:
                print(f"Error processing frame from camera {self.config.id}: {e}")
            
            # Maintain target framerate
            elapsed = time.time() - start_time
            sleep_time = max(0, target_interval - elapsed)
            time.sleep(sleep_time)
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_streaming()
        if self.cap:
            self.cap.release()
        print(f"Camera {self.config.name} cleaned up")


class DepthCameraHandler(CameraHandler):
    """Specialized handler for Intel RealSense depth cameras"""
    
    def __init__(self, camera_config: CameraConfig, participant_id: str):
        super().__init__(camera_config, participant_id)
        self.pipeline = None
        
    def initialize(self) -> bool:
        """Initialize RealSense camera"""
        try:
            import pyrealsense2 as rs
            
            # Configure streams
            self.pipeline = rs.pipeline()
            config = rs.config()
            config.enable_stream(rs.stream.depth, 
                               self.config.resolution[0], 
                               self.config.resolution[1], 
                               rs.format.z16, 
                               self.config.fps)
            
            # Start streaming
            self.pipeline.start(config)
            
            # Create LSL outlet for depth data
            depth_info = StreamInfo(
                self.video_stream_name,
                'Depth',
                channel_count=self.config.resolution[0] * self.config.resolution[1],
                nominal_srate=self.config.fps,
                channel_format='float32',
                source_id=f"{self.video_stream_name}_{self.config.id}"
            )
            self.outlet = StreamOutlet(depth_info)
            
            print(f"Initialized RealSense depth camera {self.config.name}")
            return True
            
        except ImportError:
            print("pyrealsense2 not available, falling back to regular camera")
            return super().initialize()
        except Exception as e:
            print(f"Error initializing RealSense camera: {e}")
            return False
    
    def _stream_loop(self):
        """RealSense specific streaming loop"""
        if not self.pipeline:
            super()._stream_loop()
            return
            
        import pyrealsense2 as rs
        target_interval = 1.0 / self.config.fps
        
        while self.running:
            start_time = time.time()
            
            try:
                frames = self.pipeline.wait_for_frames(timeout_ms=1000)
                depth_frame = frames.get_depth_frame()
                
                if not depth_frame:
                    continue
                
                # Convert to numpy array
                depth_image = np.asanyarray(depth_frame.get_data())
                depth_data = depth_image.astype(np.float32).flatten()
                
                timestamp = local_clock()
                self.outlet.push_sample(depth_data, timestamp)
                
            except Exception as e:
                print(f"Error processing RealSense frame: {e}")
            
            # Maintain target framerate
            elapsed = time.time() - start_time
            sleep_time = max(0, target_interval - elapsed)
            time.sleep(sleep_time)
    
    def cleanup(self):
        """Clean up RealSense resources"""
        self.stop_streaming()
        if self.pipeline:
            try:
                self.pipeline.stop()
            except:
                pass
        print(f"RealSense camera {self.config.name} cleaned up")