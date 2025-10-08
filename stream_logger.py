#!/usr/bin/env python3
"""
LSL Stream Logger
Logs all available LSL streams to files with organized folder structure

Usage:
    python stream_logger.py [--config CONFIG_FILE] [--duration SECONDS]
"""

import argparse
import csv
import os
import time
import threading
import signal
import sys
from datetime import datetime
from typing import Dict, List, Optional
import numpy as np
import cv2
import wave

from pylsl import StreamInlet, resolve_streams, local_clock
from config_manager import ConfigManager


class StreamLogger:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.running = False
        self.inlets: Dict[str, StreamInlet] = {}
        self.log_threads: List[threading.Thread] = []
        self.log_folder = self._create_log_folder()
        
        # Data files
        self.csv_files = {}
        self.csv_writers = {}
        self.video_writers = {}
        self.audio_files = {}
        
        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, stopping logging...")
        self.stop_logging()
        sys.exit(0)
    
    def _create_log_folder(self) -> str:
        """Create organized log folder structure based on config"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        folder_name = f"{self.config_manager.experiment_name}_{self.config_manager.participant_id}_{timestamp}"
        log_folder = os.path.join("logs", folder_name)
        
        # Create subfolders
        os.makedirs(log_folder, exist_ok=True)
        os.makedirs(os.path.join(log_folder, "video"), exist_ok=True)
        os.makedirs(os.path.join(log_folder, "audio"), exist_ok=True)
        os.makedirs(os.path.join(log_folder, "data"), exist_ok=True)
        os.makedirs(os.path.join(log_folder, "frames"), exist_ok=True)
        os.makedirs(os.path.join(log_folder, "depth"), exist_ok=True)
        
        print(f"Created log folder: {log_folder}")
        return log_folder
    
    def discover_and_connect_streams(self) -> bool:
        """Discover and connect to all available LSL streams"""
        print("Discovering LSL streams...")
        
        streams = resolve_streams()
        if not streams:
            print("No LSL streams found!")
            return False
        
        print(f"Found {len(streams)} streams:")
        
        connected_count = 0
        for stream in streams:
            stream_name = stream.name()
            stream_type = stream.type()
            
            try:
                inlet = StreamInlet(stream)
                self.inlets[stream_name] = {
                    'inlet': inlet,
                    'type': stream_type,
                    'info': stream
                }
                print(f"  Connected: {stream_name} ({stream_type})")
                connected_count += 1
                
            except Exception as e:
                print(f"  Error connecting to {stream_name}: {e}")
        
        if connected_count == 0:
            print("Failed to connect to any streams!")
            return False
        
        print(f"Successfully connected to {connected_count} streams")
        return True
    
    def _setup_csv_logger(self, stream_name: str, stream_type: str, channel_count: int):
        """Setup CSV logging for a stream"""
        csv_path = os.path.join(self.log_folder, "data", f"{stream_name}.csv")
        
        csv_file = open(csv_path, 'w', newline='')
        
        # Create headers based on stream type
        if stream_type == 'Physiology':
            # Use BioHarness channel names if available
            headers = ['timestamp'] + [f'channel_{i}' for i in range(channel_count)]
        elif stream_type == 'Audio':
            headers = ['timestamp'] + [f'audio_sample_{i}' for i in range(channel_count)]
        elif stream_type == 'Video':
            headers = ['timestamp', 'frame_path', 'jpeg_size']
        elif stream_type == 'Depth':
            headers = ['timestamp', 'frame_path', 'min_depth', 'max_depth', 'mean_depth']
        else:
            headers = ['timestamp'] + [f'data_{i}' for i in range(channel_count)]
        
        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        
        self.csv_files[stream_name] = csv_file
        self.csv_writers[stream_name] = writer
        
        print(f"  CSV logger setup for {stream_name}")
    
    def _setup_video_logger(self, stream_name: str):
        """Setup video logging for video streams"""
        video_path = os.path.join(self.log_folder, "video", f"{stream_name}.avi")
        
        # We'll set this up when we get the first frame to know dimensions
        self.video_writers[stream_name] = {
            'path': video_path,
            'writer': None,
            'frame_count': 0
        }
        
        print(f"  Video logger setup for {stream_name}")
    
    def _setup_depth_logger(self, stream_name: str, width: int = 1280, height: int = 800):
        """Setup depth stream logging"""
        depth_folder = os.path.join(self.log_folder, "depth", stream_name)
        os.makedirs(depth_folder, exist_ok=True)
        
        self.video_writers[stream_name] = {
            'folder': depth_folder,
            'width': width,
            'height': height,
            'frame_count': 0
        }
        
        print(f"  Depth logger setup for {stream_name}")
    
    def _setup_audio_logger(self, stream_name: str, sample_rate: int = 22050):
        """Setup WAV file logging for audio streams"""
        audio_path = os.path.join(self.log_folder, "audio", f"{stream_name}.wav")
        
        # Open WAV file for writing
        wav_file = wave.open(audio_path, 'wb')
        wav_file.setnchannels(1)  # Mono
        wav_file.setsampwidth(2)  # 16-bit
        wav_file.setframerate(sample_rate)
        
        self.audio_files[stream_name] = wav_file
        
        print(f"  Audio logger setup for {stream_name}")
    
    def _log_video_stream(self, stream_name: str, inlet: StreamInlet):
        """Log video stream data"""
        print(f"Starting video logging for {stream_name}")
        
        frame_folder = os.path.join(self.log_folder, "frames", stream_name)
        os.makedirs(frame_folder, exist_ok=True)
        
        frame_count = 0
        
        while self.running:
            try:
                sample, timestamp = inlet.pull_sample(timeout=1.0)
                if sample and len(sample) > 0:
                    try:
                        # Decode JPEG data
                        jpeg_hex = sample[0]
                        jpeg_bytes = bytes.fromhex(jpeg_hex)
                        
                        # Save raw frame
                        frame_filename = f"frame_{frame_count:06d}_{timestamp:.6f}.jpg"
                        frame_path = os.path.join(frame_folder, frame_filename)
                        
                        with open(frame_path, 'wb') as f:
                            f.write(jpeg_bytes)
                        
                        # Log to CSV
                        if stream_name in self.csv_writers:
                            self.csv_writers[stream_name].writerow({
                                'timestamp': timestamp,
                                'frame_path': frame_path,
                                'jpeg_size': len(jpeg_bytes)
                            })
                            self.csv_files[stream_name].flush()
                        
                        frame_count += 1
                        
                        # Optional: Also decode and save to video file
                        if stream_name in self.video_writers:
                            nparr = np.frombuffer(jpeg_bytes, np.uint8)
                            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                            
                            if frame is not None:
                                video_info = self.video_writers[stream_name]
                                
                                if video_info['writer'] is None:
                                    # Initialize video writer with frame dimensions
                                    h, w = frame.shape[:2]
                                    fourcc = cv2.VideoWriter_fourcc(*'XVID')
                                    video_info['writer'] = cv2.VideoWriter(
                                        video_info['path'], fourcc, 15.0, (w, h)
                                    )
                                
                                video_info['writer'].write(frame)
                                video_info['frame_count'] += 1
                        
                    except Exception as e:
                        print(f"Error processing video frame: {e}")
                        
            except Exception as e:
                if self.running:
                    print(f"Error in video logging for {stream_name}: {e}")
                    time.sleep(0.1)
    
    def _log_depth_stream(self, stream_name: str, inlet: StreamInlet):
        """Log depth stream data"""
        print(f"Starting depth logging for {stream_name}")
        
        depth_info = self.video_writers[stream_name]
        depth_folder = depth_info['folder']
        width = depth_info['width']
        height = depth_info['height']
        frame_count = 0
        
        while self.running:
            try:
                sample, timestamp = inlet.pull_sample(timeout=1.0)
                if sample:
                    # Reshape flat array back to image
                    depth_array = np.array(sample, dtype=np.float32).reshape((height, width))
                    
                    # Save as numpy file (preserves depth values)
                    depth_filename = f"depth_{frame_count:06d}_{timestamp:.6f}.npy"
                    depth_path = os.path.join(depth_folder, depth_filename)
                    np.save(depth_path, depth_array)
                    
                    # Also save visualization as PNG
                    depth_normalized = cv2.normalize(depth_array, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                    depth_colormap = cv2.applyColorMap(depth_normalized, cv2.COLORMAP_JET)
                    viz_path = depth_path.replace('.npy', '_viz.png')
                    cv2.imwrite(viz_path, depth_colormap)
                    
                    # Log to CSV
                    if stream_name in self.csv_writers:
                        self.csv_writers[stream_name].writerow({
                            'timestamp': timestamp,
                            'frame_path': depth_path,
                            'min_depth': float(np.min(depth_array)),
                            'max_depth': float(np.max(depth_array)),
                            'mean_depth': float(np.mean(depth_array))
                        })
                        self.csv_files[stream_name].flush()
                    
                    frame_count += 1
                    
                    if frame_count % 100 == 0:
                        print(f"[{stream_name}] Logged {frame_count} depth frames")
                    
            except Exception as e:
                if self.running:
                    print(f"Error in depth logging for {stream_name}: {e}")
                    import traceback
                    traceback.print_exc()
                    time.sleep(0.1)
    
    def _log_audio_stream(self, stream_name: str, inlet: StreamInlet):
        """Log audio stream data"""
        print(f"Starting audio logging for {stream_name}")
        
        while self.running:
            try:
                sample, timestamp = inlet.pull_sample(timeout=1.0)
                if sample:
                    # Log to CSV
                    if stream_name in self.csv_writers:
                        row = {'timestamp': timestamp}
                        for i, value in enumerate(sample):
                            row[f'audio_sample_{i}'] = value
                        
                        self.csv_writers[stream_name].writerow(row)
                        self.csv_files[stream_name].flush()
                    
                    # Log to WAV file
                    if stream_name in self.audio_files:
                        # Convert float sample to 16-bit integer
                        audio_sample = int(sample[0] * 32767)
                        audio_sample = max(-32768, min(32767, audio_sample))  # Clamp
                        self.audio_files[stream_name].writeframes(audio_sample.to_bytes(2, 'little', signed=True))
                
            except Exception as e:
                if self.running:
                    print(f"Error in audio logging for {stream_name}: {e}")
                    time.sleep(0.1)
    
    def _log_data_stream(self, stream_name: str, inlet: StreamInlet):
        """Log general data streams (like BioHarness)"""
        print(f"Starting data logging for {stream_name}")
        
        while self.running:
            try:
                sample, timestamp = inlet.pull_sample(timeout=1.0)
                if sample:
                    # Log to CSV
                    if stream_name in self.csv_writers:
                        row = {'timestamp': timestamp}
                        for i, value in enumerate(sample):
                            row[f'channel_{i}'] = value
                        
                        self.csv_writers[stream_name].writerow(row)
                        self.csv_files[stream_name].flush()
                
            except Exception as e:
                if self.running:
                    print(f"Error in data logging for {stream_name}: {e}")
                    time.sleep(0.1)
    
    def start_logging(self):
        """Start logging all connected streams"""
        if not self.inlets:
            print("No streams connected!")
            return False
        
        print(f"\nStarting logging to: {self.log_folder}")
        print("=" * 60)
        
        self.running = True
        
        for stream_name, stream_info in self.inlets.items():
            inlet = stream_info['inlet']
            stream_type = stream_info['type']
            
            # Get channel count
            info = inlet.info()
            channel_count = info.channel_count()
            
            # Setup appropriate loggers
            self._setup_csv_logger(stream_name, stream_type, channel_count)
            
            if stream_type == 'Video':
                self._setup_video_logger(stream_name)
                thread = threading.Thread(
                    target=self._log_video_stream, 
                    args=(stream_name, inlet), 
                    daemon=True
                )
            elif stream_type == 'Audio':
                self._setup_audio_logger(stream_name)
                thread = threading.Thread(
                    target=self._log_audio_stream, 
                    args=(stream_name, inlet), 
                    daemon=True
                )
            elif stream_type == 'Depth':
                self._setup_depth_logger(stream_name, width=1280, height=800)
                thread = threading.Thread(
                    target=self._log_depth_stream, 
                    args=(stream_name, inlet), 
                    daemon=True
                )
            else:
                thread = threading.Thread(
                    target=self._log_data_stream, 
                    args=(stream_name, inlet), 
                    daemon=True
                )
            
            thread.start()
            self.log_threads.append(thread)
        
        print(f"Started logging {len(self.log_threads)} streams")
        return True
    
    def stop_logging(self):
        """Stop all logging"""
        print("\nStopping logging...")
        self.running = False
        
        # Wait for threads to finish
        for thread in self.log_threads:
            if thread.is_alive():
                thread.join(timeout=2.0)
        
        # Close all files
        for csv_file in self.csv_files.values():
            csv_file.close()
        
        for video_info in self.video_writers.values():
            if 'writer' in video_info and video_info['writer']:
                video_info['writer'].release()
        
        for wav_file in self.audio_files.values():
            wav_file.close()
        
        print("All logging stopped and files closed")
        
        # Print summary
        self._print_summary()
    
    def _print_summary(self):
        """Print logging summary"""
        print("\n" + "=" * 60)
        print("LOGGING SUMMARY")
        print("=" * 60)
        print(f"Log folder: {self.log_folder}")
        print(f"Experiment: {self.config_manager.experiment_name}")
        print(f"Participant: {self.config_manager.participant_id}")
        
        # List created files
        for root, dirs, files in os.walk(self.log_folder):
            if files:
                rel_path = os.path.relpath(root, self.log_folder)
                if rel_path == '.':
                    rel_path = 'root'
                print(f"\n{rel_path}/")
                for file in files[:10]:  # Show first 10 files
                    file_path = os.path.join(root, file)
                    file_size = os.path.getsize(file_path)
                    print(f"  {file} ({file_size / 1024:.1f} KB)")
                if len(files) > 10:
                    print(f"  ... and {len(files) - 10} more files")
        
        print("=" * 60)
    
    def run(self, duration: Optional[float] = None):
        """Main run loop"""
        if not self.discover_and_connect_streams():
            return
        
        if not self.start_logging():
            return
        
        try:
            if duration:
                print(f"Logging for {duration} seconds...")
                time.sleep(duration)
            else:
                print("Logging indefinitely. Press Ctrl+C to stop.")
                while self.running:
                    time.sleep(1)
        except KeyboardInterrupt:
            print("\nInterrupted by user")
        finally:
            self.stop_logging()


def main():
    parser = argparse.ArgumentParser(description='LSL Stream Logger')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--duration', '-d', type=float,
                        help='Logging duration in seconds (default: unlimited)')
    
    args = parser.parse_args()
    
    try:
        logger = StreamLogger(args.config)
        logger.run(args.duration)
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()