#!/usr/bin/env python3
"""
Full LSL Stream Logger with Enhanced Depth Visualization
Logs all available LSL streams to files with organized folder structure
and shows real-time video/depth display with clearer depth mapping.
"""

import argparse
import csv
import os
import time
import threading
import signal
import sys
from datetime import datetime, timedelta
from typing import Dict, Optional

import numpy as np
import cv2
import wave

from pylsl import StreamInlet, resolve_streams, local_clock
from config_manager import ConfigManager

class StreamLogger:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_manager = ConfigManager(config_path)
        self.running = False
        self.inlets: Dict[str, Dict] = {}
        self.log_threads: list[threading.Thread] = []
        self.log_folder = self._create_log_folder()

        # Data files
        self.csv_files = {}
        self.csv_writers = {}
        self.video_writers = {}
        self.audio_files = {}

        # Statistics tracking
        self.stats = {}
        self.start_time = None
        self.stats_lock = threading.Lock()

        # Visual display
        self.display_frames = {}
        self.display_lock = threading.Lock()
        self.display_enabled = True

        # Setup signal handling
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)

    def _signal_handler(self, signum, frame):
        print(f"\nReceived signal {signum}, stopping logging...")
        self.stop_logging()
        sys.exit(0)

    def _create_log_folder(self) -> str:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        folder_name = f"{self.config_manager.experiment_name}_{self.config_manager.participant_id}_{timestamp}"
        log_folder = os.path.join("logs", folder_name)
        os.makedirs(os.path.join(log_folder, "video"), exist_ok=True)
        os.makedirs(os.path.join(log_folder, "audio"), exist_ok=True)
        os.makedirs(os.path.join(log_folder, "data"), exist_ok=True)
        os.makedirs(os.path.join(log_folder, "frames"), exist_ok=True)
        os.makedirs(os.path.join(log_folder, "depth"), exist_ok=True)
        print(f"Created log folder: {log_folder}")
        return log_folder

    def _init_stream_stats(self, stream_name: str, stream_type: str):
        with self.stats_lock:
            self.stats[stream_name] = {
                'type': stream_type,
                'samples': 0,
                'frames': 0,
                'bytes': 0,
                'last_timestamp': None,
                'fps': 0.0,
                'sample_rate': 0.0,
                'errors': 0
            }

    def _update_stats(self, stream_name: str, **kwargs):
        with self.stats_lock:
            if stream_name in self.stats:
                for key, value in kwargs.items():
                    if key in ['samples', 'frames', 'bytes', 'errors']:
                        self.stats[stream_name][key] += value
                    else:
                        self.stats[stream_name][key] = value

    def discover_and_connect_streams(self) -> bool:
        print("Discovering LSL streams...")
        streams = resolve_streams()
        if not streams:
            print("No LSL streams found!")
            return False

        for stream in streams:
            stream_name = stream.name()
            stream_type = stream.type()
            try:
                inlet = StreamInlet(stream)
                self.inlets[stream_name] = {'inlet': inlet, 'type': stream_type, 'info': stream}
                self._init_stream_stats(stream_name, stream_type)
                print(f"Connected: {stream_name} ({stream_type})")
            except Exception as e:
                print(f"Error connecting to {stream_name}: {e}")

        return len(self.inlets) > 0

    def _setup_csv_logger(self, stream_name: str, stream_type: str, channel_count: int):
        csv_path = os.path.join(self.log_folder, "data", f"{stream_name}.csv")
        csv_file = open(csv_path, 'w', newline='')

        if stream_type == 'Physiology' or stream_type not in ['Video', 'Depth', 'Audio']:
            headers = ['timestamp'] + [f'channel_{i}' for i in range(channel_count)]
        elif stream_type == 'Audio':
            headers = ['timestamp'] + [f'audio_sample_{i}' for i in range(channel_count)]
        elif stream_type == 'Video':
            headers = ['timestamp', 'frame_path', 'jpeg_size']
        elif stream_type == 'Depth':
            headers = ['timestamp', 'frame_path', 'min_depth', 'max_depth', 'mean_depth']

        writer = csv.DictWriter(csv_file, fieldnames=headers)
        writer.writeheader()
        self.csv_files[stream_name] = csv_file
        self.csv_writers[stream_name] = writer

    def _setup_video_logger(self, stream_name: str):
        video_path = os.path.join(self.log_folder, "video", f"{stream_name}.avi")
        self.video_writers[stream_name] = {'path': video_path, 'writer': None, 'frame_count': 0, 'last_time': time.time()}

    def _setup_depth_logger(self, stream_name: str, width: int = 1280, height: int = 800):
        depth_folder = os.path.join(self.log_folder, "depth", stream_name)
        os.makedirs(depth_folder, exist_ok=True)
        self.video_writers[stream_name] = {'folder': depth_folder, 'width': width, 'height': height, 'frame_count': 0, 'last_time': time.time()}

    def _setup_audio_logger(self, stream_name: str, sample_rate: int = 22050):
        audio_path = os.path.join(self.log_folder, "audio", f"{stream_name}.wav")
        wav_file = wave.open(audio_path, 'wb')
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        self.audio_files[stream_name] = {'file': wav_file, 'last_time': time.time(), 'sample_count': 0}

    def _update_display_frame(self, stream_name: str, frame: np.ndarray):
        with self.display_lock:
            max_width = 960
            h, w = frame.shape[:2]
            if w > max_width:
                scale = max_width / w
                frame = cv2.resize(frame, (int(w*scale), int(h*scale)))
            self.display_frames[stream_name] = frame.copy()

    def _display_stats(self):
        while self.running:
            try:
                os.system('cls' if os.name == 'nt' else 'clear')
                elapsed = time.time() - self.start_time if self.start_time else 0
                print(f"Elapsed Time: {str(timedelta(seconds=int(elapsed)))}")
                with self.stats_lock:
                    for name, stat in self.stats.items():
                        print(f"{name}: {stat}")
                time.sleep(1)
            except Exception as e:
                print(f"Display error: {e}")

    def _display_visual_feeds(self):
        while self.running:
            with self.display_lock:
                for name, frame in self.display_frames.items():
                    if frame is not None:
                        cv2.imshow(name, frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                self.running = False
                break
        cv2.destroyAllWindows()

    # --- Logging Methods ---
    def _log_video_stream(self, name, inlet):
        frame_count = 0
        while self.running:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                try:
                    jpeg_bytes = bytes.fromhex(sample[0])
                    nparr = np.frombuffer(jpeg_bytes, np.uint8)
                    frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
                    if frame is not None:
                        self._update_display_frame(name, frame)
                        self.csv_writers[name].writerow({'timestamp': timestamp, 'frame_path': f'{name}_frame_{frame_count}.jpg', 'jpeg_size': len(jpeg_bytes)})
                        self.csv_files[name].flush()
                        cv2.imwrite(os.path.join(self.log_folder, 'frames', f'{name}_frame_{frame_count}.jpg'), frame)
                        frame_count += 1
                except Exception as e:
                    self._update_stats(name, errors=1)

    def _log_depth_stream(self, name, inlet):
        info = self.video_writers[name]
        width, height = info['width'], info['height']
        frame_count = 0
        while self.running:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                try:
                    depth_array = np.array(sample, dtype=np.float32).reshape((height, width))
                    # Improved depth normalization and visualization
                    min_val, max_val = np.min(depth_array), np.max(depth_array)
                    depth_clipped = np.clip(depth_array, min_val, max_val)
                    normalized_depth = ((depth_clipped - min_val) / (max_val - min_val + 1e-6)) * 255.0
                    normalized_depth = normalized_depth.astype(np.uint8)
                    depth_colorized = cv2.applyColorMap(normalized_depth, cv2.COLORMAP_TURBO)
                    depth_enhanced = cv2.equalizeHist(cv2.cvtColor(depth_colorized, cv2.COLOR_BGR2GRAY))
                    depth_display = cv2.applyColorMap(depth_enhanced, cv2.COLORMAP_TURBO)
                    self._update_display_frame(name, depth_display)
                    np.save(os.path.join(info['folder'], f'depth_{frame_count}.npy'), depth_array)
                    self.csv_writers[name].writerow({
                        'timestamp': timestamp,
                        'frame_path': f'depth_{frame_count}.npy',
                        'min_depth': float(min_val),
                        'max_depth': float(max_val),
                        'mean_depth': float(np.mean(depth_array))
                    })
                    self.csv_files[name].flush()
                    frame_count += 1
                except Exception as e:
                    self._update_stats(name, errors=1)

    def _log_audio_stream(self, name, inlet):
        info = self.audio_files[name]
        while self.running:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                try:
                    arr = np.array(sample)
                    arr_int16 = (arr * 32767).astype(np.int16)
                    info['file'].writeframes(arr_int16.tobytes())
                    row = {'timestamp': timestamp}
                    for i, v in enumerate(arr):
                        row[f'audio_sample_{i}'] = v
                    self.csv_writers[name].writerow(row)
                    self.csv_files[name].flush()
                except Exception as e:
                    self._update_stats(name, errors=1)

    def _log_data_stream(self, name, inlet):
        while self.running:
            sample, timestamp = inlet.pull_sample(timeout=1.0)
            if sample:
                try:
                    row = {'timestamp': timestamp}
                    for i, v in enumerate(sample):
                        row[f'channel_{i}'] = v
                    self.csv_writers[name].writerow(row)
                    self.csv_files[name].flush()
                except Exception as e:
                    self._update_stats(name, errors=1)

    # --- Start/Stop ---
    def start_logging(self):
        if not self.inlets:
            print("No streams connected!")
            return False
        self.running = True
        self.start_time = time.time()

        threading.Thread(target=self._display_stats, daemon=True).start()
        if self.display_enabled:
            threading.Thread(target=self._display_visual_feeds, daemon=True).start()

        for name, info in self.inlets.items():
            inlet = info['inlet']
            stream_type = info['type']
            channel_count = inlet.info().channel_count()
            self._setup_csv_logger(name, stream_type, channel_count)

            if stream_type == 'Video':
                self._setup_video_logger(name)
                t = threading.Thread(target=self._log_video_stream, args=(name, inlet), daemon=True)
            elif stream_type == 'Audio':
                self._setup_audio_logger(name)
                t = threading.Thread(target=self._log_audio_stream, args=(name, inlet), daemon=True)
            elif stream_type == 'Depth':
                self._setup_depth_logger(name)
                t = threading.Thread(target=self._log_depth_stream, args=(name, inlet), daemon=True)
            else:
                t = threading.Thread(target=self._log_data_stream, args=(name, inlet), daemon=True)
            t.start()
            self.log_threads.append(t)
        return True

    def stop_logging(self):
        self.running = False
        for t in self.log_threads:
            t.join(timeout=2.0)
        for f in self.csv_files.values():
            f.close()
        for v in self.video_writers.values():
            if 'writer' in v and v['writer']:
                v['writer'].release()
        for a in self.audio_files.values():
            a['file'].close()
        os.system('cls' if os.name == 'nt' else 'clear')

    def run(self, duration: Optional[float] = None):
        if not self.discover_and_connect_streams():
            return
        if not self.start_logging():
            return
        try:
            if duration:
                time.sleep(duration)
            else:
                while self.running:
                    time.sleep(1)
        finally:
            self.stop_logging()

# --- Main ---
def main():
    parser = argparse.ArgumentParser(description='LSL Stream Logger with Enhanced Depth Visualization')
    parser.add_argument('--config', '-c', default='config.yaml')
    parser.add_argument('--duration', '-d', type=float, help='Logging duration in seconds')
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
