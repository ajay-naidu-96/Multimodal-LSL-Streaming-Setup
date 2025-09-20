import pyaudio
import numpy as np
import threading
import time
from typing import Optional
from pylsl import StreamOutlet, StreamInfo, local_clock
from config_manager import MicrophoneConfig

class AudioHandler:
    def __init__(self, mic_config: MicrophoneConfig, participant_id: str):
        self.config = mic_config
        self.participant_id = participant_id
        self.audio = None
        self.stream = None
        self.outlet = None
        self.running = False
        self.thread = None
        
        # Audio parameters
        self.chunk_size = 1024  # Number of frames per buffer
        self.format = pyaudio.paFloat32
        
        # Stream name
        self.stream_name = f"Audio_{self.config.source}_{participant_id}"
    
    def initialize(self) -> bool:
        """Initialize audio input and LSL outlet"""
        try:
            # Initialize PyAudio
            self.audio = pyaudio.PyAudio()
            
            # Find audio device (this is a simplified version)
            device_index = self._find_audio_device()
            if device_index is None:
                print("Could not find suitable audio device")
                return False
            
            # Create audio stream
            self.stream = self.audio.open(
                format=self.format,
                channels=self.config.channels,
                rate=self.config.sample_rate,
                input=True,
                input_device_index=device_index,
                frames_per_buffer=self.chunk_size
            )
            
            # Create LSL outlet
            audio_info = StreamInfo(
                self.stream_name,
                'Audio',
                channel_count=self.config.channels,
                nominal_srate=self.config.sample_rate,
                channel_format='float32',
                source_id=f"{self.stream_name}_mic"
            )
            
            self.outlet = StreamOutlet(audio_info)
            print(f"Initialized audio handler for {self.config.source}")
            return True
            
        except Exception as e:
            print(f"Error initializing audio handler: {e}")
            return False
    
    def _find_audio_device(self) -> Optional[int]:
        """Find suitable audio input device"""
        try:
            # Get default input device
            default_device = self.audio.get_default_input_device_info()
            device_index = default_device['index']
            
            # Check if device supports our requirements
            device_info = self.audio.get_device_info_by_index(device_index)
            if (device_info['maxInputChannels'] >= self.config.channels and
                self.config.sample_rate <= device_info['defaultSampleRate']):
                return device_index
            
            # If default doesn't work, search for suitable device
            for i in range(self.audio.get_device_count()):
                device_info = self.audio.get_device_info_by_index(i)
                if (device_info['maxInputChannels'] >= self.config.channels and
                    self.config.sample_rate <= device_info['defaultSampleRate']):
                    return i
            
            return None
            
        except Exception as e:
            print(f"Error finding audio device: {e}")
            return None
    
    def start_streaming(self):
        """Start audio streaming in a separate thread"""
        if self.running:
            print("Audio is already streaming")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        print(f"Started audio streaming for {self.config.source}")
    
    def stop_streaming(self):
        """Stop audio streaming"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        print(f"Stopped audio streaming for {self.config.source}")
    
    def _stream_loop(self):
        """Main audio streaming loop"""
        while self.running and self.stream:
            try:
                # Read audio data
                data = self.stream.read(self.chunk_size, exception_on_overflow=False)
                audio_data = np.frombuffer(data, dtype=np.float32)
                
                # Reshape for multi-channel audio
                if self.config.channels > 1:
                    audio_data = audio_data.reshape(-1, self.config.channels)
                
                # Send each frame with timestamp
                timestamp = local_clock()
                
                if self.config.channels == 1:
                    # Mono audio - send samples one by one
                    for sample in audio_data:
                        self.outlet.push_sample([sample], timestamp)
                        timestamp += 1.0 / self.config.sample_rate
                else:
                    # Multi-channel audio
                    for frame in audio_data:
                        self.outlet.push_sample(frame.tolist(), timestamp)
                        timestamp += 1.0 / self.config.sample_rate
                
            except Exception as e:
                if self.running:  # Only print error if we're supposed to be running
                    print(f"Error in audio streaming: {e}")
                break
    
    def cleanup(self):
        """Clean up audio resources"""
        self.stop_streaming()
        
        if self.stream:
            self.stream.stop_stream()
            self.stream.close()
        
        if self.audio:
            self.audio.terminate()
        
        print("Audio handler cleaned up")


class CameraAudioExtractor:
    """Extract audio from camera feed (placeholder implementation)"""
    
    def __init__(self, camera_name: str, participant_id: str, sample_rate: int = 44100):
        self.camera_name = camera_name
        self.participant_id = participant_id
        self.sample_rate = sample_rate
        self.outlet = None
        self.running = False
        self.thread = None
        
        self.stream_name = f"Audio_{camera_name}_{participant_id}"
    
    def initialize(self) -> bool:
        """Initialize camera audio extraction"""
        try:
            # Create LSL outlet for camera audio
            audio_info = StreamInfo(
                self.stream_name,
                'Audio',
                channel_count=1,
                nominal_srate=self.sample_rate,
                channel_format='float32',
                source_id=f"{self.stream_name}_camera"
            )
            
            self.outlet = StreamOutlet(audio_info)
            print(f"Initialized camera audio extractor for {self.camera_name}")
            return True
            
        except Exception as e:
            print(f"Error initializing camera audio extractor: {e}")
            return False
    
    def start_streaming(self):
        """Start extracting audio from camera"""
        if self.running:
            print("Camera audio extraction already running")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._extraction_loop, daemon=True)
        self.thread.start()
        print(f"Started camera audio extraction for {self.camera_name}")
    
    def stop_streaming(self):
        """Stop audio extraction"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        print(f"Stopped camera audio extraction for {self.camera_name}")
    
    def _extraction_loop(self):
        """Camera audio extraction loop (placeholder)"""
        # This is a placeholder implementation
        # In reality, you would need to:
        # 1. Access the camera's audio stream (if available)
        # 2. Extract audio samples
        # 3. Send them via LSL
        
        chunk_duration = 1.0 / self.sample_rate
        
        while self.running:
            try:
                # Generate placeholder audio data (silence)
                # In real implementation, this would be actual camera audio
                audio_sample = [0.0]  # Silent sample
                timestamp = local_clock()
                
                self.outlet.push_sample(audio_sample, timestamp)
                time.sleep(chunk_duration)
                
            except Exception as e:
                if self.running:
                    print(f"Error in camera audio extraction: {e}")
                break
    
    def cleanup(self):
        """Clean up resources"""
        self.stop_streaming()
        print(f"Camera audio extractor for {self.camera_name} cleaned up")