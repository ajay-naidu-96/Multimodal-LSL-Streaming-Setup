import yaml
import os
from typing import Dict, List, Any
from dataclasses import dataclass

@dataclass
class CameraConfig:
    id: int
    name: str
    stream_type: str
    resolution: List[int]
    fps: int

@dataclass
class MicrophoneConfig:
    source: str
    sample_rate: int
    channels: int

@dataclass
class BioHarnessConfig:
    device_id: str
    stream_name: str
    sample_rate: int

@dataclass
class LSLConfig:
    chunk_size: int
    max_buffered: int
    stream_timeout: float

class ConfigManager:
    def __init__(self, config_path: str = "config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if not os.path.exists(self.config_path):
            raise FileNotFoundError(f"Configuration file not found: {self.config_path}")
        
        with open(self.config_path, 'r') as f:
            config = yaml.safe_load(f)
        
        return config
    
    @property
    def experiment_name(self) -> str:
        return self.config.get('experiment_name', 'DefaultExperiment')
    
    @property
    def participant_id(self) -> str:
        return self.config.get('participant_id', 'P1')
    
    @property
    def hostname(self) -> str:
        return self.config.get('hostname', 'localhost')
    
    @property
    def cameras(self) -> List[CameraConfig]:
        camera_configs = []
        for cam_config in self.config.get('cameras', []):
            camera_configs.append(CameraConfig(
                id=cam_config['id'],
                name=cam_config['name'],
                stream_type=cam_config['stream_type'],
                resolution=cam_config.get('resolution', [640, 480]),
                fps=cam_config.get('fps', 30)
            ))
        return camera_configs
    
    @property
    def microphone(self) -> MicrophoneConfig:
        mic_config = self.config.get('microphone', {})
        return MicrophoneConfig(
            source=mic_config.get('source', 'RGB1'),
            sample_rate=mic_config.get('sample_rate', 44100),
            channels=mic_config.get('channels', 1)
        )
    
    @property
    def bioharness(self) -> BioHarnessConfig:
        bio_config = self.config.get('bioharness', {})
        return BioHarnessConfig(
            device_id=bio_config.get('device_id', 'BH00000'),
            stream_name=bio_config.get('stream_name', 'BioHarness'),
            sample_rate=bio_config.get('sample_rate', 250)
        )
    
    @property
    def lsl_config(self) -> LSLConfig:
        lsl_config = self.config.get('lsl_config', {})
        return LSLConfig(
            chunk_size=lsl_config.get('chunk_size', 32),
            max_buffered=lsl_config.get('max_buffered', 360),
            stream_timeout=lsl_config.get('stream_timeout', 5.0)
        )
    
    def get_stream_name(self, modality: str) -> str:
        """Generate LSL stream name with participant ID"""
        return f"{modality}_{self.participant_id}"
    
    def validate_config(self) -> bool:
        """Validate configuration parameters"""
        try:
            # Check required fields
            assert self.experiment_name, "experiment_name is required"
            assert self.participant_id, "participant_id is required"
            
            # Validate cameras
            camera_names = [cam.name for cam in self.cameras]
            assert len(camera_names) == len(set(camera_names)), "Camera names must be unique"
            
            # Validate microphone source
            if self.microphone.source not in camera_names:
                print(f"Warning: Microphone source '{self.microphone.source}' not found in camera names")
            
            return True
        except AssertionError as e:
            print(f"Configuration validation error: {e}")
            return False