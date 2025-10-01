#!/usr/bin/env python3
"""
Main LSL Streamer for Multi-Modal Data Capture
CLaSP / Eager Project

This script initializes and manages all data streams for a single participant:
- Multiple camera streams (RGB + Depth)
- Audio stream
- BioHarness physiological data

Usage:
    python main_streamer.py [--config CONFIG_FILE] [--simulate]
"""

import argparse
import signal
import sys
import time
from typing import List, Optional
import threading

from config_manager import ConfigManager
from camera_handler import CameraHandler, DepthCameraHandler
from audio_handler import AudioHandler, CameraAudioExtractor
from integrated_bioharness import BioHarnessHandler


class MultiModalStreamer:
    """Main class for managing all data streams for a participant"""
    
    def __init__(self, config_path: str = "config.yaml", simulate: bool = False):
        self.config_manager = ConfigManager(config_path)
        self.simulate = simulate
        self.running = False
        
        # Stream handlers
        self.camera_handlers: List[CameraHandler] = []
        self.audio_handler: Optional[AudioHandler] = None
        self.camera_audio_handler: Optional[CameraAudioExtractor] = None
        self.bioharness_handler: Optional[BioHarnessHandler] = None
        
        # Setup signal handling for graceful shutdown
        signal.signal(signal.SIGINT, self._signal_handler)
        signal.signal(signal.SIGTERM, self._signal_handler)
    
    def _signal_handler(self, signum, frame):
        """Handle shutdown signals gracefully"""
        print(f"\nReceived signal {signum}, shutting down...")
        self.stop_all_streams()
        sys.exit(0)
    
    def initialize_all_streams(self) -> bool:
        """Initialize all stream handlers"""
        print("Initializing multi-modal data streams...")
        
        # Validate configuration
        if not self.config_manager.validate_config():
            print("Configuration validation failed")
            return False
        
        success = True
        
        # Initialize camera streams
        success &= self._initialize_cameras()
        
        # Initialize audio stream
        # success &= self._initialize_audio()
        
        # Initialize BioHarness stream
        success &= self._initialize_bioharness()
        
        if success:
            print(f"All streams initialized for participant {self.config_manager.participant_id}")
        else:
            print("Some streams failed to initialize")
        
        return success
    
    def _initialize_cameras(self) -> bool:
        """Initialize all camera streams"""
        cameras = self.config_manager.cameras
        if not cameras:
            print("No cameras configured")
            return True
        
        success = True
        for camera_config in cameras:
            try:
                if camera_config.stream_type == "Depth":
                    handler = DepthCameraHandler(camera_config, self.config_manager.participant_id)
                else:
                    handler = CameraHandler(camera_config, self.config_manager.participant_id)
                
                if handler.initialize():
                    self.camera_handlers.append(handler)
                    print(f"✓ Camera {camera_config.name} initialized")
                else:
                    print(f"✗ Failed to initialize camera {camera_config.name}")
                    success = False
                    
            except Exception as e:
                print(f"✗ Error initializing camera {camera_config.name}: {e}")
                success = False
        
        return success
    
    def _initialize_audio(self) -> bool:
        """Initialize audio stream"""
        mic_config = self.config_manager.microphone
        
        try:
            # Initialize microphone audio
            self.audio_handler = AudioHandler(mic_config, self.config_manager.participant_id)
            if self.audio_handler.initialize():
                print(f"✓ Microphone audio initialized")
                
                # Also initialize camera audio extractor if requested
                self.camera_audio_handler = CameraAudioExtractor(
                    mic_config.source, 
                    self.config_manager.participant_id,
                    mic_config.sample_rate
                )
                if self.camera_audio_handler.initialize():
                    print(f"✓ Camera audio extractor initialized for {mic_config.source}")
                else:
                    print(f"✗ Camera audio extractor failed for {mic_config.source}")
                
                return True
            else:
                print("✗ Failed to initialize microphone audio")
                return False
                
        except Exception as e:
            print(f"✗ Error initializing audio: {e}")
            return False
    
    def _initialize_bioharness(self) -> bool:
        """Initialize BioHarness stream"""
        bio_config = self.config_manager.bioharness
        
        try:
            
            self.bioharness_handler = BioHarnessHandler(
                bio_config, self.config_manager.participant_id
            )
            
            if self.bioharness_handler.initialize():
                print(f"✓ BioHarness initialized ({'simulated' if self.simulate else 'real'})")
                return True
            else:
                print("✗ Failed to initialize BioHarness")
                return False
                
        except Exception as e:
            print(f"✗ Error initializing BioHarness: {e}")
            return False
    
    def start_all_streams(self):
        """Start all initialized streams"""
        if self.running:
            print("Streams are already running")
            return
        
        print("\nStarting all data streams...")
        self.running = True
        
        # Start camera streams
        for handler in self.camera_handlers:
            handler.start_streaming()
        
        # Start audio streams
        if self.audio_handler:
            self.audio_handler.start_streaming()
        
        if self.camera_audio_handler:
            self.camera_audio_handler.start_streaming()
        
        # Start BioHarness stream
        if self.bioharness_handler:
            self.bioharness_handler.start_streaming()
        
        print(f"All streams started for participant {self.config_manager.participant_id}")
        self._print_stream_info()
    
    def stop_all_streams(self):
        """Stop all running streams"""
        if not self.running:
            return
        
        print("\nStopping all data streams...")
        self.running = False
        
        # Stop all handlers
        for handler in self.camera_handlers:
            handler.stop_streaming()
        
        if self.audio_handler:
            self.audio_handler.stop_streaming()
        
        if self.camera_audio_handler:
            self.camera_audio_handler.stop_streaming()
        
        if self.bioharness_handler:
            self.bioharness_handler.stop_streaming()
        
        print("All streams stopped")
    
    def cleanup_all(self):
        """Clean up all resources"""
        print("\nCleaning up resources...")
        
        for handler in self.camera_handlers:
            handler.cleanup()
        
        if self.audio_handler:
            self.audio_handler.cleanup()
        
        if self.camera_audio_handler:
            self.camera_audio_handler.cleanup()
        
        if self.bioharness_handler:
            self.bioharness_handler.cleanup()
        
        print("Cleanup complete")
    
    def _print_stream_info(self):
        """Print information about active streams"""
        print("\n" + "="*60)
        print(f"ACTIVE LSL STREAMS - {self.config_manager.experiment_name}")
        print(f"Participant: {self.config_manager.participant_id}")
        print(f"Hostname: {self.config_manager.hostname}")
        print("="*60)
        
        # Camera streams
        for handler in self.camera_handlers:
            print(f"📹 {handler.video_stream_name} ({handler.config.stream_type})")
            print(f"   Resolution: {handler.config.resolution}, FPS: {handler.config.fps}")
        
        # Audio streams
        if self.audio_handler:
            print(f"🎤 {self.audio_handler.stream_name}")
            print(f"   Sample Rate: {self.audio_handler.config.sample_rate} Hz")
        
        if self.camera_audio_handler:
            print(f"🎤 {self.camera_audio_handler.stream_name} (from camera)")
            print(f"   Sample Rate: {self.camera_audio_handler.sample_rate} Hz")
        
        # BioHarness stream
        if self.bioharness_handler:
            print(f"💓 {self.bioharness_handler.stream_name}")
            print(f"   Channels: {len(self.bioharness_handler.channels)}")
            print(f"   Sample Rate: {self.bioharness_handler.config.sample_rate} Hz")
            if self.simulate:
                print("   Mode: SIMULATED DATA")
        
        print("="*60)
        print("Streams are now broadcasting via LSL network")
        print("Use LSL viewer or data collection scripts to receive data")
        print("Press Ctrl+C to stop all streams")
        print("="*60)
    
    def run_interactive(self):
        """Run in interactive mode with user prompts"""
        print(f"\n🚀 Multi-Modal LSL Streamer")
        print(f"Experiment: {self.config_manager.experiment_name}")
        print(f"Participant: {self.config_manager.participant_id}")
        
        if not self.initialize_all_streams():
            print("Failed to initialize streams. Exiting.")
            return
        
        input("\nPress Enter to start streaming...")
        self.start_all_streams()
        
        try:
            while self.running:
                time.sleep(1)
        except KeyboardInterrupt:
            pass
        finally:
            self.stop_all_streams()
            self.cleanup_all()
    
    def get_stream_status(self) -> dict:
        """Get status of all streams"""
        status = {
            'participant_id': self.config_manager.participant_id,
            'experiment': self.config_manager.experiment_name,
            'running': self.running,
            'streams': {
                'cameras': len(self.camera_handlers),
                'audio': self.audio_handler is not None,
                'camera_audio': self.camera_audio_handler is not None,
                'bioharness': self.bioharness_handler is not None,
                'simulation_mode': self.simulate
            }
        }
        return status


def main():
    parser = argparse.ArgumentParser(description='Multi-Modal LSL Data Streamer')
    parser.add_argument('--config', '-c', default='config.yaml',
                        help='Path to configuration file (default: config.yaml)')
    parser.add_argument('--simulate', '-s', action='store_true',
                        help='Use simulated data for BioHarness')
    parser.add_argument('--non-interactive', action='store_true',
                        help='Run without user prompts (for scripts)')
    
    args = parser.parse_args()
    
    try:
        streamer = MultiModalStreamer(args.config, args.simulate)
        
        if args.non_interactive:
            # For automated/scripted use
            if streamer.initialize_all_streams():
                streamer.start_all_streams()
                print("Streams started. Running until interrupted...")
                try:
                    while streamer.running:
                        time.sleep(1)
                except KeyboardInterrupt:
                    pass
                finally:
                    streamer.stop_all_streams()
                    streamer.cleanup_all()
            else:
                print("Failed to initialize streams")
                sys.exit(1)
        else:
            # Interactive mode
            streamer.run_interactive()
    
    except Exception as e:
        print(f"Error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()