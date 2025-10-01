#!/usr/bin/env python3
"""
Integrated BioHarness Handler using tested Zephyr library
Based on working code from physiological data collection system
"""

import threading
import time
import asyncio
import logging
from typing import Optional
from pylsl import StreamOutlet, StreamInfo, local_clock
from config_manager import BioHarnessConfig

# Import the working Zephyr BioHarness interface
try:
    from zephyr.core import BioHarness
    from zephyr.core.protocol import get_unit
    ZEPHYR_AVAILABLE = True
except ImportError:
    print("Warning: Zephyr BioHarness library not found. Install from working code.")
    ZEPHYR_AVAILABLE = False

logger = logging.getLogger(__name__)


class BioHarnessHandler:
    """Handler for BioHarness physiological data streaming using tested Zephyr library"""
    
    def __init__(self, bio_config: BioHarnessConfig, participant_id: str):
        self.config = bio_config
        self.participant_id = participant_id
        self.link = None
        self.outlets = {}  # Multiple outlets for different data types
        self.running = False
        self.thread = None
        self.loop = None
        
        # Stream name prefix
        self.stream_prefix = f"Zephyr_{participant_id}"
        
        if not ZEPHYR_AVAILABLE:
            raise ImportError("Zephyr BioHarness library is required but not found")
    
    def initialize(self) -> bool:
        """Initialize BioHarness connection and LSL outlets"""
        try:
            logger.info(f"Connecting to BioHarness at {self.config.device_id}")
            
            # Create event loop for async operations
            self.loop = asyncio.new_event_loop()
            
            # Connect to BioHarness (will auto-discover if device_id is empty/unknown)
            address = self.config.device_id if self.config.device_id not in ['unknown', 'BH12345', 'TEST'] else ''
            
            # Run async initialization
            success = self.loop.run_until_complete(self._async_init(address))
            
            if success:
                logger.info(f"Successfully initialized BioHarness for {self.config.device_id}")
                return True
            else:
                logger.error("Failed to initialize BioHarness")
                return False
                
        except Exception as e:
            logger.error(f"Error initializing BioHarness handler: {e}")
            return False
    
    async def _async_init(self, address: str) -> bool:
        """Async initialization of BioHarness connection"""
        try:
            # Create BioHarness link
            self.link = BioHarness(address, port=1, timeout=20)
            
            # Get device info
            infos = await self.link.get_infos()
            logger.info(f"BioHarness device info:")
            for key, val in infos.items():
                logger.info(f"  {key}: {val}")
            
            # Enable data streams we want
            await self._enable_streams()
            
            return True
            
        except Exception as e:
            logger.error(f"Async init failed: {e}")
            return False
    
    async def _enable_streams(self):
        """Enable various BioHarness data streams"""
        
        # Enable General data (heart rate, respiration, etc.)
        await self.link.toggle_general(self._on_general_data)
        logger.info("Enabled General data stream")
        
        # Enable Summary data (comprehensive physiological metrics)
        await self.link.toggle_summary(self._on_summary_data, ival=1)
        logger.info("Enabled Summary data stream")
        
        # Optionally enable other streams
        # await self.link.toggle_ecg(self._on_ecg_data)
        # await self.link.toggle_breathing(self._on_breathing_data)
        # await self.link.toggle_accel100mg(self._on_accel_data)
        # await self.link.toggle_rtor(self._on_rtor_data)
        # await self.link.toggle_events(self._on_events)
    
    def _create_outlet(self, stream_name: str, stream_type: str, channels: list) -> StreamOutlet:
        """Create LSL outlet for a specific data stream"""
        info = StreamInfo(
            f"{self.stream_prefix}_{stream_name}",
            stream_type,
            channel_count=len(channels),
            nominal_srate=1.0,  # Most BioHarness data is 1 Hz
            channel_format='float32',
            source_id=f"{self.stream_prefix}_{stream_name}_{self.config.device_id}"
        )
        
        # Add channel metadata
        desc = info.desc()
        desc.append_child_value('manufacturer', 'Medtronic')
        desc.append_child_value('model', 'Zephyr BioHarness')
        
        chns = desc.append_child('channels')
        for channel_name in channels:
            ch = chns.append_child('channel')
            ch.append_child_value('label', channel_name)
            unit = get_unit(channel_name)
            if unit:
                ch.append_child_value('unit', unit)
            ch.append_child_value('type', 'physiological')
        
        return StreamOutlet(info)
    
    def _on_general_data(self, msg):
        """Handler for general data packets"""
        try:
            # Create outlet if not exists
            if 'General' not in self.outlets:
                channels = list(msg.as_dict().keys())
                self.outlets['General'] = self._create_outlet('General', 'Misc', channels)
            
            # Send data
            data = list(msg.as_dict().values())
            timestamp = local_clock()
            self.outlets['General'].push_sample(data, timestamp)
            
        except Exception as e:
            logger.error(f"Error handling general data: {e}")
    
    def _on_summary_data(self, msg):
        """Handler for summary data packets"""
        try:
            # Create outlet if not exists
            if 'Summary' not in self.outlets:
                channels = list(msg.as_dict().keys())
                self.outlets['Summary'] = self._create_outlet('Summary', 'Misc', channels)
            
            # Send data
            data = list(msg.as_dict().values())
            timestamp = local_clock()
            self.outlets['Summary'].push_sample(data, timestamp)
            
        except Exception as e:
            logger.error(f"Error handling summary data: {e}")
    
    def _on_ecg_data(self, msg):
        """Handler for ECG waveform data"""
        try:
            if 'ECG' not in self.outlets:
                info = StreamInfo(
                    f"{self.stream_prefix}_ECG",
                    'ECG',
                    1,
                    nominal_srate=250,  # ECG is 250 Hz
                    channel_format='float32',
                    source_id=f"{self.stream_prefix}_ECG"
                )
                self.outlets['ECG'] = StreamOutlet(info)
            
            # Push ECG waveform samples
            self.outlets['ECG'].push_chunk([[v] for v in msg.waveform])
            
        except Exception as e:
            logger.error(f"Error handling ECG data: {e}")
    
    def _on_breathing_data(self, msg):
        """Handler for breathing waveform data"""
        try:
            if 'Breathing' not in self.outlets:
                info = StreamInfo(
                    f"{self.stream_prefix}_Breathing",
                    'Respiration',
                    1,
                    nominal_srate=18,  # Breathing waveform rate
                    channel_format='float32',
                    source_id=f"{self.stream_prefix}_Breathing"
                )
                self.outlets['Breathing'] = StreamOutlet(info)
            
            # Push breathing waveform samples
            self.outlets['Breathing'].push_chunk([[v] for v in msg.waveform])
            
        except Exception as e:
            logger.error(f"Error handling breathing data: {e}")
    
    def _on_accel_data(self, msg):
        """Handler for accelerometer data"""
        try:
            if 'Accel' not in self.outlets:
                info = StreamInfo(
                    f"{self.stream_prefix}_Accel",
                    'Mocap',
                    3,  # X, Y, Z
                    nominal_srate=50,
                    channel_format='float32',
                    source_id=f"{self.stream_prefix}_Accel"
                )
                self.outlets['Accel'] = StreamOutlet(info)
            
            # Push accelerometer data (X, Y, Z)
            samples = [[x, y, z] for x, y, z in zip(msg.accel_x, msg.accel_y, msg.accel_z)]
            self.outlets['Accel'].push_chunk(samples)
            
        except Exception as e:
            logger.error(f"Error handling accel data: {e}")
    
    def _on_rtor_data(self, msg):
        """Handler for R-to-R interval data"""
        try:
            if 'RtoR' not in self.outlets:
                info = StreamInfo(
                    f"{self.stream_prefix}_RtoR",
                    'Misc',
                    1,
                    nominal_srate=18,
                    channel_format='float32',
                    source_id=f"{self.stream_prefix}_RtoR"
                )
                self.outlets['RtoR'] = StreamOutlet(info)
            
            # Push R-to-R intervals
            self.outlets['RtoR'].push_chunk([[v] for v in msg.waveform])
            
        except Exception as e:
            logger.error(f"Error handling R-to-R data: {e}")
    
    def _on_events(self, msg):
        """Handler for event packets"""
        try:
            if 'Events' not in self.outlets:
                info = StreamInfo(
                    f"{self.stream_prefix}_Events",
                    'Markers',
                    1,
                    nominal_srate=0,  # Irregular events
                    channel_format='string',
                    source_id=f"{self.stream_prefix}_Events"
                )
                self.outlets['Events'] = StreamOutlet(info)
            
            # Push event marker
            event_str = f"{msg.event_string}/{msg.event_data}"
            self.outlets['Events'].push_sample([event_str], msg.stamp)
            
        except Exception as e:
            logger.error(f"Error handling events: {e}")
    
    def start_streaming(self):
        """Start BioHarness data streaming"""
        if self.running:
            logger.warning("BioHarness is already streaming")
            return
        
        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        logger.info(f"Started BioHarness streaming for {self.config.device_id}")
    
    def stop_streaming(self):
        """Stop BioHarness streaming"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        logger.info(f"Stopped BioHarness streaming for {self.config.device_id}")
    
    def _stream_loop(self):
        """Main streaming loop - keeps event loop running"""
        asyncio.set_event_loop(self.loop)
        
        try:
            # Run the event loop - data callbacks will handle streaming
            while self.running:
                self.loop.run_until_complete(asyncio.sleep(0.1))
                
        except Exception as e:
            if self.running:
                logger.error(f"Error in streaming loop: {e}")
    
    def cleanup(self):
        """Clean up BioHarness resources"""
        self.stop_streaming()
        
        if self.link:
            self.link.shutdown()
        
        if self.loop:
            self.loop.close()
        
        logger.info(f"BioHarness handler for {self.config.device_id} cleaned up")

