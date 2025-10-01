import threading
import time
import serial
import struct
from typing import Optional, Dict, Any
from pylsl import StreamOutlet, StreamInfo, local_clock
from config_manager import BioHarnessConfig

class BioHarnessHandler:
    """Handler for BioHarness physiological data streaming"""
    
    def __init__(self, bio_config: BioHarnessConfig, participant_id: str):
        self.config = bio_config
        self.participant_id = participant_id
        self.serial_connection = None
        self.outlet = None
        self.running = False
        self.thread = None
        
        # BioHarness data channels
        self.channels = [
            'heart_rate',
            'rr_interval', 
            'breathing_rate',
            'breathing_amplitude',
            'posture',
            'activity',
            'peak_acceleration',
            'ecg_signal',
            'breathing_signal',
            'temperature',
            'battery_level'
        ]
        
        # Stream name
        self.stream_name = f"{self.config.stream_name}_{participant_id}"
    
    def initialize(self) -> bool:
        """Initialize BioHarness connection and LSL outlet"""
        try:
            # Find the BioHarness serial port
            serial_port = self._find_bioharness_port()
            if not serial_port:
                print("Error: BioHarness device not found. Cannot initialize stream.")
                return False

            # Initialize serial connection to BioHarness
            self.serial_connection = serial.Serial(
                serial_port, 
                baudrate=115200,
                timeout=1.0
            )
            print(f"Connected to BioHarness on {serial_port}")
            
            # Create LSL outlet
            bio_info = StreamInfo(
                self.stream_name,
                'Physiology',
                channel_count=len(self.channels),
                nominal_srate=self.config.sample_rate,
                channel_format='float32',
                source_id=f"{self.stream_name}_{self.config.device_id}"
            )
            
            # Set channel names
            channels = bio_info.desc().append_child("channels")
            for channel in self.channels:
                ch = channels.append_child("channel")
                ch.append_child_value("label", channel)
                ch.append_child_value("unit", self._get_channel_unit(channel))
                ch.append_child_value("type", "physiological")
            
            self.outlet = StreamOutlet(bio_info)
            print(f"Initialized BioHarness handler for {self.config.device_id}")
            return True
            
        except Exception as e:
            print(f"Error initializing BioHarness handler: {e}")
            return False
    
    def _find_bioharness_port(self) -> Optional[str]:
        """Find BioHarness serial port (simplified implementation)"""
        import serial.tools.list_ports
        
        # Look for BioHarness device
        for port in serial.tools.list_ports.comports():
            if 'BioHarness' in port.description or 'Zephyr' in port.description:
                return port.device
        
        return None
    
    def _get_channel_unit(self, channel: str) -> str:
        """Get unit for each channel"""
        units = {
            'heart_rate': 'BPM',
            'rr_interval': 'ms',
            'breathing_rate': 'BPM',
            'breathing_amplitude': 'mV',
            'posture': 'degrees',
            'activity': 'VMU',
            'peak_acceleration': 'g',
            'ecg_signal': 'mV',
            'breathing_signal': 'mV',
            'temperature': 'C',
            'battery_level': '%'
        }
        return units.get(channel, 'unknown')
    
    def start_streaming(self):
        """Start BioHarness data streaming"""
        if self.running:
            print("BioHarness is already streaming")
            return
        
        if not self.serial_connection or not self.outlet:
            print("Cannot start streaming. BioHarness handler is not properly initialized.")
            return

        self.running = True
        self.thread = threading.Thread(target=self._stream_loop, daemon=True)
        self.thread.start()
        print(f"Started BioHarness streaming for {self.config.device_id}")
    
    def stop_streaming(self):
        """Stop BioHarness streaming"""
        self.running = False
        if self.thread and self.thread.is_alive():
            self.thread.join(timeout=5.0)
        print(f"Stopped BioHarness streaming for {self.config.device_id}")
    
    def _stream_loop(self):
        """Main BioHarness streaming loop"""
        sample_interval = 1.0 / self.config.sample_rate
        
        while self.running:
            start_time = time.time()
            
            try:
                # Read real BioHarness data
                data = self._read_bioharness_data()
                
                if data:
                    timestamp = local_clock()
                    self.outlet.push_sample(data, timestamp)
                
            except serial.SerialException as e:
                print(f"Serial connection error: {e}. Stopping stream.")
                self.running = False # Stop the loop on major connection error
            except Exception as e:
                if self.running:
                    print(f"Error reading BioHarness data: {e}")

            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            time.sleep(sleep_time)
    
    def _read_bioharness_data(self) -> Optional[list]:
        """Read actual BioHarness data from serial connection"""
        if not self.serial_connection or not self.serial_connection.is_open:
            return None
        
        # Read BioHarness packet (simplified implementation)
        # Actual BioHarness protocol would be more complex
        if self.serial_connection.in_waiting > 0:
            raw_data = self.serial_connection.read(22)  # BioHarness packet size
            if len(raw_data) >= 22:
                # Parse BioHarness data format (simplified)
                parsed_data = self._parse_bioharness_packet(raw_data)
                return parsed_data
        
        return None
            
    def _parse_bioharness_packet(self, raw_data: bytes) -> Optional[list]:
        """Parse BioHarness data packet"""
        try:
            # Simplified parsing - actual format would depend on BioHarness model
            # This is a placeholder implementation
            
            # Extract values from packet (example format)
            heart_rate = struct.unpack('B', raw_data[2:3])[0]
            rr_interval = struct.unpack('>H', raw_data[3:5])[0]
            breathing_rate = raw_data[5]
            breathing_amplitude = struct.unpack('>H', raw_data[6:8])[0] / 100.0
            posture = struct.unpack('b', raw_data[8:9])[0]
            activity = struct.unpack('>H', raw_data[9:11])[0] / 100.0
            peak_accel = struct.unpack('>H', raw_data[11:13])[0] / 100.0
            ecg_signal = struct.unpack('>h', raw_data[13:15])[0] / 1000.0
            breathing_signal = struct.unpack('>h', raw_data[15:17])[0] / 1000.0
            temperature = struct.unpack('>H', raw_data[17:19])[0] / 10.0
            battery_level = raw_data[19]
            
            return [
                float(heart_rate),
                float(rr_interval),
                float(breathing_rate),
                float(breathing_amplitude),
                float(posture),
                float(activity),
                float(peak_accel),
                float(ecg_signal),
                float(breathing_signal),
                float(temperature),
                float(battery_level)
            ]
            
        except Exception as e:
            print(f"Error parsing BioHarness packet: {e}")
            return None
    
    def cleanup(self):
        """Clean up BioHarness resources"""
        self.stop_streaming()
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        
        print(f"BioHarness handler for {self.config.device_id} cleaned up")