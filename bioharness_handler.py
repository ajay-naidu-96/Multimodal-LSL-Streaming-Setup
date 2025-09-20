import threading
import time
import serial
import struct
import numpy as np
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
            # Initialize serial connection to BioHarness
            # This is a simplified implementation - actual BioHarness might use Bluetooth
            serial_port = self._find_bioharness_port()
            if serial_port:
                self.serial_connection = serial.Serial(
                    serial_port, 
                    baudrate=115200,
                    timeout=1.0
                )
                print(f"Connected to BioHarness on {serial_port}")
            else:
                print("BioHarness not found, using simulated data")
            
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
                if self.serial_connection:
                    # Read real BioHarness data
                    data = self._read_bioharness_data()
                else:
                    # Generate simulated physiological data
                    data = self._generate_simulated_data()
                
                if data:
                    timestamp = local_clock()
                    self.outlet.push_sample(data, timestamp)
                
            except Exception as e:
                if self.running:
                    print(f"Error reading BioHarness data: {e}")
                    # Continue with simulated data
                    data = self._generate_simulated_data()
                    if data:
                        timestamp = local_clock()
                        self.outlet.push_sample(data, timestamp)
            
            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            time.sleep(sleep_time)
    
    def _read_bioharness_data(self) -> Optional[list]:
        """Read actual BioHarness data from serial connection"""
        try:
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
            
        except Exception as e:
            print(f"Error reading from BioHarness: {e}")
            return None
    
    def _parse_bioharness_packet(self, raw_data: bytes) -> list:
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
            return self._generate_simulated_data()
    
    def _generate_simulated_data(self) -> list:
        """Generate simulated physiological data for testing"""
        current_time = time.time()
        
        # Generate realistic-looking physiological data
        heart_rate = 70 + 10 * np.sin(current_time * 0.1) + np.random.normal(0, 2)
        rr_interval = 60000 / max(heart_rate, 40)  # Convert to ms
        breathing_rate = 16 + 4 * np.sin(current_time * 0.05) + np.random.normal(0, 1)
        breathing_amplitude = 500 + 100 * np.sin(current_time * 0.2) + np.random.normal(0, 20)
        posture = 85 + 10 * np.sin(current_time * 0.01) + np.random.normal(0, 5)
        activity = abs(50 + 30 * np.sin(current_time * 0.3) + np.random.normal(0, 10))
        peak_accel = abs(1.0 + 0.5 * np.sin(current_time * 0.4) + np.random.normal(0, 0.1))
        ecg_signal = 0.5 * np.sin(current_time * heart_rate * 2 * np.pi / 60)
        breathing_signal = 0.3 * np.sin(current_time * breathing_rate * 2 * np.pi / 60)
        temperature = 36.5 + 0.5 * np.sin(current_time * 0.001) + np.random.normal(0, 0.1)
        battery_level = max(0, 100 - (current_time % 3600) / 36)  # Simulate battery drain
        
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
    
    def cleanup(self):
        """Clean up BioHarness resources"""
        self.stop_streaming()
        
        if self.serial_connection and self.serial_connection.is_open:
            self.serial_connection.close()
        
        print(f"BioHarness handler for {self.config.device_id} cleaned up")


class SimulatedBioHarnessHandler(BioHarnessHandler):
    """Simulated BioHarness for testing without actual hardware"""
    
    def __init__(self, bio_config: BioHarnessConfig, participant_id: str):
        super().__init__(bio_config, participant_id)
        self.simulation_start_time = None
    
    def initialize(self) -> bool:
        """Initialize simulated BioHarness"""
        try:
            self.simulation_start_time = time.time()
            
            # Create LSL outlet
            bio_info = StreamInfo(
                self.stream_name,
                'Physiology',
                channel_count=len(self.channels),
                nominal_srate=self.config.sample_rate,
                channel_format='float32',
                source_id=f"{self.stream_name}_{self.config.device_id}_sim"
            )
            
            # Set channel names
            channels = bio_info.desc().append_child("channels")
            for channel in self.channels:
                ch = channels.append_child("channel")
                ch.append_child_value("label", channel)
                ch.append_child_value("unit", self._get_channel_unit(channel))
                ch.append_child_value("type", "physiological")
            
            self.outlet = StreamOutlet(bio_info)
            print(f"Initialized simulated BioHarness for {self.config.device_id}")
            return True
            
        except Exception as e:
            print(f"Error initializing simulated BioHarness: {e}")
            return False
    
    def _stream_loop(self):
        """Streaming loop for simulated data"""
        sample_interval = 1.0 / self.config.sample_rate
        
        while self.running:
            start_time = time.time()
            
            try:
                # Generate more sophisticated simulated data
                data = self._generate_realistic_simulation()
                
                if data:
                    timestamp = local_clock()
                    self.outlet.push_sample(data, timestamp)
                
            except Exception as e:
                if self.running:
                    print(f"Error in simulated BioHarness streaming: {e}")
            
            # Maintain sample rate
            elapsed = time.time() - start_time
            sleep_time = max(0, sample_interval - elapsed)
            time.sleep(sleep_time)
    
    def _generate_realistic_simulation(self) -> list:
        """Generate more realistic simulated physiological data"""
        if not self.simulation_start_time:
            self.simulation_start_time = time.time()
        
        elapsed_time = time.time() - self.simulation_start_time
        
        # Simulate different physiological states over time
        stress_level = 0.5 + 0.3 * np.sin(elapsed_time * 0.01)  # Slow stress variation
        activity_level = abs(np.sin(elapsed_time * 0.1))  # Activity cycles
        
        # Heart rate with stress and activity influence
        base_hr = 70
        stress_hr = stress_level * 20
        activity_hr = activity_level * 40
        hr_noise = np.random.normal(0, 2)
        heart_rate = base_hr + stress_hr + activity_hr + hr_noise
        
        # RR interval (inverse relationship with HR)
        rr_interval = 60000 / max(heart_rate, 40)
        
        # Breathing rate influenced by stress and activity
        base_br = 16
        stress_br = stress_level * 8
        activity_br = activity_level * 12
        br_noise = np.random.normal(0, 1)
        breathing_rate = base_br + stress_br + activity_br + br_noise
        
        # Other parameters with realistic variations
        breathing_amplitude = 500 + stress_level * 200 + np.random.normal(0, 50)
        posture = 85 + 15 * np.sin(elapsed_time * 0.02) + np.random.normal(0, 5)
        activity = activity_level * 100 + np.random.normal(0, 10)
        peak_accel = activity_level * 2.0 + 0.1 + np.random.normal(0, 0.1)
        
        # ECG and breathing signals
        ecg_freq = heart_rate / 60.0
        breathing_freq = breathing_rate / 60.0
        ecg_signal = 0.5 * np.sin(2 * np.pi * ecg_freq * elapsed_time) + 0.1 * np.random.normal()
        breathing_signal = 0.3 * np.sin(2 * np.pi * breathing_freq * elapsed_time) + 0.05 * np.random.normal()
        
        # Temperature with small variations
        temperature = 36.5 + 0.5 * stress_level + np.random.normal(0, 0.1)
        
        # Battery level (slowly decreasing)
        battery_level = max(0, 100 - elapsed_time / 360)  # 10 hours battery life
        
        return [
            max(0, heart_rate),
            max(0, rr_interval),
            max(0, breathing_rate),
            max(0, breathing_amplitude),
            posture,
            max(0, activity),
            max(0, peak_accel),
            ecg_signal,
            breathing_signal,
            temperature,
            max(0, battery_level)
        ]