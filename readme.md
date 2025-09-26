# Multi-Modal LSL Streaming Setup

echo "# Multimodal-LSL-Streaming-Setup" >> README.md
git init
git add README.md
git commit -m "first commit"
git branch -M main
git remote add origin git@github.com:ajay-naidu-96/Multimodal-LSL-Streaming-Setup.git
git push -u origin main

This system provides real-time streaming of multi-modal data (video, audio, physiological) using Lab Streaming Layer (LSL) for two-person experimental setups.

## Overview

Each participant machine streams:
- 2 RGB camera feeds
- 1 depth camera feed  
- 1 microphone audio stream
- 1 camera audio stream
- 1 BioHarness physiological data stream

All streams are synchronized and timestamped via LSL for network-wide data collection.

## Installation

1. **Install Python dependencies:**
```bash
pip install -r requirements.txt
```

2. **Install LSL (if not using pip):**
   - Download from [LSL releases](https://github.com/sccn/labstreaminglayer/releases)
   - Follow platform-specific installation instructions

3. **Optional hardware-specific drivers:**
   - **Intel RealSense:** Install [librealsense](https://github.com/IntelRealSense/librealsense) and uncomment `pyrealsense2` in requirements.txt
   - **BioHarness:** Ensure proper USB/Bluetooth drivers are installed

## Configuration

Edit `config.yaml` to match your setup:

```yaml
experiment_name: "YourExperiment"
participant_id: "P1"  # or "P2" for second machine
hostname: "midgard"   # or "huginn"

cameras:
  - id: 0              # Camera device ID
    name: "RGB1"
    stream_type: "Video"
    resolution: [640, 480]
    fps: 30
  - id: 1
    name: "RGB2" 
    stream_type: "Video"
    resolution: [640, 480]
    fps: 30
  - id: 2
    name: "DepthCam"
    stream_type: "Depth"
    resolution: [640, 480]
    fps: 30

microphone:
  source: "RGB1"       # Extract audio from this camera
  sample_rate: 44100
  channels: 1

bioharness:
  device_id: "BH12345"
  stream_name: "BioHarness"
  sample_rate: 250
```

## Usage

### Interactive Mode (Recommended)
```bash
# Start with real hardware
python main_streamer.py

# Start with simulated BioHarness data
python main_streamer.py --simulate
```

### Non-Interactive Mode (for scripts)
```bash
python main_streamer.py --non-interactive --simulate
```

### Custom Configuration
```bash
python main_streamer.py --config custom_config.yaml
```

## Stream Types and Data Formats

### Video Streams
- **Type:** `Video` or `Depth`
- **Format:** JPEG-compressed for RGB, raw float32 for depth
- **Naming:** `{CameraName}_{ParticipantID}` (e.g., `RGB1_P1`)

### Audio Streams  
- **Type:** `Audio`
- **Format:** float32 samples
- **Naming:** `Audio_{Source}_{ParticipantID}` (e.g., `Audio_RGB1_P1`)

### Physiological Streams
- **Type:** `Physiology` 
- **Channels:** heart_rate, rr_interval, breathing_rate, breathing_amplitude, posture, activity, peak_acceleration, ecg_signal, breathing_signal, temperature, battery_level
- **Format:** float32 multi-channel
- **Naming:** `BioHarness_{ParticipantID}`

## Integration with Existing Code

The new streaming system can work alongside your existing sensor collection:

```python
# Your existing code in sensors.py can collect these new LSL streams
stream_names = {
    "RGB1_P1": rgb_labels,
    "RGB2_P1": rgb_labels, 
    "DepthCam_P1": depth_labels,
    "Audio_RGB1_P1": audio_labels,
    "BioHarness_P1": bioharness_labels,
    # ... existing streams
    "pupil_capture": pupil_capture_labels,
    "GSR1": GSR_labels,
    "GSR2": GSR_labels,
}
```

## Network Setup

### Two-Machine Configuration
- **Machine A (midgard):** Streams for Participant 1
- **Machine B (huginn):** Streams for Participant 2  

All LSL streams are automatically discoverable across the network. Your existing data collection script can run on either machine or a third collection machine.

## Data Collection

Use your existing `sensors.py` script or LSL-compatible software:

1. **Stream Detection:**
```python
from lsl_stream_detect import print_available_streams
print_available_streams(verbose=True)
```

2. **Data Collection:**
```python
# Modify your existing sensors.py to include the new stream types
# The collect_and_filter_streams function will automatically discover them
```

## Hardware Requirements

### Minimum
- 2+ USB cameras per machine
- USB microphone or camera with audio
- Network connection between machines

### Recommended  
- Intel RealSense depth camera
- Zephyr BioHarness physiological monitor
- Gigabit Ethernet for high-bandwidth streaming
- Dedicated GPU for video processing (optional)

## Troubleshooting

### Camera Issues
```bash
# List available cameras
ls /dev/video*

# Test camera access
python -c "import cv2; cap = cv2.VideoCapture(0); print('Camera 0:', cap.isOpened())"
```

### Audio Issues
```bash
# List audio devices
python -c "import pyaudio; p = pyaudio.PyAudio(); [print(i, p.get_device_info_by_index(i)['name']) for i in range(p.get_device_count())]"
```

### LSL Network Issues
- Ensure firewall allows LSL traffic (typically UDP port 16571)
- Check network connectivity between machines
- Verify LSL installation with `python -c "import pylsl; print('LSL working')"`

### BioHarness Connection
- Check USB/Bluetooth connection
- Verify device ID in configuration
- Use `--simulate` flag for testing without hardware

## File Structure

```
├── main_streamer.py      