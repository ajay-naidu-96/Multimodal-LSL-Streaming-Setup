experiment_name: "TwoPersonStudy"
participant_id: "P1"
hostname: "midgard"  # or "huginn" for second machine

cameras:
  - id: 0
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
  source: "RGB1"  # Extract audio from this camera
  sample_rate: 44100
  channels: 1

bioharness:
  device_id: "BH12345"
  stream_name: "BioHarness"
  sample_rate: 250  # Hz

# LSL Stream configuration
lsl_config:
  chunk_size: 32
  max_buffered: 360
  stream_timeout: 5.0