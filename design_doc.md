# Design Document: LSL Streaming Setup for Multi-Modal Data Capture

**Author:** CLaSP / Eager Project
**Date:** 2025-09-19

---

## Overview
This document describes the design of an LSL (Lab Streaming Layer) system for recording multi-modal data per participant in a two-person experimental setup. Each machine is responsible for streaming one participant’s data in real-time.  

The streams include:  
- Two regular video streams (RGB cameras)  
- One depth camera stream  
- One BioHarness physiological data stream  
- One audio stream (microphone from a camera)  

---

## Objectives
- Provide synchronized, timestamped streams via LSL for each modality.  
- Allow flexible configuration through a central config file (e.g., YAML/JSON).  
- Support independent operation per participant machine to reduce load.  
- Ensure extensibility for future sensors or modalities.  

---

## System Architecture

### High-Level Design
Each machine runs the following components:

1. **Camera Handlers**: Manage two RGB cameras and one depth camera using device SDKs (e.g., Intel RealSense).  
2. **Audio Handler**: Extracts audio from one of the RGB cameras.  
3. **BioHarness Handler**: Interfaces with the wearable physiological sensor.  
4. **LSL Streamer**: Creates individual LSL outlets per modality and pushes timestamped samples/frames.  
5. **Configuration Manager**: Loads per-participant settings (device IDs, stream names, experiment name).  

### Per-Person Stream Layout
- Video 1 (RGB)  
- Video 2 (RGB)  
- Depth Camera  
- Microphone (from one camera)  
- BioHarness Data  

### Multi-Person Setup
For two participants:
- **Machine A** handles Participant 1 (2 RGB, 1 depth, 1 BioHarness, 1 mic).  
- **Machine B** handles Participant 2 (2 RGB, 1 depth, 1 BioHarness, 1 mic).  

All streams are discoverable via LSL across the network.  

---

## Configuration
Configuration will be stored in a structured file (`config.yaml`), allowing experimenters to specify participant ID, experiment name, and device mappings.  

### Example
```yaml
experiment_name: "TwoPersonStudy"
participant_id: "P1"
cameras:
  - id: 0
    name: "RGB1"
    stream_type: "Video"
  - id: 1
    name: "RGB2"
    stream_type: "Video"
  - id: 2
    name: "DepthCam"
    stream_type: "Depth"
microphone:
  source: "RGB1"
bioharness:
  device_id: "BH12345"
