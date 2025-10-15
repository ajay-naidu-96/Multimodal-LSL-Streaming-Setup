import pylsl

print("Looking for LSL streams...\n")

# Resolve all active LSL streams
streams = pylsl.resolve_streams()

if not streams:
    print("⚠️ No LSL streams found.")
    exit(0)

# List available streams
print("Available LSL streams:")
for i, s in enumerate(streams):
    info = pylsl.StreamInfo(s.name(), s.type(), s.channel_count(), s.nominal_srate(), s.channel_format(), s.source_id())
    print(f"[{i}] Name: {info.name()}, Type: {info.type()}, Channels: {info.channel_count()}, ID: {info.source_id()}")

# Choose one
choice = int(input("\nEnter stream number to connect to: "))
inlet = pylsl.StreamInlet(streams[choice])

print(f"\n✅ Connected to '{streams[choice].name()}' stream. Receiving samples...\n(Press Ctrl+C to stop)\n")

try:
    while True:
        sample, timestamp = inlet.pull_sample(timeout=5.0)
        if sample:
            print(f"{timestamp:.3f}: {sample}")
except KeyboardInterrupt:
    print("\n👋 Viewer stopped.")
