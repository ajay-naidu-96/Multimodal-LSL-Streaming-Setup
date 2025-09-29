import os
os.environ['LSL_MULTICAST_TTL'] = '0'  # Localhost only

from pylsl import StreamInfo, StreamOutlet
import time

print("Creating LSL outlet...")
info = StreamInfo('WinTest', 'Test', 1, 100, 'float32', 'test123')
outlet = StreamOutlet(info, max_buffered=10)
print("SUCCESS! Outlet created")

for i in range(5):
    outlet.push_sample([float(i)])
    print(f"Sent sample {i}")
    time.sleep(0.5)