import pylsl
import time

info = pylsl.StreamInfo('Speller_Decoded', 'Markers', 1, 0, 'string', 'mock_backend')
outlet = pylsl.StreamOutlet(info)

print("Mock backend running...")
while True:
    time.sleep(5)
    print("Pushing marker DECODED_A")
    outlet.push_sample(['DECODED_A'])
