"""
Script to generate a simple alarm sound.
Run this if you don't have an alarm.wav file.
"""

import numpy as np
from scipy.io import wavfile
import os

def generate_alarm_sound():
    """Generate a simple beep alarm sound."""
    
    # Audio parameters
    sample_rate = 44100  # Hz
    duration = 1.0  # seconds
    frequency = 880  # Hz (A5 note)
    
    # Generate time array
    t = np.linspace(0, duration, int(sample_rate * duration))
    
    # Generate beep with envelope to avoid clicks
    beep = np.sin(2 * np.pi * frequency * t)
    
    # Apply fade in/out envelope
    fade_samples = int(0.05 * sample_rate)  # 50ms fade
    envelope = np.ones_like(beep)
    envelope[:fade_samples] = np.linspace(0, 1, fade_samples)
    envelope[-fade_samples:] = np.linspace(1, 0, fade_samples)
    
    beep = beep * envelope
    
    # Convert to 16-bit PCM
    audio_data = (beep * 32767).astype(np.int16)
    
    # Create assets directory if it doesn't exist
    os.makedirs('assets', exist_ok=True)
    
    # Save to file
    output_path = 'assets/alarm.wav'
    wavfile.write(output_path, sample_rate, audio_data)
    
    print(f"✓ Alarm sound generated successfully: {output_path}")
    print(f"  Sample rate: {sample_rate} Hz")
    print(f"  Duration: {duration} seconds")
    print(f"  Frequency: {frequency} Hz")

if __name__ == "__main__":
    try:
        generate_alarm_sound()
    except Exception as e:
        print(f"✗ Error generating alarm sound: {e}")
        print("  Make sure scipy is installed: pip install scipy")
