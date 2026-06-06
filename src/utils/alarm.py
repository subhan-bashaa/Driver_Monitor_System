"""
Alarm system for alerting the driver.
Plays an alarm sound in a non-blocking manner using threading.
"""

import time
import threading
import pygame
from src.utils.constants import ALARM_PATH, ALARM_COOLDOWN

# Initialize pygame mixer for audio playback
pygame.mixer.init()


class AlarmManager:
    """
    Manages alarm playback with cooldown to prevent spam.
    """
    
    def __init__(self):
        """Initialize the alarm manager with cooldown tracking."""
        self.last_alarm_time = 0
        self.is_playing = False
    
    def play_alarm(self):
        """
        Play the alarm sound in a separate thread (non-blocking).
        Implements cooldown to prevent continuous alarm spam.
        
        Returns:
            bool: True if alarm was played, False if still in cooldown
        """
        current_time = time.time()
        
        # Check if we're still in cooldown period
        if current_time - self.last_alarm_time < ALARM_COOLDOWN:
            return False
        
        # Check if alarm is already playing
        if self.is_playing:
            return False
        
        # Start alarm in separate thread
        alarm_thread = threading.Thread(target=self._play_sound, daemon=True)
        alarm_thread.start()
        
        self.last_alarm_time = current_time
        return True
    
    def _play_sound(self):
        """
        Internal method to play the sound file.
        Runs in a separate thread to avoid blocking.
        """
        try:
            self.is_playing = True
            sound = pygame.mixer.Sound(ALARM_PATH)
            sound.play()
            # Wait for sound to finish
            while pygame.mixer.get_busy():
                pygame.time.Clock().tick(10)
        except Exception as e:
            print(f"Error playing alarm: {e}")
        finally:
            self.is_playing = False


# Global alarm manager instance
_alarm_manager = AlarmManager()


def play_alarm():
    """
    Convenience function to play the alarm.
    
    Returns:
        bool: True if alarm was played, False if still in cooldown
    """
    return _alarm_manager.play_alarm()
