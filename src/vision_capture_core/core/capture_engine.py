"""
Core module for vision capture engine functionality.
"""
from enum import Enum
import time
from typing import Optional


class CaptureState(Enum):
    """Enum representing the possible states of the capture engine."""
    RUNNING = "running"
    PAUSED = "paused"
    SLEEPING = "sleeping"
    STOPPED = "stopped"


class CaptureEngine:
    """
    Main engine class for handling image capture and preprocessing.
    
    This class manages the capture state and provides methods to control
    the capture process, including the ability to put the engine to sleep.
    """
    
    def __init__(self):
        """Initialize the capture engine in a stopped state."""
        self._state = CaptureState.STOPPED
        self._sleep_start_time: Optional[float] = None
        self._sleep_duration: Optional[float] = None
    
    @property
    def state(self) -> CaptureState:
        """Get the current state of the capture engine."""
        if (self._state == CaptureState.SLEEPING and 
                self._sleep_start_time is not None and 
                self._sleep_duration is not None):
            elapsed = time.time() - self._sleep_start_time
            if elapsed >= self._sleep_duration:
                self._state = CaptureState.RUNNING
                self._sleep_start_time = None
                self._sleep_duration = None
        
        return self._state
    
    def start(self) -> None:
        """Start the capture engine."""
        if self._state == CaptureState.STOPPED or self._state == CaptureState.PAUSED:
            self._state = CaptureState.RUNNING
    
    def stop(self) -> None:
        """Stop the capture engine."""
        self._state = CaptureState.STOPPED
        self._sleep_start_time = None
        self._sleep_duration = None
    
    def pause(self) -> None:
        """Pause the capture engine."""
        if self._state == CaptureState.RUNNING:
            self._state = CaptureState.PAUSED
    
    def sleep(self, duration: Optional[float] = None) -> None:
        """
        Put the capture engine to sleep for the specified duration.
        
        Args:
            duration: Sleep duration in seconds. If None, the engine will
                     sleep indefinitely until explicitly woken up.
        """
        self._state = CaptureState.SLEEPING
        self._sleep_start_time = time.time()
        self._sleep_duration = duration
    
    def wake(self) -> None:
        """
        Wake up the capture engine from sleep state.
        
        If the engine is in sleep state, this will transition it to running state.
        If the engine is in any other state, this method has no effect.
        """
        if self._state == CaptureState.SLEEPING:
            self._state = CaptureState.RUNNING
            self._sleep_start_time = None
            self._sleep_duration = None
    
    def is_sleeping(self) -> bool:
        """Check if the capture engine is currently in sleep state."""
        return self.state == CaptureState.SLEEPING
