"""
Tests for the CaptureEngine class.
"""
import time
import unittest
from unittest.mock import patch

from vision_capture_core.core.capture_engine import CaptureEngine, CaptureState


class TestCaptureEngine(unittest.TestCase):
    """Test cases for the CaptureEngine class."""
    
    def setUp(self):
        """Set up a fresh CaptureEngine instance for each test."""
        self.engine = CaptureEngine()
    
    def test_initial_state(self):
        """Test that the engine initializes in the STOPPED state."""
        self.assertEqual(self.engine.state, CaptureState.STOPPED)
    
    def test_start_stop(self):
        """Test the start and stop methods."""
        self.engine.start()
        self.assertEqual(self.engine.state, CaptureState.RUNNING)
        
        self.engine.stop()
        self.assertEqual(self.engine.state, CaptureState.STOPPED)
    
    def test_pause(self):
        """Test the pause method."""
        self.engine.start()
        self.engine.pause()
        self.assertEqual(self.engine.state, CaptureState.PAUSED)
    
    def test_sleep_indefinite(self):
        """Test putting the engine to sleep indefinitely."""
        self.engine.start()
        self.engine.sleep()
        self.assertEqual(self.engine.state, CaptureState.SLEEPING)
        self.assertTrue(self.engine.is_sleeping())
    
    def test_sleep_with_duration(self):
        """Test putting the engine to sleep for a specific duration."""
        self.engine.start()
        self.engine.sleep(duration=0.1)
        self.assertEqual(self.engine.state, CaptureState.SLEEPING)
        
        time.sleep(0.15)
        
        self.assertEqual(self.engine.state, CaptureState.RUNNING)
        self.assertFalse(self.engine.is_sleeping())
    
    def test_wake(self):
        """Test waking the engine from sleep."""
        self.engine.start()
        self.engine.sleep()
        self.assertEqual(self.engine.state, CaptureState.SLEEPING)
        
        self.engine.wake()
        self.assertEqual(self.engine.state, CaptureState.RUNNING)
        self.assertFalse(self.engine.is_sleeping())
    
    @patch('time.time')
    def test_sleep_duration_calculation(self, mock_time):
        """Test that sleep duration is calculated correctly."""
        mock_time.side_effect = [100.0, 100.5]
        
        self.engine.start()
        self.engine.sleep(duration=1.0)
        
        self.assertEqual(self.engine.state, CaptureState.SLEEPING)
        
        mock_time.side_effect = [101.5]
        
        self.assertEqual(self.engine.state, CaptureState.RUNNING)


if __name__ == '__main__':
    unittest.main()
