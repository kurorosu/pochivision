# vision-capture-core
Real-time image capture &amp; preprocessing engine for AI vision.

## Overview

Vision-capture-core is a real-time image capture and preprocessing engine designed for AI vision applications. Its primary purpose is to acquire image data from various sources and prepare it for subsequent analysis by AI models.

## Features

- **State Management**: Control the capture engine with start, stop, pause, and sleep functionality
- **Sleep Mode**: Put the capture engine to sleep to conserve resources when not actively needed
- **Automatic Wake**: Configure the engine to wake up automatically after a specified duration

## Usage

```python
from vision_capture_core.core.capture_engine import CaptureEngine

# Create a new capture engine
engine = CaptureEngine()

# Start the engine
engine.start()

# Put the engine to sleep for 5 seconds
engine.sleep(duration=5)

# Check if the engine is sleeping
if engine.is_sleeping():
    print("Engine is sleeping")

# Wake the engine manually (if needed before the duration expires)
engine.wake()

# Stop the engine when done
engine.stop()
```

## Installation

```bash
pip install vision-capture-core
```

## License

This project is licensed under the MIT License - see the LICENSE file for details.
