# Vision Capture Core

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

Real-time image capture & preprocessing engine for AI vision applications.

## Features

- Multiple camera support with configurable profiles
- Real-time image capture and processing
- Pipeline architecture for image processing
- Extensible processor system
- Command-line interface for easy configuration

## Installation

```bash
# Clone the repository
git clone https://github.com/yourusername/vision-capture-core.git
cd vision-capture-core

# Install dependencies
pip install -r requirements.txt
```

## Quick Start

Run the application with default settings:

```bash
python app.py
```

## Command-Line Interface

Vision Capture Core provides a flexible command-line interface:

```bash
# Use a specific camera device (by index)
# This will use camera 1 with profile "0"
python app.py --camera 1

# Use a specific camera profile from config
python app.py --profile "high_res"

# Use both a specific camera and profile
python app.py --camera 2 --profile "high_fps"

# List all available camera profiles
python app.py --list-profiles

# Use an alternative config file
python app.py --config "my_config.json"
```

### CLI Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--camera` | `-c` | Specify camera device index (0, 1, 2...). When used alone, will always use profile "0" |
| `--profile` | `-p` | Specify camera profile from config.json |
| `--list-profiles` | `-l` | Display all available camera profiles |
| `--config` | | Specify a config file path (default: config.json) |

## Configuration

The application uses a JSON configuration file to define camera profiles and processing pipelines.

### Example Configuration

```json
{
  "processors": ["grayscale", "blur"],
  "grayscale": {},
  "blur": {
    "kernel_size": [15, 15],
    "sigma": 0
  },
  "mode": "pipeline",
  "cameras": {
    "0": {
      "width": 3264,
      "height": 2448,
      "fps": 30,
      "backend": "DSHOW"
    },
    "high_res": {
      "width": 3840,
      "height": 2160,
      "fps": 30,
      "backend": "DSHOW"
    },
    "high_fps": {
      "width": 1280,
      "height": 720,
      "fps": 60,
      "backend": "DSHOW"
    }
  },
  "selected_camera_index": 0
}
```

### Camera Profiles

Camera profiles can be named with numbers or custom strings (e.g., "high_res", "high_fps"). 

**Important Note**: 
- When specifying only `--camera` without `--profile`, the application will always use profile "0" regardless of the camera index.
- If neither camera index nor profile is specified, the app will use the `selected_camera_index` value from the config file.
- Make sure your config.json has a profile named "0" when using `--camera` without `--profile`.

## Architecture

Vision Capture Core follows SOLID principles with a modular architecture:

- **Core**: Central pipeline execution
- **CaptureLib**: Camera and system management
- **Processors**: Image processing modules
- **Capture Runner**: UI and application control

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
