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

The application uses a JSON configuration file to define camera profiles and processing pipelines. Each camera profile has independent settings for image processing algorithms and parameters.

### Configuration File Structure

```json
{
  "cameras": {
    "0": {
      "width": 3200,
      "height": 2400,
      "fps": 30,
      "backend": "DSHOW",
      "processors": ["blur", "grayscale"],
      "mode": "parallel",
      "blur": {
        "kernel_size": [15, 15],
        "sigma": 0
      },
      "grayscale": {}
    },
    "high_res": {
      "width": 3840,
      "height": 2160,
      "fps": 30,
      "backend": "DSHOW",
      "processors": ["blur"],
      "mode": "parallel",
      "blur": {
        "kernel_size": [31, 31],
        "sigma": 0
      }
    },
    "high_fps": {
      "width": 1280,
      "height": 720,
      "fps": 60,
      "backend": "DSHOW",
      "processors": ["grayscale"],
      "mode": "pipeline",
      "grayscale": {}
    }
  },
  "selected_camera_index": 0
}
```

### Required and Optional Parameters

**Global settings (required)**:
- `cameras`: Object defining camera profiles (required)
- `selected_camera_index`: Camera index used when not specified on command line (required)

**For each camera profile (in `cameras`)**:

Required parameters:
- `processors`: Array of processor names to use (required, must not be empty)

Optional parameters with default values:
- `width`: Camera resolution width (optional, default: 640)
- `height`: Camera resolution height (optional, default: 480)
- `fps`: Frame rate (optional, default: 30)
- `backend`: Camera backend type (optional, default: none)
- `mode`: Processing mode (optional, default: "parallel")
  - `"parallel"`: Each processor processes the original image independently
  - `"pipeline"`: Each processor receives the output of the previous processor
- Processor settings: Objects with processor name as key (optional, default: empty object)

**Example processor settings**:
- `blur`: Blur processing parameters
  - `kernel_size`: Kernel size (e.g., [15, 15])
  - `sigma`: Sigma value (e.g., 0)
- `grayscale`: Grayscale conversion (no parameters)

### Camera Profile Notes

- Profile names that are numbers (e.g., "0") must match the `--camera` argument on the command line
- When only `--camera` is specified on the command line, profile "0" will be used
- Each camera profile can specify different processors and settings
- The `processors` list in each profile is required and must not be empty
- Specifying an unregistered processor will result in an error

## Architecture

Vision Capture Core follows SOLID principles with a modular architecture:

- **Core**: Central pipeline execution
- **CaptureLib**: Camera and system management
- **Processors**: Image processing modules
- **Capture Runner**: UI and application control

## Available Processors

The following processors are currently available:

1. **grayscale**: Converts color images to grayscale
   - Parameters: none
   ```json
   "grayscale": {}
   ```

2. **blur**: Applies Gaussian blur
   - Parameters:
     - `kernel_size`: Blur kernel size (e.g., [15, 15])
     - `sigma`: Gaussian blur sigma value (e.g., 0)
   ```json
   "blur": {
     "kernel_size": [15, 15],
     "sigma": 0
   }
   ```

## Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

## License

This project is licensed under the MIT License - see the LICENSE file for details.
