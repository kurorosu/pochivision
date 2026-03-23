# pochivision

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

[日本語](README.md)

Real-time image capture & preprocessing engine for AI vision applications, featuring a plugin-based processor architecture for easy extensibility.

## Requirements

- Python 3.12+

## Features

- Multiple camera support with configurable profiles
- Real-time image capture and processing
- Pipeline / parallel execution modes for image processing
- Extensible processor & feature extractor system via registry pattern
- Configurable preview window size
- Recording with selectable codec
- Command-line interface for easy configuration

## Directory Structure

```
pochivision/
├── pochivision/
│   ├── cli/                  # CLI entrypoint (vcc command)
│   ├── capture_runner/       # Camera capture and preview runner
│   ├── capturelib/           # Camera setup, config, logging, recording
│   ├── core/                 # Pipeline executor
│   ├── exceptions/           # Custom exception classes
│   ├── feature_extractors/   # Feature extraction plugins
│   ├── processors/           # Image processing pipeline
│   ├── tools/                # Utility scripts
│   └── utils/                # Shared utilities
├── tests/                    # Test suite
├── config.json               # Application configuration
└── pyproject.toml            # Project metadata & dependencies
```

## Installation

```bash
# Clone the repository
git clone https://github.com/kurorosu/pochivision.git
cd pochivision

# Install dependencies
uv sync
```

## For Developers

To install all dependencies including development, testing, and linting tools:

```bash
uv sync --group dev
```

## Quick Start

Run the application with default settings:

```bash
uv run vcc
```

## Command-Line Interface

pochivision provides a flexible command-line interface via the `vcc` command:

```bash
# Use a specific camera device (by index)
uv run vcc --camera 1

# Use a specific camera profile from config
uv run vcc --profile "high_res"

# Use both a specific camera and profile
uv run vcc --camera 2 --profile "high_fps"

# List all available camera profiles
uv run vcc --list-profiles

# Use an alternative config file
uv run vcc --config "my_config.json"

# Disable recording
uv run vcc --no-recording
```

### CLI Arguments

| Argument | Short | Description |
|----------|-------|-------------|
| `--camera` | `-c` | Specify camera device index (0, 1, 2...). When used alone, will always use profile "0" |
| `--profile` | `-p` | Specify camera profile from config.json |
| `--list-profiles` | `-l` | Display all available camera profiles |
| `--config` | | Specify a config file path (default: config.json) |
| `--no-recording` | | Disable recording functionality |

## Configuration

The application uses a JSON configuration file to define camera profiles, processing pipelines, recording, and preview settings.

### Configuration File Structure

```json
{
  "cameras": {
    "0": {
      "width": 3200,
      "height": 2400,
      "fps": 30,
      "backend": "DSHOW",
      "label": "Tokyo_Lab",
      "processors": ["resize", "gaussian_blur", "std_bin", "contour", "mask_composition"],
      "mode": "pipeline",
      "id_interval": 4,
      "gaussian_blur": { "kernel_size": [19, 19], "sigma": 0 },
      "std_bin": { "threshold": 20 },
      "resize": { "width": 1600, "height": 1200, "preserve_aspect_ratio": true, "aspect_ratio_mode": "width" },
      "contour": { "retrieval_mode": "list", "approximation_method": "simple", "min_area": 100 },
      "mask_composition": { "target_image": "original", "use_white_pixels": true, "enable_cropping": true }
    }
  },
  "recording": {
    "select_format": "mjpg"
  },
  "preview": {
    "width": 1280,
    "height": 720
  },
  "selected_camera_index": 0,
  "id_interval": 1
}
```

### Top-Level Settings

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `cameras` | Yes | - | Camera profile definitions |
| `selected_camera_index` | Yes | - | Camera index used when not specified on CLI |
| `id_interval` | No | 1 | Global capture ID interval |
| `recording.select_format` | No | `"mjpg"` | Recording codec (`mp4v`, `xvid`, `mjpg`, `ffv1`, etc.) |
| `preview.width` | No | 1280 | Preview window width |
| `preview.height` | No | 720 | Preview window height |

### Camera Profile Settings

| Key | Required | Default | Description |
|-----|----------|---------|-------------|
| `processors` | Yes | - | Array of processor names (must not be empty) |
| `width` | No | 640 | Camera resolution width |
| `height` | No | 480 | Camera resolution height |
| `fps` | No | 30 | Frame rate |
| `backend` | No | none | Camera backend (`DSHOW`, `MSMF`, `V4L2`, etc.) |
| `mode` | No | `"parallel"` | `"parallel"` or `"pipeline"` |
| `label` | No | none | Custom label for output file naming |
| `id_interval` | No | 1 | Per-profile capture ID interval |

### Camera Profile Notes

- Profile names that are numbers (e.g., "0") must match the `--camera` argument on the command line
- When only `--camera` is specified on the command line, profile "0" will be used
- Each camera profile can specify different processors and settings
- Specifying an unregistered processor will result in an error

## Architecture

pochivision follows SOLID principles with a modular architecture:

- **CLI**: Command-line entrypoint (`vcc` command)
- **Core**: Central pipeline execution (pipeline / parallel modes)
- **CaptureLib**: Camera setup, config handling, logging, recording
- **Processors**: Image processing modules (registry pattern)
- **Feature Extractors**: Feature extraction plugins (registry pattern)
- **Capture Runner**: Live preview and application control

## Available Processors

| # | Name | Description | Key Parameters |
|---|------|-------------|----------------|
| 1 | `grayscale` | Convert to grayscale | none |
| 2 | `gaussian_blur` | Gaussian blur | `kernel_size`, `sigma` |
| 3 | `average_blur` | Average blur | `kernel_size` |
| 4 | `median_blur` | Median blur | `kernel_size` (odd int) |
| 5 | `bilateral_filter` | Edge-preserving blur | `d`, `sigmaColor`, `sigmaSpace` |
| 6 | `motion_blur` | Linear motion blur | `kernel_size` (odd int), `angle` |
| 7 | `std_bin` | Standard binarization | `threshold` |
| 8 | `otsu_bin` | Otsu binarization | none |
| 9 | `gauss_adapt_bin` | Gaussian adaptive binarization | `block_size`, `c` |
| 10 | `mean_adapt_bin` | Mean adaptive binarization | `block_size`, `c` |
| 11 | `resize` | Resize image | `width`, `height`, `preserve_aspect_ratio`, `aspect_ratio_mode` |
| 12 | `canny_edge` | Canny edge detection | `threshold1`, `threshold2`, `aperture_size` |
| 13 | `contour` | Contour detection | `retrieval_mode`, `min_area`, `select_mode`, `contour_rank` |
| 14 | `clahe` | CLAHE contrast enhancement | `clip_limit`, `tile_grid_size`, `color_mode` |
| 15 | `equalize` | Histogram equalization | `color_mode` |
| 16 | `mask_composition` | Compose mask with source image | `target_image`, `use_white_pixels`, `enable_cropping` |

## Available Feature Extractors

Feature extractors analyze processed images and output numerical features for downstream AI tasks.

| # | Name | Description |
|---|------|-------------|
| 1 | `rgb` | RGB channel statistics (mean, std, etc.) |
| 2 | `hsv` | HSV channel statistics |
| 3 | `brightness` | Brightness statistics |
| 4 | `glcm` | GLCM texture features (contrast, energy, etc.) |
| 5 | `hlac` | HLAC texture features |
| 6 | `lbp` | LBP texture features |
| 7 | `fft` | FFT frequency features |
| 8 | `swt` | SWT frequency features |
| 9 | `circle_counter` | Circle detection and counting |

## License

This project is licensed under the MIT License - see the LICENSE file for details.
