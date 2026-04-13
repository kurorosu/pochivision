"""BaseValidator.validate_config 実装の振る舞いを検証するテスト."""

import pytest

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.validators.base import BaseValidator
from pochivision.processors.validators.binarization.adaptive import (
    GaussianAdaptiveBinarizationValidator,
    MeanAdaptiveBinarizationValidator,
)
from pochivision.processors.validators.binarization.otsu import (
    OtsuBinarizationValidator,
)
from pochivision.processors.validators.binarization.standard import (
    StandardBinarizationValidator,
)
from pochivision.processors.validators.blur.average import AverageBlurValidator
from pochivision.processors.validators.blur.bilateral import BilateralFilterValidator
from pochivision.processors.validators.blur.gaussian import GaussianBlurValidator
from pochivision.processors.validators.blur.median import MedianBlurValidator
from pochivision.processors.validators.blur.motion import MotionBlurValidator
from pochivision.processors.validators.clahe.clahe import CLAHEInputValidator
from pochivision.processors.validators.contour.contour import ContourValidator
from pochivision.processors.validators.edge_detection.canny import CannyEdgeValidator
from pochivision.processors.validators.equalize.equalize import EqualizeInputValidator
from pochivision.processors.validators.grayscale.grayscale import GrayscaleValidator
from pochivision.processors.validators.mask_composition.mask_composition import (
    MaskCompositionValidator,
)
from pochivision.processors.validators.resize.resize import ResizeConfigValidator

ALL_VALIDATORS = [
    AverageBlurValidator,
    BilateralFilterValidator,
    GaussianBlurValidator,
    MedianBlurValidator,
    MotionBlurValidator,
    StandardBinarizationValidator,
    OtsuBinarizationValidator,
    GaussianAdaptiveBinarizationValidator,
    MeanAdaptiveBinarizationValidator,
    GrayscaleValidator,
    ResizeConfigValidator,
    EqualizeInputValidator,
    CLAHEInputValidator,
    CannyEdgeValidator,
    ContourValidator,
    MaskCompositionValidator,
]


def test_base_validator_declares_validate_config_abstract() -> None:
    """BaseValidator に validate_config が抽象メソッドとして定義されていることを確認."""
    assert "validate_config" in BaseValidator.__abstractmethods__


@pytest.mark.parametrize("validator_cls", ALL_VALIDATORS)
def test_each_validator_implements_validate_config(validator_cls: type) -> None:
    """全バリデータが validate_config を実装していることを確認."""
    assert hasattr(validator_cls, "validate_config")
    assert "validate_config" not in getattr(validator_cls, "__abstractmethods__", set())


# --- blur ---


def test_average_blur_rejects_non_list_kernel_size() -> None:
    with pytest.raises(ProcessorValidationError):
        AverageBlurValidator({"kernel_size": 5})


def test_average_blur_rejects_zero_kernel() -> None:
    with pytest.raises(ProcessorValidationError):
        AverageBlurValidator({"kernel_size": [0, 3]})


def test_average_blur_accepts_valid() -> None:
    AverageBlurValidator({"kernel_size": [3, 5]})


def test_bilateral_filter_accepts_empty_config() -> None:
    BilateralFilterValidator({})


def test_gaussian_blur_rejects_even_kernel() -> None:
    with pytest.raises(ProcessorValidationError):
        GaussianBlurValidator({"kernel_size": [4, 5], "sigma": 1.0})


def test_gaussian_blur_accepts_valid() -> None:
    GaussianBlurValidator({"kernel_size": [3, 5], "sigma": 1.0})


def test_median_blur_rejects_even_kernel() -> None:
    with pytest.raises(ProcessorValidationError):
        MedianBlurValidator({"kernel_size": 4})


def test_median_blur_accepts_valid() -> None:
    MedianBlurValidator({"kernel_size": 5})


def test_motion_blur_rejects_zero_kernel() -> None:
    with pytest.raises(ProcessorValidationError):
        MotionBlurValidator({"kernel_size": 0})


def test_motion_blur_rejects_non_int_kernel() -> None:
    with pytest.raises(ProcessorValidationError):
        MotionBlurValidator({"kernel_size": "5"})


def test_motion_blur_accepts_valid() -> None:
    MotionBlurValidator({"kernel_size": 7})


# --- binarization ---


def test_standard_binarization_rejects_out_of_range_threshold() -> None:
    with pytest.raises(ProcessorValidationError):
        StandardBinarizationValidator({"threshold": 300})


def test_standard_binarization_rejects_negative_threshold() -> None:
    with pytest.raises(ProcessorValidationError):
        StandardBinarizationValidator({"threshold": -1})


def test_standard_binarization_accepts_valid() -> None:
    StandardBinarizationValidator({"threshold": 128})


def test_otsu_binarization_accepts_empty() -> None:
    OtsuBinarizationValidator({})


def test_gaussian_adaptive_rejects_even_block_size() -> None:
    with pytest.raises(ProcessorValidationError):
        GaussianAdaptiveBinarizationValidator({"block_size": 4})


def test_gaussian_adaptive_accepts_valid() -> None:
    GaussianAdaptiveBinarizationValidator({"block_size": 11})


def test_mean_adaptive_rejects_too_small_block_size() -> None:
    with pytest.raises(ProcessorValidationError):
        MeanAdaptiveBinarizationValidator({"block_size": 1})


def test_mean_adaptive_accepts_valid() -> None:
    MeanAdaptiveBinarizationValidator({"block_size": 11})


# --- grayscale ---


def test_grayscale_accepts_empty() -> None:
    GrayscaleValidator({})


# --- resize ---


def test_resize_rejects_zero_width() -> None:
    with pytest.raises(ProcessorValidationError):
        ResizeConfigValidator({"width": 0})


def test_resize_rejects_invalid_aspect_ratio_mode() -> None:
    with pytest.raises(ProcessorValidationError):
        ResizeConfigValidator({"aspect_ratio_mode": "diagonal"})


def test_resize_accepts_valid() -> None:
    ResizeConfigValidator({"width": 640, "height": 480, "aspect_ratio_mode": "width"})


# --- equalize ---


def test_equalize_rejects_invalid_color_mode() -> None:
    with pytest.raises(ProcessorValidationError):
        EqualizeInputValidator({"color_mode": "rgb"})


def test_equalize_accepts_valid_color_mode() -> None:
    EqualizeInputValidator({"color_mode": "lab"})


# --- clahe ---


def test_clahe_rejects_non_positive_clip_limit() -> None:
    with pytest.raises(ProcessorValidationError):
        CLAHEInputValidator({"clip_limit": 0})


def test_clahe_rejects_invalid_tile_grid_size() -> None:
    with pytest.raises(ProcessorValidationError):
        CLAHEInputValidator({"tile_grid_size": [8]})


def test_clahe_rejects_non_positive_tile_grid_size_element() -> None:
    with pytest.raises(ProcessorValidationError):
        CLAHEInputValidator({"tile_grid_size": [0, 8]})


def test_clahe_accepts_valid() -> None:
    CLAHEInputValidator(
        {"color_mode": "gray", "clip_limit": 2.0, "tile_grid_size": [8, 8]}
    )


# --- canny ---


def test_canny_rejects_even_aperture_size() -> None:
    with pytest.raises(ProcessorValidationError):
        CannyEdgeValidator({"threshold1": 10.0, "threshold2": 50.0, "aperture_size": 4})


def test_canny_rejects_aperture_out_of_range() -> None:
    with pytest.raises(ProcessorValidationError):
        CannyEdgeValidator({"threshold1": 10.0, "threshold2": 50.0, "aperture_size": 9})


def test_canny_rejects_threshold_order() -> None:
    with pytest.raises(ProcessorValidationError):
        CannyEdgeValidator({"threshold1": 100.0, "threshold2": 50.0})


def test_canny_rejects_negative_threshold() -> None:
    with pytest.raises(ProcessorValidationError):
        CannyEdgeValidator({"threshold1": -1.0, "threshold2": 50.0})


def test_canny_accepts_valid() -> None:
    CannyEdgeValidator({"threshold1": 10.0, "threshold2": 50.0, "aperture_size": 3})


# --- contour ---


def test_contour_rejects_negative_min_area() -> None:
    with pytest.raises(ProcessorValidationError):
        ContourValidator({"min_area": -1})


def test_contour_rejects_invalid_select_mode() -> None:
    with pytest.raises(ProcessorValidationError):
        ContourValidator({"select_mode": "random"})


def test_contour_accepts_valid() -> None:
    ContourValidator(
        {
            "retrieval_mode": "list",
            "approximation_method": "simple",
            "min_area": 100,
            "select_mode": "rank",
            "contour_rank": 0,
        }
    )


# --- mask_composition ---


def test_mask_composition_rejects_negative_crop_margin() -> None:
    with pytest.raises(ProcessorValidationError):
        MaskCompositionValidator({"crop_margin": -1})


def test_mask_composition_accepts_valid() -> None:
    MaskCompositionValidator({"crop_margin": 10})
