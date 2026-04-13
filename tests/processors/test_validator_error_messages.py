"""バリデータのエラーメッセージフォーマットを検証するテスト.

Issue #379 の Acceptance Criteria を確認する:
- プロセッサ名 (``[processor_name]``) がプレフィックスとして含まれる
- 受け取った値 (dtype, shape, 設定値等) が含まれる
- 期待値が含まれる
"""

from __future__ import annotations

import numpy as np
import pytest

from pochivision.exceptions import ProcessorValidationError
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


@pytest.mark.parametrize(
    "validator,expected_name",
    [
        (GaussianBlurValidator({}), "gaussian_blur"),
        (AverageBlurValidator({}), "average_blur"),
        (BilateralFilterValidator({}), "bilateral_filter"),
        (MedianBlurValidator({}), "median_blur"),
        (MotionBlurValidator({}), "motion_blur"),
        (StandardBinarizationValidator({}), "std_bin"),
        (OtsuBinarizationValidator({}), "otsu_bin"),
        (GaussianAdaptiveBinarizationValidator({}), "gauss_adapt_bin"),
        (MeanAdaptiveBinarizationValidator({}), "mean_adapt_bin"),
        (CLAHEInputValidator(), "clahe"),
        (EqualizeInputValidator(), "equalize"),
        (GrayscaleValidator({}), "grayscale"),
        (CannyEdgeValidator({}), "canny_edge"),
        (ContourValidator({}), "contour"),
        (MaskCompositionValidator({}), "mask_composition"),
        (ResizeConfigValidator({}), "resize"),
    ],
)
def test_processor_name_attribute_matches_registry(validator, expected_name):
    """各バリデータの ``processor_name`` がレジストリ登録名と一致する."""
    assert validator.processor_name == expected_name


@pytest.mark.parametrize(
    "validator,expected_name",
    [
        (GaussianBlurValidator({}), "gaussian_blur"),
        (AverageBlurValidator({}), "average_blur"),
        (BilateralFilterValidator({}), "bilateral_filter"),
        (MedianBlurValidator({}), "median_blur"),
        (MotionBlurValidator({}), "motion_blur"),
        (StandardBinarizationValidator({}), "std_bin"),
        (OtsuBinarizationValidator({}), "otsu_bin"),
        (GaussianAdaptiveBinarizationValidator({}), "gauss_adapt_bin"),
        (MeanAdaptiveBinarizationValidator({}), "mean_adapt_bin"),
        (CLAHEInputValidator(), "clahe"),
        (EqualizeInputValidator(), "equalize"),
        (GrayscaleValidator({}), "grayscale"),
        (CannyEdgeValidator({}), "canny_edge"),
        (ContourValidator({}), "contour"),
        (MaskCompositionValidator({}), "mask_composition"),
        (ResizeConfigValidator({}), "resize"),
    ],
)
def test_non_ndarray_input_message_contains_prefix_and_actual_type(
    validator, expected_name
):
    """非 ndarray 入力で ``[name]`` プレフィックスと実型名を含むメッセージが出る."""
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image("not_an_image")  # type: ignore[arg-type]

    msg = str(excinfo.value)
    assert f"[{expected_name}]" in msg
    assert "got str" in msg


@pytest.mark.parametrize(
    "validator,expected_name",
    [
        (GaussianBlurValidator({}), "gaussian_blur"),
        (AverageBlurValidator({}), "average_blur"),
        (StandardBinarizationValidator({}), "std_bin"),
        (CLAHEInputValidator(), "clahe"),
        (GrayscaleValidator({}), "grayscale"),
    ],
)
def test_empty_image_message_contains_prefix_and_shape(validator, expected_name):
    """空画像入力で ``[name]`` プレフィックスと shape を含むメッセージが出る."""
    empty = np.array([], dtype=np.uint8)
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(empty)

    msg = str(excinfo.value)
    assert f"[{expected_name}]" in msg
    assert "shape" in msg


def test_blur_dtype_error_includes_actual_dtype():
    """blur 系バリデータの dtype エラーが期待値と実 dtype を含む."""
    validator = GaussianBlurValidator({})
    image = np.zeros((4, 4, 3), dtype=np.float32)
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(image)

    msg = str(excinfo.value)
    assert "[gaussian_blur]" in msg
    assert "uint8" in msg
    assert "float32" in msg


def test_blur_channel_error_includes_actual_channel_count():
    """blur 系バリデータのチャンネル数エラーが実値を含む."""
    validator = AverageBlurValidator({})
    image = np.zeros((4, 4, 4), dtype=np.uint8)
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(image)

    msg = str(excinfo.value)
    assert "[average_blur]" in msg
    assert "got 4" in msg


def test_clahe_dtype_error_includes_processor_name_and_dtype():
    """CLAHE バリデータの dtype エラーがプレフィックスと実 dtype を含む."""
    validator = CLAHEInputValidator()
    image = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(image)

    msg = str(excinfo.value)
    assert "[clahe]" in msg
    assert "float32" in msg


def test_equalize_dtype_error_includes_processor_name_and_dtype():
    """equalize バリデータの dtype エラーがプレフィックスと実 dtype を含む."""
    validator = EqualizeInputValidator()
    image = np.zeros((4, 4), dtype=np.float64)
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(image)

    msg = str(excinfo.value)
    assert "[equalize]" in msg
    assert "float64" in msg


def test_otsu_invalid_ndim_error_includes_prefix_and_shape():
    """otsu バリデータの ndim エラーがプレフィックスと shape を含む."""
    validator = OtsuBinarizationValidator({})
    image = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(image)

    msg = str(excinfo.value)
    assert "[otsu_bin]" in msg
    assert "ndim=4" in msg


def test_canny_invalid_channel_error_includes_prefix_and_shape():
    """canny バリデータの ndim エラーがプレフィックスと shape を含む."""
    validator = CannyEdgeValidator({})
    image = np.zeros((4, 4, 4), dtype=np.uint8)  # 4 channels: invalid
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(image)

    msg = str(excinfo.value)
    assert "[canny_edge]" in msg
    assert "shape=" in msg


def test_adaptive_block_size_type_error_includes_prefix():
    """adaptive バリデータの block_size 型エラーがプレフィックスを含む."""
    with pytest.raises(ProcessorValidationError) as excinfo:
        GaussianAdaptiveBinarizationValidator({"block_size": "bad"})

    msg = str(excinfo.value)
    assert "[gauss_adapt_bin]" in msg
    assert "block_size" in msg


def test_adaptive_block_size_even_error_includes_prefix_and_value():
    """adaptive バリデータの block_size 偶数エラーが値を含む."""
    with pytest.raises(ProcessorValidationError) as excinfo:
        MeanAdaptiveBinarizationValidator({"block_size": 4})

    msg = str(excinfo.value)
    assert "[mean_adapt_bin]" in msg
    assert "got 4" in msg


def test_median_kernel_size_error_includes_prefix_and_value():
    """median バリデータの kernel_size エラーがプレフィックスと値を含む."""
    with pytest.raises(ProcessorValidationError) as excinfo:
        MedianBlurValidator({"kernel_size": 4})

    msg = str(excinfo.value)
    assert "[median_blur]" in msg
    assert "4" in msg


def test_gaussian_kernel_size_shape_error_includes_prefix_and_value():
    """gaussian_blur バリデータの kernel_size 形状エラーが値を含む."""
    with pytest.raises(ProcessorValidationError) as excinfo:
        GaussianBlurValidator({"kernel_size": [3, 3, 3]})

    msg = str(excinfo.value)
    assert "[gaussian_blur]" in msg
    assert "[3, 3, 3]" in msg


def test_mask_composition_non_binary_message_includes_prefix_and_unique_values():
    """mask_composition の非 binary エラーがプレフィックスと unique 値を含む."""
    validator = MaskCompositionValidator({})
    image = np.array([[0, 100, 255], [0, 100, 255]], dtype=np.uint8)
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(image)

    msg = str(excinfo.value)
    assert "[mask_composition]" in msg
    assert "100" in msg


def test_grayscale_dtype_error_includes_prefix_and_dtype():
    """grayscale バリデータの dtype エラーがプレフィックスと実 dtype を含む."""
    validator = GrayscaleValidator({})
    image = np.zeros((4, 4), dtype=np.float32)
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(image)

    msg = str(excinfo.value)
    assert "[grayscale]" in msg
    assert "float32" in msg


def test_contour_invalid_shape_message_includes_prefix_and_shape():
    """contour バリデータの ndim エラーがプレフィックスと shape を含む."""
    validator = ContourValidator({})
    image = np.zeros((2, 2, 2, 2), dtype=np.uint8)
    with pytest.raises(ProcessorValidationError) as excinfo:
        validator.validate_image(image)

    msg = str(excinfo.value)
    assert "[contour]" in msg
    assert "shape=" in msg
