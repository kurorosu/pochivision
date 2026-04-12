"""マスク合成プロセッサを提供するモジュール."""

from typing import Any

import cv2
import numpy as np

from pochivision.exceptions import ProcessorRuntimeError, ProcessorValidationError
from pochivision.processors import BaseProcessor
from pochivision.processors.registry import register_processor
from pochivision.processors.resize import ResizeProcessor
from pochivision.processors.validators.mask_composition import MaskCompositionValidator


@register_processor("mask_composition")
class MaskCompositionProcessor(BaseProcessor):
    """2値化画像をマスクとして使用し ターゲット画像と合成するプロセッサ.

    セマンティクス:
        入力された 2値マスク画像の白領域 (画素値 >= 128) を "有効" 領域とみなし,
        その部分に ``target_image`` のピクセルを出力する.
        "無効" 領域 (画素値 < 128) は黒 (0) で埋める.

        ``use_white_pixels=False`` の場合はマスクの白黒を反転した上で上記規則を
        適用するため, 黒領域に ``target_image`` が出力される.

        以前の実装では無効領域に元のマスク画像 (必要に応じてカラー化したもの) を
        重ねる不明瞭な挙動があったが, 直感に反するため本バージョンから削除した.

    マスク画像とターゲット画像のサイズが異なる場合, ターゲット画像を
    マスク画像のサイズにリサイズする.
    オプションで, マスクの白ピクセル領域に基づいてトリミングを行うことができる.

    このプロセッサはパイプラインモードでのみ使用可能. パラレルモードでは動作しない.

    登録名:
        "mask_composition"

    設定例:
        {
            "mask_composition": {
                "target_image": "original",
                # 合成する元画像の識別子 ("original" または他のプロセッサ名)
                "use_white_pixels": true,
                # true: 白領域に target_image を出力. false: 黒領域に target_image を出力
                "enable_cropping": false,
                # true: マスクの白ピクセル領域に基づいてトリミング
                "crop_margin": 5
                # トリミング時の余白ピクセル数
            }
        }
    """

    def __init__(self, name: str, config: dict[str, Any]) -> None:
        """プロセッサを初期化.

        Args:
            name (str): プロセッサ名.
            config (dict[str, Any]): 設定パラメータ.

        Raises:
            ProcessorRuntimeError: パラレルモードで実行しようとした場合.
        """
        super().__init__(name, config)
        self.validator = MaskCompositionValidator(self.config)

        # パラメータ解析
        default_config = self.get_default_config()
        self.target_image_name = self.config.get(
            "target_image", default_config["target_image"]
        )
        self.use_white_pixels = self.config.get(
            "use_white_pixels", default_config["use_white_pixels"]
        )
        self.enable_cropping = self.config.get(
            "enable_cropping", default_config["enable_cropping"]
        )
        self.crop_margin = self.config.get("crop_margin", default_config["crop_margin"])

        # ターゲット画像 (実行時に設定)
        self.target_image: np.ndarray | None = None

        # リサイズプロセッサの準備 (サイズは process 内で動的設定)
        resize_config = ResizeProcessor.get_default_config()
        # マスク合成では正確なサイズ合わせが必要なため アスペクト比保持を無効化
        resize_config["preserve_aspect_ratio"] = False
        self.resize_processor = ResizeProcessor(
            name="resize_for_mask_composition", config=resize_config
        )

    def set_target_image(self, image: np.ndarray) -> None:
        """合成するターゲット画像を設定.

        Args:
            image (np.ndarray): ターゲット画像.
        """
        self.target_image = image

    def set_pipeline_mode(self, mode: str) -> None:
        """パイプラインモードを設定.

        Args:
            mode (str): パイプラインモード ("pipeline" または "parallel").

        Raises:
            ProcessorRuntimeError: パラレルモードで実行しようとした場合.
        """
        if mode == "parallel":
            raise ProcessorRuntimeError(
                "MaskCompositionProcessor can only be used in pipeline mode"
            )

    def _find_crop_bounds(self, mask: np.ndarray) -> tuple[int, int, int, int] | None:
        """マスクの白ピクセル領域に基づいてトリミング範囲を計算.

        Args:
            mask (np.ndarray): 2値化マスク画像 (グレースケール).

        Returns:
            tuple[int, int, int, int] | None: トリミング範囲
                (y_min, y_max, x_min, x_max). 白ピクセルが存在しない場合は None.
        """
        white_pixels = np.where(mask == 255)
        if len(white_pixels[0]) == 0:
            return None

        raw_y_min = int(np.min(white_pixels[0]))
        raw_y_max = int(np.max(white_pixels[0]))
        raw_x_min = int(np.min(white_pixels[1]))
        raw_x_max = int(np.max(white_pixels[1]))

        height, width = mask.shape
        y_min = max(0, raw_y_min - self.crop_margin)
        y_max = min(height - 1, raw_y_max + self.crop_margin)
        x_min = max(0, raw_x_min - self.crop_margin)
        x_max = min(width - 1, raw_x_max + self.crop_margin)

        return (y_min, y_max, x_min, x_max)

    def _crop_image(
        self, image: np.ndarray, bounds: tuple[int, int, int, int]
    ) -> np.ndarray:
        """指定された範囲で画像をトリミング.

        Args:
            image (np.ndarray): トリミング対象の画像.
            bounds (tuple[int, int, int, int]): トリミング範囲
                (y_min, y_max, x_min, x_max).

        Returns:
            np.ndarray: トリミングされた画像.
        """
        y_min, y_max, x_min, x_max = bounds
        return image[y_min : y_max + 1, x_min : x_max + 1]

    def _to_grayscale_mask(self, mask_image: np.ndarray) -> np.ndarray:
        """マスク画像をグレースケール 2値マスクに変換.

        入力がカラー (3ch) の場合は BGR→GRAY 変換を行う.
        既にグレースケール (2D) の場合はそのまま返す.

        Args:
            mask_image (np.ndarray): マスク画像 (2D グレースケールまたは 3ch カラー).

        Returns:
            np.ndarray: 2D グレースケールマスク.
        """
        if mask_image.ndim == 3:
            return cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
        return mask_image

    def _align_target_to_mask(
        self, target_image: np.ndarray, mask_shape: tuple[int, ...]
    ) -> np.ndarray:
        """ターゲット画像をマスクの高さ/幅にリサイズ.

        Args:
            target_image (np.ndarray): ターゲット画像.
            mask_shape (tuple[int, ...]): マスク画像の shape.

        Returns:
            np.ndarray: マスクと同じ (H, W) を持つターゲット画像.
        """
        if (
            target_image.shape[0] == mask_shape[0]
            and target_image.shape[1] == mask_shape[1]
        ):
            return target_image.copy()

        self.resize_processor.width = mask_shape[1]
        self.resize_processor.height = mask_shape[0]
        return self.resize_processor.process(target_image)

    def _validate_target_mask_compatibility(
        self, target_image: np.ndarray, mask_gray: np.ndarray
    ) -> None:
        """target_image と mask_gray の shape / dtype 整合性を検証.

        Args:
            target_image (np.ndarray): リサイズ済みターゲット画像.
            mask_gray (np.ndarray): グレースケールマスク.

        Raises:
            ProcessorValidationError: shape または dtype が不整合な場合.
        """
        if target_image.shape[:2] != mask_gray.shape[:2]:
            raise ProcessorValidationError(
                "target_image and mask shape mismatch: "
                f"target={target_image.shape[:2]}, mask={mask_gray.shape[:2]}"
            )
        if target_image.dtype != np.uint8 or mask_gray.dtype != np.uint8:
            raise ProcessorValidationError(
                "target_image and mask must be dtype uint8: "
                f"target={target_image.dtype}, mask={mask_gray.dtype}"
            )

    def process(self, mask_image: np.ndarray) -> np.ndarray:
        """マスク画像を使用してターゲット画像と合成.

        白領域 (>= 128) に ``target_image`` のピクセルを出力し,
        黒領域 (< 128) を 0 で埋める.
        ``use_white_pixels=False`` の場合は白黒を反転して適用する.

        Args:
            mask_image (np.ndarray): 2値化マスク画像 (グレースケール or カラー).

        Returns:
            np.ndarray: 合成後の画像. ターゲット画像と同じチャンネル数.

        Raises:
            ProcessorRuntimeError: ターゲット画像未設定または処理失敗時.
            ProcessorValidationError: 入力画像や shape / dtype が不正な場合.
        """
        if self.target_image is None:
            raise ProcessorRuntimeError(
                f"Target image '{self.target_image_name}' is not set"
            )

        # 入力マスクのバリデーション (2値画像であること等)
        self.validator.validate_image(mask_image)

        try:
            # マスクをグレースケール 2値に統一
            mask_gray = self._to_grayscale_mask(mask_image)

            # ターゲットをマスクサイズに合わせる
            target_image = self._align_target_to_mask(
                self.target_image, mask_gray.shape
            )

            # shape / dtype の整合性を検証
            self._validate_target_mask_compatibility(target_image, mask_gray)

            # use_white_pixels=False の場合は白黒を反転して扱う
            effective_mask = (
                mask_gray if self.use_white_pixels else cv2.bitwise_not(mask_gray)
            )

            # 有効領域に target_image を出力, 無効領域は 0
            result = cv2.bitwise_and(target_image, target_image, mask=effective_mask)

            if self.enable_cropping:
                crop_bounds = self._find_crop_bounds(mask_gray)
                if crop_bounds is not None:
                    result = self._crop_image(result, crop_bounds)

            return result

        except ProcessorValidationError:
            raise
        except cv2.error as e:
            raise ProcessorRuntimeError(f"Error during mask composition: {e}")
        except Exception as e:
            raise ProcessorRuntimeError(f"Unexpected error in {self.name}: {e}")

    @staticmethod
    def get_default_config() -> dict[str, Any]:
        """マスク合成プロセッサのデフォルト設定を返す.

        Returns:
            dict[str, Any]: デフォルト設定.
        """
        return {
            "target_image": "original",  # デフォルトはオリジナル画像
            "use_white_pixels": True,  # 白領域に target_image を出力
            "enable_cropping": False,  # デフォルトではトリミング無効
            "crop_margin": 5,  # デフォルトのトリミング余白は 5 ピクセル
        }
