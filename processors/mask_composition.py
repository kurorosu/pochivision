"""マスク合成プロセッサを提供するモジュール."""

from typing import Any, Dict, Optional

import cv2
import numpy as np

from exceptions import ProcessorRuntimeError
from processors import BaseProcessor
from processors.registry import register_processor
from processors.resize import ResizeProcessor
from processors.validators.mask_composition import MaskCompositionValidator


@register_processor("mask_composition")
class MaskCompositionProcessor(BaseProcessor):
    """
    2値化画像をマスクとして使用し、元画像と合成するプロセッサー.

    2値化画像の白ピクセル部分または黒ピクセル部分を、指定した元画像のピクセルで置き換えます。
    マスク画像とターゲット画像のサイズが異なる場合は、ターゲット画像をリサイズします。

    このプロセッサはパイプラインモードでのみ使用可能です。パラレルモードでは動作しません。

    登録名:
        "mask_composition"

    設定例:
        {
            "mask_composition": {
                "target_image": "original",
                # 合成する元画像の識別子（"original"または他のプロセッサ名）
                "use_white_pixels": true
                # trueの場合、白ピクセル部分を元画像で置き換え。falseの場合は黒ピクセル部分を置き換え
            }
        }
    """

    def __init__(self, name: str, config: Dict[str, Any]) -> None:
        """
        MaskCompositionProcessorを初期化.

        Args:
            name (str): プロセッサ名.
            config (Dict[str, Any]): 設定パラメータ.

        Raises:
            ProcessorRuntimeError: パラレルモードで実行しようとした場合.
        """
        super().__init__(name, config)
        self.validator = MaskCompositionValidator(self.config)
        self.validator.validate_config()

        # 設定パラメータ
        default_config = self.get_default_config()
        self.target_image_name = self.config.get(
            "target_image", default_config["target_image"]
        )
        self.use_white_pixels = self.config.get(
            "use_white_pixels", default_config["use_white_pixels"]
        )

        # ターゲット画像（実行時に設定）
        self.target_image: Optional[np.ndarray] = None

        # リサイズプロセッサの準備（サイズはprocess内で動的設定）
        resize_config = ResizeProcessor.get_default_config()
        # マスク合成では正確なサイズ合わせが必要なため、アスペクト比保持を無効化
        resize_config["preserve_aspect_ratio"] = False
        self.resize_processor = ResizeProcessor(
            name="resize_for_mask_composition", config=resize_config
        )

    def set_target_image(self, image: np.ndarray) -> None:
        """
        合成するターゲット画像を設定.

        Args:
            image (np.ndarray): ターゲット画像.
        """
        self.target_image = image

    def set_pipeline_mode(self, mode: str) -> None:
        """
        パイプラインモードを設定.

        Args:
            mode (str): パイプラインモード ("pipeline" または "parallel").

        Raises:
            ProcessorRuntimeError: パラレルモードで実行しようとした場合.
        """
        if mode == "parallel":
            raise ProcessorRuntimeError(
                "MaskCompositionProcessor can only be used in pipeline mode"
            )

    def process(self, mask_image: np.ndarray) -> np.ndarray:
        """
        マスク画像を使用して元画像と合成.

        Args:
            mask_image (np.ndarray): マスク画像（2値化画像）.

        Returns:
            np.ndarray: 合成後の画像.

        Raises:
            ProcessorRuntimeError: ターゲット画像が設定されていない場合や処理中にエラーが発生した場合.
        """
        if self.target_image is None:
            raise ProcessorRuntimeError(
                f"Target image '{self.target_image_name}' is not set"
            )

        # 入力画像のバリデーション
        self.validator.validate_image(mask_image)

        try:
            # マスク画像をグレースケールに変換（既にグレースケールの場合はそのまま）
            if len(mask_image.shape) == 3:
                mask_gray = cv2.cvtColor(mask_image, cv2.COLOR_BGR2GRAY)
            else:
                mask_gray = mask_image

            # ターゲット画像とマスク画像のサイズが異なる場合、リサイズする
            target_image = self.target_image.copy()
            if (
                target_image.shape[0] != mask_image.shape[0]
                or target_image.shape[1] != mask_image.shape[1]
            ):

                # リサイズプロセッサのパラメータを設定
                self.resize_processor.width = mask_image.shape[1]
                self.resize_processor.height = mask_image.shape[0]

                # リサイズ実行
                target_image = self.resize_processor.process(target_image)

            # 結果画像の初期化（マスクの大きさに合わせる）
            result = np.zeros_like(target_image)

            # マスクを作成（白ピクセルまたは黒ピクセルを使用）
            if self.use_white_pixels:
                # 白ピクセル部分（255）をマスクとして使用
                mask = mask_gray
            else:
                # 黒ピクセル部分（0）をマスクとして使用
                mask = cv2.bitwise_not(mask_gray)

            # マスク部分にターゲット画像を適用
            result = cv2.bitwise_and(target_image, target_image, mask=mask)

            # マスク以外の部分をマスク画像で埋める
            inv_mask = cv2.bitwise_not(mask)

            # マスク画像がグレースケールの場合、3チャンネルに変換
            if len(mask_image.shape) == 2:
                mask_to_use = cv2.cvtColor(mask_image, cv2.COLOR_GRAY2BGR)
            else:
                mask_to_use = mask_image

            mask_part = cv2.bitwise_and(mask_to_use, mask_to_use, mask=inv_mask)
            result = cv2.add(result, mask_part)

            return result

        except cv2.error as e:
            raise ProcessorRuntimeError(f"Error during mask composition: {e}")
        except Exception as e:
            raise ProcessorRuntimeError(f"Unexpected error in {self.name}: {e}")

    @staticmethod
    def get_default_config() -> Dict[str, Any]:
        """
        マスク合成プロセッサのデフォルト設定を返す.

        Returns:
            Dict[str, Any]: デフォルト設定.
        """
        return {
            "target_image": "original",  # デフォルトはオリジナル画像
            "use_white_pixels": True,  # デフォルトでは白ピクセル部分を元画像で置き換え
        }
