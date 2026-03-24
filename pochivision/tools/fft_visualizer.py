#!/usr/bin/env python3
"""Simple FFT Visualization Tool.

画像のFFTを可視化し、ローパス・ハイパスフィルタを切り替えて表示できるツールです。
"""

import argparse
import sys
from pathlib import Path
from typing import Optional

import cv2
import numpy as np


class SimpleFFTVisualizer:
    """シンプルなFFTビジュアライザークラス."""

    def __init__(self, image_path: str):
        """初期化処理.

        Args:
            image_path (str): 画像ファイルのパス
        """
        self.image_path = Path(image_path)
        self.img: Optional[np.ndarray] = None
        self.fshift: Optional[np.ndarray] = None
        self.spectrum_display: Optional[np.ndarray] = None
        self.window_name = "FFT Visualizer - Spectrum (left) and Filtered Image (right)"
        self.filter_mode = "original"  # "original", "lowpass", "highpass"
        self.filter_radius = 50

    def load_image(self) -> bool:
        """画像を読み込みます."""
        if not self.image_path.exists():
            print(f"エラー: 画像ファイルが見つかりません: {self.image_path}")
            return False

        # 画像読み込み（元の形式で読み込み）
        original_img = cv2.imread(str(self.image_path))

        if original_img is None:
            print(f"エラー: 画像ファイルを読み込めません: {self.image_path}")
            return False

        # グレースケール変換（3チャンネルのグレースケール画像にも対応）
        if len(original_img.shape) == 3:
            self.img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        elif len(original_img.shape) == 2:
            self.img = original_img.copy()
        else:
            print(
                f"エラー: 対応していない画像形式です。画像の次元: {original_img.shape}"
            )
            return False

        original_height, original_width = self.img.shape
        print(f"画像を読み込みました: {self.image_path}")
        print(f"元画像サイズ: {original_width} x {original_height}")

        # FHD（1920x1080）を超える場合はFHDにリサイズ（表示用と処理用の両方）
        fhd_width, fhd_height = 1920, 1080
        if original_width > fhd_width or original_height > fhd_height:
            # アスペクト比を保持してFHDに収まるようにリサイズ
            scale_w = fhd_width / original_width
            scale_h = fhd_height / original_height
            scale = min(scale_w, scale_h)

            new_width = int(original_width * scale)
            new_height = int(original_height * scale)

            self.img = cv2.resize(
                self.img, (new_width, new_height), interpolation=cv2.INTER_AREA
            )
            print(
                f"FHDサイズにリサイズしました: {new_width} x {new_height} "
                f"(元: {original_width} x {original_height})"
            )
        else:
            print("リサイズは不要です（FHD以下）")

        return True

    def compute_fft(self) -> None:
        """FFTを計算します."""
        if self.img is None:
            raise ValueError("画像が読み込まれていません")

        # FFT + シフト
        f = np.fft.fft2(self.img)
        self.fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(self.fshift) + 1)

        # 表示用に正規化
        self.spectrum_display = cv2.normalize(
            magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    def apply_filter(self) -> np.ndarray:
        """現在のフィルタモードに応じて画像をフィルタリングします."""
        if self.img is None:
            raise ValueError("画像が読み込まれていません")
        if self.fshift is None:
            raise ValueError("FFTが計算されていません")

        if self.filter_mode == "original":
            return self.img

        # フィルタマスクを作成（NumPy配列操作で高速化）
        mask = np.zeros_like(self.fshift, dtype=complex)
        center_y, center_x = self.fshift.shape[0] // 2, self.fshift.shape[1] // 2

        # 距離マップを一度に計算
        y_indices, x_indices = np.ogrid[: self.fshift.shape[0], : self.fshift.shape[1]]
        distance_map = np.sqrt(
            (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2
        )

        if self.filter_mode == "lowpass":
            # 低周波のみ通す
            mask[distance_map <= self.filter_radius] = self.fshift[
                distance_map <= self.filter_radius
            ]
        elif self.filter_mode == "highpass":
            # 高周波のみ通す
            mask[distance_map > self.filter_radius] = self.fshift[
                distance_map > self.filter_radius
            ]

        # 逆FFT
        f_ishift = np.fft.ifftshift(mask)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        # 正規化
        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def update_display(self) -> None:
        """表示を更新します."""
        if self.spectrum_display is None:
            raise ValueError("スペクトラムが計算されていません")

        # フィルタリングされた画像を取得
        filtered_img = self.apply_filter()

        # カラー画像に変換
        spectrum_color = cv2.cvtColor(self.spectrum_display, cv2.COLOR_GRAY2BGR)
        filtered_color = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

        # フィルター半径を可視化
        if self.filter_mode != "original":
            center_y, center_x = (
                spectrum_color.shape[0] // 2,
                spectrum_color.shape[1] // 2,
            )

            if self.filter_mode == "lowpass":
                # ローパス: 内側の円（通す範囲）を緑で表示
                cv2.circle(
                    spectrum_color,
                    (center_x, center_y),
                    self.filter_radius,
                    (0, 255, 0),
                    2,
                )
                cv2.putText(
                    spectrum_color,
                    "PASS",
                    (center_x - 20, center_y - self.filter_radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )
            elif self.filter_mode == "highpass":
                # ハイパス: 外側の円（カットする範囲）を赤で表示
                cv2.circle(
                    spectrum_color,
                    (center_x, center_y),
                    self.filter_radius,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    spectrum_color,
                    "CUT",
                    (center_x - 15, center_y - self.filter_radius - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 0, 255),
                    2,
                )
                cv2.putText(
                    spectrum_color,
                    "PASS",
                    (center_x - 20, center_y + self.filter_radius + 25),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.5,
                    (0, 255, 0),
                    2,
                )

        # 結合して表示
        combined = np.hstack([spectrum_color, filtered_color])

        # 表示用にFHDを超える場合はさらにリサイズ
        display_height, display_width = combined.shape[:2]
        fhd_width, fhd_height = 1920, 1080
        if display_width > fhd_width or display_height > fhd_height:
            scale_w = fhd_width / display_width
            scale_h = fhd_height / display_height
            scale = min(scale_w, scale_h)

            new_display_width = int(display_width * scale)
            new_display_height = int(display_height * scale)
            combined = cv2.resize(
                combined,
                (new_display_width, new_display_height),
                interpolation=cv2.INTER_AREA,
            )

        # 情報を表示
        info_text = f"Filter: {self.filter_mode.upper()} | Radius: {self.filter_radius}"
        cv2.putText(
            combined,
            info_text,
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.7,
            (255, 255, 255),
            2,
        )

        instructions = "Press 'f': cycle filters | '+/-': radius | 'q': quit"
        cv2.putText(
            combined,
            instructions,
            (10, combined.shape[0] - 20),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            (255, 255, 255),
            1,
        )

        cv2.imshow(self.window_name, combined)

    def run(self) -> None:
        """ビジュアライゼーションを実行します."""
        if not self.load_image():
            return

        self.compute_fft()

        # ウィンドウ作成
        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

        # ウィンドウサイズ調整（FHDに収まるように）
        if self.img is not None:
            display_width = self.img.shape[1] * 2
            display_height = self.img.shape[0]
            fhd_width, fhd_height = 1920, 1080

            if display_width > fhd_width or display_height > fhd_height:
                scale_w = fhd_width / display_width
                scale_h = fhd_height / display_height
                scale = min(scale_w, scale_h)
                cv2.resizeWindow(
                    self.window_name,
                    int(display_width * scale),
                    int(display_height * scale),
                )

        print("\n=== Simple FFT Visualizer ===")
        print("左側: 周波数スペクトラム")
        print("右側: フィルタリングされた画像")
        print("操作:")
        print("  'f': フィルタ切り替え (Original → LowPass → HighPass)")
        print("  '+': フィルタ半径を増加")
        print("  '-': フィルタ半径を減少")
        print("  'q': 終了")

        # 初期表示
        self.update_display()

        # キー入力待ち
        while True:
            key = cv2.waitKey(1) & 0xFF
            if (
                key == ord("q")
                or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            elif key == ord("f"):
                # フィルタモード切り替え
                if self.filter_mode == "original":
                    self.filter_mode = "lowpass"
                elif self.filter_mode == "lowpass":
                    self.filter_mode = "highpass"
                else:
                    self.filter_mode = "original"
                print(f"フィルタモード: {self.filter_mode}")
                self.update_display()
            elif key == ord("+") or key == ord("="):  # +キー
                # 半径を増加
                self.filter_radius = min(200, self.filter_radius + 10)
                print(f"フィルタ半径: {self.filter_radius}")
                self.update_display()
            elif key == ord("-"):  # -キー
                # 半径を減少
                self.filter_radius = max(10, self.filter_radius - 10)
                print(f"フィルタ半径: {self.filter_radius}")
                self.update_display()

        cv2.destroyAllWindows()


def main() -> None:
    """メイン関数."""
    parser = argparse.ArgumentParser(description="Simple FFT Visualization Tool")
    parser.add_argument("image_path", help="画像ファイルのパス")

    args = parser.parse_args()

    try:
        visualizer = SimpleFFTVisualizer(args.image_path)
        visualizer.run()
    except KeyboardInterrupt:
        print("\n中断されました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
