"""fft サブコマンド: FFT ビジュアライザー."""

import sys
from pathlib import Path
from typing import Optional

import click
import cv2
import numpy as np


class SimpleFFTVisualizer:
    """シンプルなFFTビジュアライザークラス."""

    def __init__(self, image_path: str):
        """初期化処理.

        Args:
            image_path: 画像ファイルのパス.
        """
        self.image_path = Path(image_path)
        self.img: Optional[np.ndarray] = None
        self.fshift: Optional[np.ndarray] = None
        self.spectrum_display: Optional[np.ndarray] = None
        self.window_name = "FFT Visualizer - Spectrum (left) and Filtered Image (right)"
        self.filter_mode = "original"
        self.filter_radius = 50

    def load_image(self) -> bool:
        """画像を読み込む."""
        if not self.image_path.exists():
            print(f"エラー: 画像ファイルが見つかりません: {self.image_path}")
            return False

        original_img = cv2.imread(str(self.image_path))

        if original_img is None:
            print(f"エラー: 画像ファイルを読み込めません: {self.image_path}")
            return False

        if len(original_img.shape) == 3:
            self.img = cv2.cvtColor(original_img, cv2.COLOR_BGR2GRAY)
        elif len(original_img.shape) == 2:
            self.img = original_img.copy()
        else:
            print(
                f"エラー: 対応していない画像形式です. 画像の次元: {original_img.shape}"
            )
            return False

        original_height, original_width = self.img.shape
        print(f"画像を読み込みました: {self.image_path}")
        print(f"元画像サイズ: {original_width} x {original_height}")

        fhd_width, fhd_height = 1920, 1080
        if original_width > fhd_width or original_height > fhd_height:
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
            print("リサイズは不要です (FHD以下)")

        return True

    def compute_fft(self) -> None:
        """FFTを計算する."""
        if self.img is None:
            raise ValueError("画像が読み込まれていません")

        f = np.fft.fft2(self.img)
        self.fshift = np.fft.fftshift(f)
        magnitude_spectrum = 20 * np.log(np.abs(self.fshift) + 1)

        self.spectrum_display = cv2.normalize(
            magnitude_spectrum, None, 0, 255, cv2.NORM_MINMAX
        ).astype(np.uint8)

    def apply_filter(self) -> np.ndarray:
        """現在のフィルタモードに応じて画像をフィルタリングする."""
        if self.img is None:
            raise ValueError("画像が読み込まれていません")
        if self.fshift is None:
            raise ValueError("FFTが計算されていません")

        if self.filter_mode == "original":
            return self.img

        mask = np.zeros_like(self.fshift, dtype=complex)
        center_y, center_x = self.fshift.shape[0] // 2, self.fshift.shape[1] // 2

        y_indices, x_indices = np.ogrid[: self.fshift.shape[0], : self.fshift.shape[1]]
        distance_map = np.sqrt(
            (y_indices - center_y) ** 2 + (x_indices - center_x) ** 2
        )

        if self.filter_mode == "lowpass":
            mask[distance_map <= self.filter_radius] = self.fshift[
                distance_map <= self.filter_radius
            ]
        elif self.filter_mode == "highpass":
            mask[distance_map > self.filter_radius] = self.fshift[
                distance_map > self.filter_radius
            ]

        f_ishift = np.fft.ifftshift(mask)
        img_back = np.fft.ifft2(f_ishift)
        img_back = np.abs(img_back)

        return cv2.normalize(img_back, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    def update_display(self) -> None:
        """表示を更新する."""
        if self.spectrum_display is None:
            raise ValueError("スペクトラムが計算されていません")

        filtered_img = self.apply_filter()

        spectrum_color = cv2.cvtColor(self.spectrum_display, cv2.COLOR_GRAY2BGR)
        filtered_color = cv2.cvtColor(filtered_img, cv2.COLOR_GRAY2BGR)

        if self.filter_mode != "original":
            center_y, center_x = (
                spectrum_color.shape[0] // 2,
                spectrum_color.shape[1] // 2,
            )

            if self.filter_mode == "lowpass":
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

        combined = np.hstack([spectrum_color, filtered_color])

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
        """ビジュアライゼーションを実行する."""
        if not self.load_image():
            return

        self.compute_fft()

        cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)

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

        self.update_display()

        while True:
            key = cv2.waitKey(1) & 0xFF
            if (
                key == ord("q")
                or cv2.getWindowProperty(self.window_name, cv2.WND_PROP_VISIBLE) < 1
            ):
                break
            elif key == ord("f"):
                if self.filter_mode == "original":
                    self.filter_mode = "lowpass"
                elif self.filter_mode == "lowpass":
                    self.filter_mode = "highpass"
                else:
                    self.filter_mode = "original"
                print(f"フィルタモード: {self.filter_mode}")
                self.update_display()
            elif key == ord("+") or key == ord("="):
                self.filter_radius = min(200, self.filter_radius + 10)
                print(f"フィルタ半径: {self.filter_radius}")
                self.update_display()
            elif key == ord("-"):
                self.filter_radius = max(10, self.filter_radius - 10)
                print(f"フィルタ半径: {self.filter_radius}")
                self.update_display()

        cv2.destroyAllWindows()


@click.command()
@click.option(
    "--input", "-i", "input_path", type=str, required=True, help="入力画像パス"
)
def fft(input_path: str) -> None:
    """FFT ビジュアライザーを起動する."""
    try:
        visualizer = SimpleFFTVisualizer(input_path)
        visualizer.run()
    except KeyboardInterrupt:
        print("\n中断されました")
    except Exception as e:
        print(f"エラーが発生しました: {e}")
        sys.exit(1)
