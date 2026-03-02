#!/usr/bin/env python3
"""FFT Visualizer用のテスト画像作成スクリプト.

様々なパターンの画像を作成してFFTの特性を確認できます。
"""

from pathlib import Path

import cv2
import numpy as np


def create_test_images():
    """テスト用画像を作成します."""
    output_dir = Path("tools/test_images")
    output_dir.mkdir(exist_ok=True)

    # 1. 縦縞模様
    print("1. 縦縞模様画像を作成中...")
    img_vertical = np.zeros((256, 256), dtype=np.uint8)
    for i in range(0, 256, 20):
        img_vertical[:, i : i + 10] = 255
    cv2.imwrite(str(output_dir / "vertical_stripes.png"), img_vertical)

    # 2. 横縞模様
    print("2. 横縞模様画像を作成中...")
    img_horizontal = np.zeros((256, 256), dtype=np.uint8)
    for i in range(0, 256, 20):
        img_horizontal[i : i + 10, :] = 255
    cv2.imwrite(str(output_dir / "horizontal_stripes.png"), img_horizontal)

    # 3. チェッカーボード
    print("3. チェッカーボード画像を作成中...")
    img_checker = np.zeros((256, 256), dtype=np.uint8)
    for i in range(0, 256, 32):
        for j in range(0, 256, 32):
            if (i // 32 + j // 32) % 2 == 0:
                img_checker[i : i + 32, j : j + 32] = 255
    cv2.imwrite(str(output_dir / "checkerboard.png"), img_checker)

    # 4. 円形パターン
    print("4. 円形パターン画像を作成中...")
    img_circle = np.zeros((256, 256), dtype=np.uint8)
    center = (128, 128)
    cv2.circle(img_circle, center, 80, 255, -1)
    cv2.imwrite(str(output_dir / "circle.png"), img_circle)

    # 5. 斜め縞模様
    print("5. 斜め縞模様画像を作成中...")
    img_diagonal = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            if (i + j) % 40 < 20:
                img_diagonal[i, j] = 255
    cv2.imwrite(str(output_dir / "diagonal_stripes.png"), img_diagonal)

    # 6. ガウシアンノイズ
    print("6. ガウシアンノイズ画像を作成中...")
    img_noise = np.random.normal(128, 50, (256, 256)).astype(np.uint8)
    cv2.imwrite(str(output_dir / "gaussian_noise.png"), img_noise)

    # 7. 正弦波パターン
    print("7. 正弦波パターン画像を作成中...")
    img_sine = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            value = 128 + 127 * np.sin(2 * np.pi * j / 32)
            img_sine[i, j] = int(value)
    cv2.imwrite(str(output_dir / "sine_wave.png"), img_sine)

    # 8. 複合パターン
    print("8. 複合パターン画像を作成中...")
    img_complex = np.zeros((256, 256), dtype=np.uint8)
    for i in range(256):
        for j in range(256):
            # 複数の周波数成分を合成
            value = 128
            value += 50 * np.sin(2 * np.pi * j / 16)  # 高周波
            value += 30 * np.sin(2 * np.pi * j / 64)  # 低周波
            value += 20 * np.sin(2 * np.pi * i / 32)  # 縦方向成分
            img_complex[i, j] = np.clip(value, 0, 255)
        cv2.imwrite(str(output_dir / "complex_pattern.png"), img_complex)

    # 9. 高解像度テスト画像（リサイズ機能のテスト用）
    print("9. 高解像度テスト画像を作成中...")
    img_hd = np.zeros((1440, 2560), dtype=np.uint8)  # QHD解像度
    for i in range(0, 2560, 40):
        img_hd[:, i : i + 20] = 255
    cv2.imwrite(str(output_dir / "high_resolution.png"), img_hd)

    print(f"\n✓ テスト画像を {output_dir} フォルダに作成しました")
    print("\n作成された画像:")
    for img_file in output_dir.glob("*.png"):
        print(f"  - {img_file.name}")

    print("\n使用例:")
    print(f"python tools/fft_visualizer.py {output_dir}/vertical_stripes.png")
    print(f"python tools/fft_visualizer.py {output_dir}/checkerboard.png")


if __name__ == "__main__":
    create_test_images()
