"""特徴量抽出器テスト用の共通ダミー画像ヘルパー."""

import numpy as np


class DummyImages:
    """振る舞いテスト用のダミー画像を生成するヘルパークラス."""

    @staticmethod
    def uniform(size: int = 64, value: int = 128) -> np.ndarray:
        """単色画像."""
        return np.full((size, size), value, dtype=np.uint8)

    @staticmethod
    def white(size: int = 64) -> np.ndarray:
        """全白画像."""
        return np.full((size, size), 255, dtype=np.uint8)

    @staticmethod
    def black(size: int = 64) -> np.ndarray:
        """全黒画像."""
        return np.zeros((size, size), dtype=np.uint8)

    @staticmethod
    def gradient(size: int = 64) -> np.ndarray:
        """滑らかな垂直グラデーション."""
        img = np.zeros((size, size), dtype=np.uint8)
        for i in range(size):
            img[i, :] = int(i * 255 / (size - 1))
        return img

    @staticmethod
    def h_stripe(size: int = 64, period: int = 4) -> np.ndarray:
        """水平ストライプ."""
        img = np.zeros((size, size), dtype=np.uint8)
        for k in range(period // 2):
            img[k::period, :] = 255
        return img

    @staticmethod
    def v_stripe(size: int = 64, period: int = 4) -> np.ndarray:
        """垂直ストライプ."""
        img = np.zeros((size, size), dtype=np.uint8)
        for k in range(period // 2):
            img[:, k::period] = 255
        return img

    @staticmethod
    def checker(size: int = 64) -> np.ndarray:
        """チェッカーボード."""
        img = np.zeros((size, size), dtype=np.uint8)
        img[0::2, 0::2] = 255
        img[1::2, 1::2] = 255
        return img

    @staticmethod
    def random(size: int = 64, seed: int = 42) -> np.ndarray:
        """ランダム画像 (シード固定)."""
        np.random.seed(seed)
        return np.random.randint(0, 256, (size, size), dtype=np.uint8)
