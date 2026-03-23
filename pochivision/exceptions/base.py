"""pochivision共通例外の基底クラスを定義するモジュール."""


class VisionCaptureError(Exception):
    """
    pochivision全体の基底例外クラス.

    すべての独自例外はこのクラスを継承すること。
    """

    pass
