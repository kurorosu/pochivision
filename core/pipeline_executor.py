import cv2

from processors.registry import PROCESSOR_REGISTRY


class PipelineExecutor(object):
    """
    画像処理プロセッサ群を管理し、処理と保存を行うパイプライン実行クラス。

    Attributes:
        processors (list): 実行対象の画像処理プロセッサのリスト。
        capture_manager (CaptureManager): 処理結果の保存先ディレクトリを管理するオブジェクト。
        mode (str): 実行モード。"parallel" または "pipeline"。
    """

    def __init__(self, processors, capture_manager, mode="parallel"):
        """
        PipelineExecutor のコンストラクタ。

        Args:
            processors (list): 画像処理プロセッサのインスタンス群。
            capture_manager (CaptureManager): 保存先ディレクトリ管理。
            mode (str): 実行モード（"parallel" または "pipeline"）。デフォルトは "parallel"。
        """
        self.processors = processors
        self.capture_manager = capture_manager
        self.mode = mode

    @classmethod
    def from_config(cls, config, capture_manager):
        """
        設定ファイル（辞書）からインスタンスを生成。

        Args:
            config (dict): JSON等から読み込んだ設定辞書。
            capture_manager (CaptureManager): 保存用のディレクトリ管理インスタンス。

        Returns:
            PipelineExecutor: 構成済みの PipelineExecutor インスタンス。
        """
        processors = []
        for name in config["processors"]:
            processor_cls = PROCESSOR_REGISTRY[name]
            processor = processor_cls(name=name, config=config.get(name, {}))
            processors.append(processor)
        return cls(
            processors=processors,
            capture_manager=capture_manager,
            mode=config.get("mode", "parallel")
        )

    def run(self, image):
        """
        指定された画像に対してプロセッサを適用し、処理結果を保存する。

        Args:
            image (np.ndarray): 入力画像。
        """
        if self.mode == "parallel":
            for processor in self.processors:
                result = processor.process(image)
                self._save(result, processor.name)

        elif self.mode == "pipeline":
            result = image
            for processor in self.processors:
                result = processor.process(result)
            self._save(result, self.processors[-1].name)

    def _save(self, image, processor_name):
        """
        処理された画像を保存する内部メソッド。

        Args:
            image (np.ndarray): 処理済み画像。
            processor_name (str): 処理に使われたプロセッサの名前。
        """
        save_dir = self.capture_manager.get_processing_dir(processor_name)
        filename = f"snapshot_{processor_name}_{int(cv2.getTickCount())}.bmp"
        path = save_dir / filename
        cv2.imwrite(str(path), image)
        print(f"[{processor_name}] 保存しました: {path}")
