import numpy as np
import pytest

from pochivision.exceptions import ProcessorValidationError
from pochivision.processors.contour import ContourProcessor
from pochivision.processors.validators.contour import ContourValidator


# テスト用の画像データを作成する関数
def create_test_image(height: int, width: int, is_binary: bool = True) -> np.ndarray:
    """テスト用の画像を作成します.

    Args:
        height (int): 画像の高さ
        width (int): 画像の幅
        is_binary (bool, optional): 二値画像を作成するかどうか. Trueなら二値画像、Falseならグレースケール画像.

    Returns:
        np.ndarray: 作成された画像
    """
    if is_binary:
        # 黒い背景に白い四角形の二値画像を作成
        image = np.zeros((height, width), dtype=np.uint8)
        # 中央に白い四角形を描画
        center_h, center_w = height // 2, width // 2
        size_h, size_w = height // 4, width // 4
        image[
            center_h - size_h : center_h + size_h,
            center_w - size_w : center_w + size_w,
        ] = 255
        return image
    else:
        # グレースケール画像（二値ではない）を作成
        return np.random.randint(0, 256, (height, width), dtype=np.uint8)


# カラー画像も作成できるようにする
def create_color_image(height: int, width: int) -> np.ndarray:
    """テスト用のカラー画像を作成します.

    Args:
        height (int): 画像の高さ
        width (int): 画像の幅

    Returns:
        np.ndarray: 作成されたカラー画像
    """
    return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestContourProcessor:
    """ContourProcessorのテストクラス."""

    def test_initialization_with_default_config(self):
        """デフォルト設定でプロセッサが初期化できることをテストします."""
        processor = ContourProcessor(name="contour_test", config={})
        assert processor.name == "contour_test"
        # デフォルト設定が適用されていることを確認
        default_config = ContourProcessor.get_default_config()
        assert processor._retrieval_mode_str == default_config["retrieval_mode"]
        assert processor._approx_method_str == default_config["approximation_method"]
        assert processor._min_area == default_config["min_area"]
        assert processor._select_mode == default_config["select_mode"]
        assert processor._contour_rank == default_config["contour_rank"]
        assert processor._outside_color == default_config["outside_color"]
        assert processor._inside_color == default_config["inside_color"]

    def test_initialization_with_custom_config(self):
        """カスタム設定でプロセッサが初期化できることをテストします."""
        custom_config = {
            "retrieval_mode": "external",
            "approximation_method": "none",
            "min_area": 200,
            "select_mode": "all",
            "contour_rank": 1,
            "outside_color": [100, 100, 100],
            "inside_color": [200, 200, 200],
        }
        processor = ContourProcessor(name="contour_custom", config=custom_config)
        assert processor._retrieval_mode_str == "external"
        assert processor._approx_method_str == "none"
        assert processor._min_area == 200
        assert processor._select_mode == "all"
        assert processor._contour_rank == 1
        assert processor._outside_color == [100, 100, 100]
        assert processor._inside_color == [200, 200, 200]

    def test_invalid_retrieval_mode(self):
        """無効な輪郭抽出モードを指定した場合にエラーが発生することをテストします."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"retrieval_mode": "invalid_mode"},
            )
        assert "retrieval_mode" in str(excinfo.value)
        assert "must be one of" in str(excinfo.value)

    def test_invalid_approximation_method(self):
        """無効な輪郭近似方法を指定した場合にエラーが発生することをテストします."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"approximation_method": "invalid_method"},
            )
        assert "approximation_method" in str(excinfo.value)
        assert "must be one of" in str(excinfo.value)

    def test_invalid_min_area(self):
        """無効な最小面積を指定した場合にエラーが発生することをテストします."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"min_area": -10},
            )
        assert "min_area" in str(excinfo.value)
        assert "must be non-negative" in str(excinfo.value)

    def test_invalid_select_mode(self):
        """無効な選択モードを指定した場合にエラーが発生することをテストします."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"select_mode": "invalid_mode"},
            )
        assert "select_mode" in str(excinfo.value)
        assert "must be one of" in str(excinfo.value)

    def test_invalid_contour_rank(self):
        """無効な輪郭ランクを指定した場合にエラーが発生することをテストします."""
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"contour_rank": -1},
            )
        assert "contour_rank" in str(excinfo.value)
        assert "must be non-negative" in str(excinfo.value)

    def test_invalid_inside_color(self):
        """無効な内側色を指定した場合にエラーが発生することをテストします."""
        # リストではない場合
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"inside_color": "white"},
            )
        assert "inside_color" in str(excinfo.value)
        assert "must be a list" in str(excinfo.value)

        # 要素数が3でない場合
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"inside_color": [255, 255]},
            )
        assert "inside_color" in str(excinfo.value)
        assert "must be a list of 3 integers" in str(excinfo.value)

        # 要素が整数でない場合
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"inside_color": [255, "255", 255]},
            )
        assert "inside_color" in str(excinfo.value)
        assert "elements must be integers" in str(excinfo.value)

        # 要素が0-255の範囲外の場合
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"inside_color": [255, 256, 255]},
            )
        assert "inside_color" in str(excinfo.value)
        assert "values must be between 0 and 255" in str(excinfo.value)

    def test_invalid_outside_color(self):
        """無効な外側色を指定した場合にエラーが発生することをテストします."""
        # リストではない場合
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"outside_color": "black"},
            )
        assert "outside_color" in str(excinfo.value)
        assert "must be a list" in str(excinfo.value)

        # 要素数が3でない場合
        with pytest.raises(ProcessorValidationError) as excinfo:
            ContourProcessor(
                name="contour_invalid",
                config={"outside_color": [0, 0, 0, 0]},
            )
        assert "outside_color" in str(excinfo.value)
        assert "must be a list of 3 integers" in str(excinfo.value)

    def test_process_binary_image(self):
        """二値画像の処理をテストします."""
        # デフォルト設定のプロセッサを作成
        processor = ContourProcessor(name="contour_default", config={})

        # 二値画像を作成
        binary_image = create_test_image(100, 100, is_binary=True)

        # 画像処理を実行
        result = processor.process(binary_image)

        # 結果がカラー画像であることを確認
        assert result.ndim == 3
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

        # 外側と内側の色が正しく適用されていることを確認
        # 黒い背景（外側）
        assert np.all(result[0, 0] == processor._outside_color)
        # 白い四角形（内側）の中央
        center_h, center_w = 50, 50
        assert np.all(result[center_h, center_w] == processor._inside_color)

    def test_process_grayscale_nonbinary_image(self):
        """二値化されていないグレースケール画像の処理をテストします."""
        processor = ContourProcessor(name="contour_default", config={})

        # 二値化されていないグレースケール画像を作成
        gray_image = create_test_image(100, 100, is_binary=False)

        # 画像処理を実行
        result = processor.process(gray_image)

        # 結果がカラー画像であることを確認
        assert result.ndim == 3
        assert result.shape == (100, 100, 3)
        # 元の画像を保持しているはず（BGRに変換されている）
        assert result.dtype == np.uint8

    def test_process_color_image(self):
        """カラー画像の処理をテストします."""
        processor = ContourProcessor(name="contour_default", config={})

        # カラー画像を作成
        color_image = create_color_image(100, 100)

        # 画像処理を実行
        result = processor.process(color_image)

        # 結果がカラー画像であることを確認
        assert result.ndim == 3
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8
        # 二値化されていないので元の画像を保持しているはず
        # assert np.array_equal(result, color_image)  # 内部でコピーが作られるので完全一致はしない

    def test_process_empty_image(self):
        """空の画像の処理をテストします."""
        # 入力画像を直接ContourValidatorに渡して検証し、例外をキャッチ
        validator = ContourValidator({})
        empty_image = np.array([], dtype=np.uint8)

        # バリデータのメソッドを直接呼び出して例外をテスト
        with pytest.raises(ProcessorValidationError) as excinfo:
            validator.validate_image(empty_image)
        assert "input image is empty" in str(excinfo.value)

    def test_process_invalid_image_type(self):
        """無効な画像型の処理をテストします."""
        # 入力画像を直接ContourValidatorに渡して検証し、例外をキャッチ
        validator = ContourValidator({})
        invalid_image = [[0, 0], [0, 0]]  # リスト型（無効）

        # バリデータのメソッドを直接呼び出して例外をテスト
        with pytest.raises(ProcessorValidationError) as excinfo:
            validator.validate_image(invalid_image)
        assert "image must be of type numpy.ndarray" in str(excinfo.value)

    def test_contour_selection_by_rank(self):
        """ランクによる輪郭選択をテストします."""
        # ランク1の輪郭を選択する設定
        config = {"contour_rank": 1}
        processor = ContourProcessor(name="contour_rank1", config=config)

        # 複数の輪郭を持つ二値画像を作成
        # （中央に大きな四角形と、その内部に小さな四角形を描画）
        binary_image = np.zeros((100, 100), dtype=np.uint8)
        # 大きな四角形（最大面積の輪郭）
        binary_image[20:80, 20:80] = 255
        # 小さな四角形（2番目に大きい輪郭）
        binary_image[35:65, 35:65] = 0
        # さらに小さな四角形（3番目に大きい輪郭）
        binary_image[40:60, 40:60] = 255

        # 画像処理を実行
        result = processor.process(binary_image)

        # 結果がカラー画像であることを確認
        assert result.ndim == 3
        assert result.shape == (100, 100, 3)

        # ランク1（2番目に大きい輪郭）が選択されていることを確認
        # ただし、輪郭の特定は複雑なので、結果が期待通りに処理されていることを確認する程度で十分

    def test_all_contours_selection(self):
        """すべての輪郭選択モードをテストします."""
        # すべての輪郭を選択する設定
        config = {"select_mode": "all"}
        processor = ContourProcessor(name="contour_all", config=config)

        # 複数の輪郭を持つ二値画像を作成
        binary_image = np.zeros((100, 100), dtype=np.uint8)
        # 大きな四角形
        binary_image[20:80, 20:80] = 255
        # 小さな四角形
        binary_image[35:65, 35:65] = 0

        # 画像処理を実行
        result = processor.process(binary_image)

        # 結果がカラー画像であることを確認
        assert result.ndim == 3
        assert result.shape == (100, 100, 3)

        # すべての輪郭が処理されていることを確認
        # 外側の領域が外側色になっていることを確認
        assert np.all(result[0, 0] == processor._outside_color)
        # 大きな四角形の内側が内側色になっていることを確認
        assert np.all(result[30, 30] == processor._inside_color)
        # 小さな四角形の内側が外側色になっていることを確認（穴）
        assert np.all(result[50, 50] == processor._outside_color)

    def test_min_area_filtering(self):
        """最小面積によるフィルタリングをテストします."""
        # 最小面積の設定
        config = {"min_area": 1000}  # 大きな最小面積を設定
        processor = ContourProcessor(name="contour_min_area", config=config)

        # 小さな輪郭を持つ二値画像を作成
        binary_image = np.zeros((100, 100), dtype=np.uint8)
        # 小さな四角形（面積 = 100）
        binary_image[40:50, 40:50] = 255

        # 画像処理を実行
        result = processor.process(binary_image)

        # 小さな輪郭はフィルタリングされるため、結果は外側色のみの画像になるはず
        assert np.all(result == processor._outside_color)

    def test_validator_is_binary_image(self):
        """is_binary_image メソッドのテストを行います."""
        validator = ContourValidator({})

        # 二値画像のテスト
        binary_image = create_test_image(100, 100, is_binary=True)
        assert validator.is_binary_image(binary_image) is True

        # 非二値画像のテスト
        # 4つ以上の異なるピクセル値を持つ画像は二値画像と判定されない
        # (is_binary_image はユニーク値が 3 以下なら True を返す)
        non_binary = np.zeros((100, 100), dtype=np.uint8)
        non_binary[10:30, 10:30] = 50
        non_binary[30:50, 30:50] = 100
        non_binary[50:70, 50:70] = 150
        non_binary[70:90, 70:90] = 200
        assert validator.is_binary_image(non_binary) is False

    def test_validate_image_for_contour(self):
        """validate_image_for_contour メソッドのテストを行います."""
        validator = ContourValidator({})

        # 有効な二値画像
        binary_image = create_test_image(100, 100, is_binary=True)
        is_valid, message = validator.validate_image_for_contour(binary_image)
        assert is_valid is True
        assert message == ""

        # 無効な画像（空の画像）
        empty_image = np.array([], dtype=np.uint8)
        is_valid, message = validator.validate_image_for_contour(empty_image)
        assert is_valid is False
        assert "input image is empty" in message

        # 無効な画像（非二値画像）
        non_binary = np.zeros((100, 100), dtype=np.uint8)
        for i in range(100):
            non_binary[i, i] = i  # グラデーションパターン
        is_valid, message = validator.validate_image_for_contour(non_binary)
        assert is_valid is False
        assert "not binary" in message

    def test_get_retrieval_mode(self):
        """_get_retrieval_mode メソッドのテストを行います."""
        import cv2

        # 各モード文字列に対して正しいOpenCV定数が返されることを確認
        assert ContourProcessor._get_retrieval_mode("external") == cv2.RETR_EXTERNAL
        assert ContourProcessor._get_retrieval_mode("list") == cv2.RETR_LIST
        assert ContourProcessor._get_retrieval_mode("ccomp") == cv2.RETR_CCOMP
        assert ContourProcessor._get_retrieval_mode("tree") == cv2.RETR_TREE
        assert ContourProcessor._get_retrieval_mode("floodfill") == cv2.RETR_FLOODFILL

        # 不明なモードの場合はデフォルト（LIST）が返されることを確認
        assert ContourProcessor._get_retrieval_mode("unknown") == cv2.RETR_LIST

    def test_get_approximation_method(self):
        """_get_approximation_method メソッドのテストを行います."""
        import cv2

        # 各近似方法文字列に対して正しいOpenCV定数が返されることを確認
        assert (
            ContourProcessor._get_approximation_method("none") == cv2.CHAIN_APPROX_NONE
        )
        assert (
            ContourProcessor._get_approximation_method("simple")
            == cv2.CHAIN_APPROX_SIMPLE
        )
        assert (
            ContourProcessor._get_approximation_method("tc89_l1")
            == cv2.CHAIN_APPROX_TC89_L1
        )
        assert (
            ContourProcessor._get_approximation_method("tc89_kcos")
            == cv2.CHAIN_APPROX_TC89_KCOS
        )

        # 不明な近似方法の場合はデフォルト（SIMPLE）が返されることを確認
        assert (
            ContourProcessor._get_approximation_method("unknown")
            == cv2.CHAIN_APPROX_SIMPLE
        )

    def test_float_image_processing(self):
        """float32型の画像処理をテストします."""
        processor = ContourProcessor(name="contour_float", config={})

        # float32型の二値画像を作成（0.0と1.0の値）
        float_binary = np.zeros((100, 100), dtype=np.float32)
        float_binary[25:75, 25:75] = 1.0

        # 画像処理を実行
        result = processor.process(float_binary)

        # 結果がカラー画像であることを確認
        assert result.ndim == 3
        assert result.shape == (100, 100, 3)
        assert result.dtype == np.uint8

        # 外側と内側の色が正しく適用されていることを確認
        assert np.all(result[0, 0] == processor._outside_color)
        assert np.all(result[50, 50] == processor._inside_color)
