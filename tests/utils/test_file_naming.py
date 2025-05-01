import pytest  # noqa: F401

from utils.file_naming import FileNamingManager


def test_default_id_interval():
    manager = FileNamingManager()
    # デフォルトは1ごとにIDが増加
    filename1, id1, image1 = manager.get_filename("original", 0)
    filename2, id2, image2 = manager.get_filename("original", 0)
    assert id1 == 1
    assert id2 == 2
    assert image1 == 1
    assert image2 == 2
    assert filename1.startswith("original_")
    assert filename2.startswith("original_")


def test_custom_id_interval():
    manager = FileNamingManager()
    manager.set_id_interval("pipeline", 0, 3)
    # 3ごとにIDが増加
    results = [manager.get_filename("pipeline", 0) for _ in range(7)]
    ids = [r[1] for r in results]
    images = [r[2] for r in results]
    assert ids == [1, 1, 1, 2, 2, 2, 3]
    assert images == [1, 2, 3, 4, 5, 6, 7]


def test_multiple_prefix_and_camera():
    manager = FileNamingManager()
    manager.set_id_interval("pipeline", 0, 2)
    manager.set_id_interval("original", 1, 4)
    # camera0/pipeline
    id0_1 = manager.get_filename("pipeline", 0)[1]
    id0_2 = manager.get_filename("pipeline", 0)[1]
    id0_3 = manager.get_filename("pipeline", 0)[1]
    assert id0_1 == 1
    assert id0_2 == 1
    assert id0_3 == 2
    # camera1/original
    id1_1 = manager.get_filename("original", 1)[1]
    id1_2 = manager.get_filename("original", 1)[1]
    id1_3 = manager.get_filename("original", 1)[1]
    id1_4 = manager.get_filename("original", 1)[1]
    id1_5 = manager.get_filename("original", 1)[1]
    assert id1_1 == 1
    assert id1_2 == 1
    assert id1_3 == 1
    assert id1_4 == 1
    assert id1_5 == 2


def test_filename_format():
    manager = FileNamingManager()
    filename, id_index, image_index = manager.get_filename("test", 2, extension="jpg")
    assert filename.startswith("test_")
    assert filename.endswith(f"_id{id_index}_image{image_index}.jpg")
