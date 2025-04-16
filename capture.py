import cv2
import json

import dearpygui.dearpygui as dpg

from capturelib.capture_manager import CaptureManager
from core import PipelineExecutor

with open("config.json", "r") as f:
    config = json.load(f)

capture_manager = CaptureManager()
pipeline = PipelineExecutor.from_config(
    config=config, capture_manager=capture_manager)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)
# 幅と高さを設定（例: 1920x1080）
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)  # 幅を設定
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # 高さを設定

WIDTH, HEIGHT = 640, 480
dpg.create_context()
dpg.create_viewport(title='Camera Viewer',
                    width=WIDTH + 200,
                    height=HEIGHT + 100)
dpg.setup_dearpygui()

with dpg.texture_registry():
    texture_id = dpg.add_dynamic_texture(
        WIDTH, HEIGHT, [0] * WIDTH * HEIGHT * 4)

with dpg.window(label="Frame", pos=(10, 10), width=WIDTH + 20, height=HEIGHT + 60):
    dpg.add_image(texture_id)

with dpg.window(label="Controls", pos=(WIDTH + 40, 10), width=130, height=100, no_scrollbar=True, no_resize=True):
    dpg.add_button(label="Capture", callback=lambda: take_snapshot())


def take_snapshot():
    ret, frame = cap.read()  # フレームを直接取得
    if ret:
        latest_frame = frame.copy()  # 最新のフレームを保存
        pipeline.run(latest_frame)  # パイプライン実行


def update():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba = rgba.flatten() / 255.0
        dpg.set_value(texture_id, rgba.tolist())


dpg.show_viewport()

while dpg.is_dearpygui_running():
    update()
    dpg.render_dearpygui_frame()

cap.release()
dpg.destroy_context()
