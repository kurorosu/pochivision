import cv2
import numpy as np
import dearpygui.dearpygui as dpg

# 画像サイズ（任意で調整）
WIDTH, HEIGHT = 640, 480

# カメラ起動
cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

# DearPyGui のセットアップ
dpg.create_context()
dpg.create_viewport(title='Camera Viewer', width=WIDTH, height=HEIGHT)
dpg.setup_dearpygui()

with dpg.texture_registry():
    texture_id = dpg.add_dynamic_texture(WIDTH, HEIGHT, [0] * WIDTH * HEIGHT * 4)

with dpg.window(label="Camera", no_scrollbar=True, no_title_bar=True, no_resize=True):
    dpg.add_image(texture_id)

dpg.show_viewport()

# 毎フレーム更新関数
def update():
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        frame = frame.flatten() / 255.0  # DearPyGui 用に正規化
        dpg.set_value(texture_id, frame.tolist())

# メインループ
while dpg.is_dearpygui_running():
    update()
    dpg.render_dearpygui_frame()

# 終了処理
cap.release()
dpg.destroy_context()
