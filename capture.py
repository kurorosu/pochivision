import cv2
import numpy as np
import dearpygui.dearpygui as dpg
import time

WIDTH, HEIGHT = 640, 480

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

dpg.create_context()
dpg.create_viewport(title='Camera Viewer', width=WIDTH + 200, height=HEIGHT + 100)
dpg.setup_dearpygui()

latest_frame = None

with dpg.texture_registry():
    texture_id = dpg.add_dynamic_texture(WIDTH, HEIGHT, [0] * WIDTH * HEIGHT * 4)

# カメラ映像用ウィンドウ（スクロールあり）
with dpg.window(label="Frame1", pos=(10, 10), width=WIDTH + 20, height=HEIGHT + 60):
    dpg.add_image(texture_id)

# コントロールウィンドウ（スクロールなし、ボタン専用）
with dpg.window(label="Controls", pos=(WIDTH + 40, 10), width=130, height=100, no_scrollbar=True, no_resize=True):
    dpg.add_button(label="Frame1 capture", callback=lambda: take_snapshot())

# 撮影処理
def take_snapshot():
    global latest_frame
    if latest_frame is not None:
        filename = f"snapshot_{int(time.time())}.bmp" 
        cv2.imwrite(filename, latest_frame)
        print(f"写真を保存しました（無圧縮BMP）: {filename}")

# ToDo:リサイズ前をlatest_frameにする
def update():
    global latest_frame
    ret, frame = cap.read()
    if ret:
        frame = cv2.resize(frame, (WIDTH, HEIGHT))
        latest_frame = frame.copy()
        rgba = cv2.cvtColor(frame, cv2.COLOR_BGR2RGBA)
        rgba = rgba.flatten() / 255.0
        dpg.set_value(texture_id, rgba.tolist())

dpg.show_viewport()

while dpg.is_dearpygui_running():
    update()
    dpg.render_dearpygui_frame()

cap.release()
dpg.destroy_context()
