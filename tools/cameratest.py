"""
2台のカメラ映像を同時に表示するテスト用スクリプト.

Windows環境でCAP_DSHOWを利用し、カメラ0とカメラ1の映像を取得・表示する.
'q'キーで終了する.

Raises:
    SystemExit: カメラが開けない場合、エラーメッセージを表示して終了する.
"""

import cv2

# 2台のカメラをCAP_DSHOW（Windows用）でオープン
cap0 = cv2.VideoCapture(0, cv2.CAP_DSHOW)
cap1 = cv2.VideoCapture(1, cv2.CAP_DSHOW)

# カメラが開けたか確認
if not cap0.isOpened():
    print("カメラ0が開けませんでした")
    exit()
if not cap1.isOpened():
    print("カメラ1が開けませんでした")
    exit()

while True:
    ret0, frame0 = cap0.read()
    ret1, frame1 = cap1.read()

    if ret0:
        cv2.imshow("Camera 0", frame0)
    else:
        print("カメラ0から映像が取得できません")

    if ret1:
        cv2.imshow("Camera 1", frame1)
    else:
        print("カメラ1から映像が取得できません")

    # 'q'キーで終了
    if cv2.waitKey(1) & 0xFF == ord("q"):
        break

cap0.release()
cap1.release()
cv2.destroyAllWindows()
