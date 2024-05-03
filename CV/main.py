import cv2

# 加载面部检测器和眼睛检测器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')
eye_cascade = cv2.CascadeClassifier('haarcascade_eye.xml')

# 定义一些参数
eye_ratio_max = 0.3 # 眼睛高度与整个面部高度的比率最大值
eye_ratio_min = 0.1 # 眼睛高度与整个面部高度的比率最小值
eye_consecutive_frames = 3 # 眨眼的连续帧数

# 初始化变量
eye_aspect_ratio = 0
blink_counter = 0
consecutive_frames_count = 0

# 打开摄像头
cap = cv2.VideoCapture(0)

while True:
    # 读取一帧图像
    ret, frame = cap.read()

    # 转换为灰度图像
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测面部
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)

    # 对于每张脸，进行眼睛检测
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        eyes = eye_cascade.detectMultiScale(roi_gray)

        # 计算眼睛高度与整个面部高度的比率
        for (ex, ey, ew, eh) in eyes:
            eye_centroid = (x + ex + ew//2, y + ey + eh//2) # 眼睛的中心点坐标
            cv2.circle(frame, eye_centroid, 2, (255, 0, 0), -1) # 在眼睛中心处绘制一个蓝点
            eye_aspect_ratio = float(eh) / float(h) # 眼睛高度与整个面部高度的比率

            # 如果眼睛高度与整个面部高度的比率超出阈值，则计算眨眼次数
            if eye_aspect_ratio > eye_ratio_max or eye_aspect_ratio < eye_ratio_min:
                consecutive_frames_count = 0
            else:
                consecutive_frames_count += 1
                if consecutive_frames_count >= eye_consecutive_frames:
                    blink_counter += 1
                    consecutive_frames_count = 0

    # 在图像中绘制眨眼次数
    cv2.putText(frame, 'Blinks: {}'.format(blink_counter), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    # 显示图像
    cv2.imshow('Eye Blink Detection', frame)

    # 检测键盘输入，如果按下q键，则退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头并关闭所有窗口
cap.release()
cv2.destroyAllWindows()