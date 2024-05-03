import cv2

# 加载OpenCV的人脸级联分类器
face_cascade = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

# 加载OpenCV的微笑级联分类器
smile_cascade = cv2.CascadeClassifier('haarcascade_smile.xml')

# 打开摄像头
video_capture = cv2.VideoCapture(0)

while True:
    # 读取每一帧图像
    ret, frame = video_capture.read()

    # 将图像转换为灰度图
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # 检测人脸
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))

    # 对每个检测到的人脸进行微笑检测
    for (x, y, w, h) in faces:
        roi_gray = gray[y:y+h, x:x+w]
        roi_color = frame[y:y+h, x:x+w]

        # 检测微笑
        smiles = smile_cascade.detectMultiScale(roi_gray, scaleFactor=1.7, minNeighbors=22, minSize=(25, 25))

        # 根据微笑检测结果绘制矩形框和显示文本
        if len(smiles) > 0:
            cv2.rectangle(roi_color, (smiles[0][0], smiles[0][1]), (smiles[0][0] + smiles[0][2], smiles[0][1] + smiles[0][3]), (0, 255, 0), 2)
            cv2.putText(roi_color, 'Smiling', (smiles[0][0], smiles[0][1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

        # 绘制人脸矩形框
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)

    # 显示结果图像
    cv2.imshow('Video', frame)

    # 按下 'q' 键退出循环
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# 释放摄像头和关闭窗口
video_capture.release()
cv2.destroyAllWindows()

# 释放摄像头和关闭窗口
video_capture.release()
cv2.destroyAllWindows()
