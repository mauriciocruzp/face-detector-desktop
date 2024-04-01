import cv2
import os
import imutils

detected_faces_path = "data_capture/photos/user_detected_faces"

if not os.path.exists(detected_faces_path):
    os.makedirs(detected_faces_path)
    print(f"folder created: ${detected_faces_path}")

cap = cv2.VideoCapture(0)
# cap = cv2.VideoCapture(0,cv2.CAP_DSHOW)
# cap = cv2.VideoCapture('VID_20201206_001148.mp4')
print("detecting...")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

print("detected!")
count = 0

while True:

    ret, frame = cap.read()
    if ret == False:
        break
    frame = imutils.resize(frame, width=640)
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    auxFrame = frame.copy()

    faces = face_classifier.detectMultiScale(gray, 1.2, 5)

    for x, y, w, h in faces:
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        face = auxFrame[y : y + h, x : x + w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
        cv2.imwrite(detected_faces_path + "/user_face_{}.jpg".format(count), face)
        count = count + 1
    cv2.imshow("frame", frame)

    k = cv2.waitKey(1)
    if k == 27 or count >= 350:
        break

print("finished!")

cap.release()
cv2.destroyAllWindows()
