import cv2
import os

photos_path = "data_capture/photos/unknown_photos"
photos_path_list = os.listdir(photos_path)

detected_faces_path = "data_capture/photos/unknown_detected_faces"

if not os.path.exists(detected_faces_path):
    os.makedirs(detected_faces_path)
    print(f"folder created: {detected_faces_path}")

face_classifier = cv2.CascadeClassifier(
    cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
)

count = 0
for image_name in photos_path_list:
    image = cv2.imread(photos_path + "/" + image_name)
    imageAux = image.copy()
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    faces = face_classifier.detectMultiScale(gray, 2.1, 1)

    for x, y, w, h in faces:
        cv2.rectangle(image, (x, y), (x + w, y + h), (128, 0, 255), 2)
    cv2.rectangle(image, (10, 5), (450, 25), (255, 255, 255), -1)

    for x, y, w, h in faces:
        face = imageAux[y : y + h, x : x + w]
        face = cv2.resize(face, (150, 150), interpolation=cv2.INTER_CUBIC)
        # cv2.imshow("face", face)
        # cv2.waitKey(0)
        cv2.imwrite(detected_faces_path + "/unknown_face_{}.jpg".format(count), face)
        count = count + 1

cv2.destroyAllWindows()
