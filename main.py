import dlib
import cv2
import face_recognition
import numpy as np

# pip install cmake
# pip install dlib


# Загрузка изображения
img = cv2.imread('image.jpg')

# Инициализация детектора лица
detector = dlib.get_frontal_face_detector()

# Обнаружение лиц на изображении
faces = detector(img, 1)

# Загрузка предобученной модели для выделения черт лица
predictor = dlib.shape_predictor('shape_predictor_68_face_landmarks.dat')

# Отображение найденных черт лица на изображении
for face in faces:
    # Нахождение координат лица на изображении
    x1 = face.left()
    y1 = face.top()
    x2 = face.right()
    y2 = face.bottom()

    # Выделение лица на изображении
    cv2.rectangle(img, (x1, y1), (x2, y2), (0, 255, 0), 2)

    # Выделение черт лица на изображении
    landmarks = predictor(img, face)
    for n in range(68):
        x = landmarks.part(n).x
        y = landmarks.part(n).y
        cv2.circle(img, (x, y), 2, (255, 0, 0), -1)

    # Выделение глаз на изображении
    left_eye = landmarks.part(36).x, landmarks.part(36).y
    right_eye = landmarks.part(45).x, landmarks.part(45).y
    cv2.rectangle(img, (left_eye[0], left_eye[1]), (right_eye[0], right_eye[1]), (0, 0, 255), 2)

    # Выделение носа на изображении
    nose = landmarks.part(30).x, landmarks.part(30).y
    cv2.rectangle(img, (nose[0]-10, nose[1]-10), (nose[0]+10, nose[1]+10), (255, 0, 0), 2)

    # Выделение губ на изображении
    mouth_left = landmarks.part(48).x, landmarks.part(48).y
    mouth_right = landmarks.part(54).x, landmarks.part(54).y
    cv2.rectangle(img, (mouth_left[0], mouth_left[1]), (mouth_right[0], mouth_right[1]), (0, 255, 255), 2)

# Загрузка изображений для сравнения
image1 = face_recognition.load_image_file("image.jpg")
image2 = face_recognition.load_image_file("image1.jpg")

# Вычисление эмбеддинга лица на каждом изображении
face_encoding1 = face_recognition.face_encodings(image1)[0]
face_encoding2 = face_recognition.face_encodings(image2)[0]

# Вычисление евклидова расстояния# Между эмбеддингами лиц
distance = np.linalg.norm(face_encoding1 - face_encoding2)

# Определение порогового значения для сравнения расстояний
threshold = 0.6

print("img1: ", face_encoding1)
print("img2: ", face_encoding2)
print("distance:", distance)

# Сравнение расстояний и вывод результата
if distance < threshold:
    print("Лица на изображениях одинаковы")
else:
    print("Лица на изображениях разные")

# Отображение изображения с найденными чертами лицаcv2.imshow('Распознавание черт лица на изображении', img)
cv2.imshow('Распознавание черт лица на изображении', img)
cv2.waitKey(0)
cv2.destroyAllWindows()