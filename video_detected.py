import face_recognition
import cv2
import time
import configparser

# Подключаем config-файл
config = configparser.ConfigParser()
config.read("settings.ini")

print("Введите название видео:\n")
movie = input()  # "hamilton_clip.mp4"
input_movie = cv2.VideoCapture(movie)
length = int(input_movie.get(cv2.CAP_PROP_FRAME_COUNT))

# Параметры получаемого видео(убедитесь, что разрешение / частота кадров соответствуют входному видео!)
fourcc = cv2.VideoWriter_fourcc(*'XVID')
output_movie = cv2.VideoWriter(config["settings"]["OUTPUT"], fourcc, 29.97, (640, 360))

# Фото известных лиц

image_1 = face_recognition.load_image_file(config["settings"]["KNOW_FACE_1"])
image_1_face_encoding = face_recognition.face_encodings(image_1)[0]

image_2 = face_recognition.load_image_file(config["settings"]["KNOW_FACE_2"])
image_2_face_encoding = face_recognition.face_encodings(image_2)[0]

known_faces = [
    image_1_face_encoding,
    image_2_face_encoding
]

face_locations = []
face_encodings = []
face_names = []
frame_number = 0
prevTime = 0

while True:
    # Взять один кадр из видео
    ret, frame = input_movie.read()
    frame_number += 1

    if not ret:
        break

    # Переводим из BGR в RGB
    rgb_frame = frame[:, :, ::-1]

    # Поиск лиц в кадре
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    face_names = []
    for face_encoding in face_encodings:
        # Известно ли системе лицо
        match = face_recognition.compare_faces(known_faces, face_encoding, tolerance=0.50)

        name = None
        if match[0]:
            name = config["settings"]["KNOW_NAME_1"]
        elif match[1]:
            name = config["settings"]["KNOW_NAME_2"]

        face_names.append(name)

    for (top, right, bottom, left), name in zip(face_locations, face_names):
        if not name:
            continue

        # Отрисовка рамки вокруг лица
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Отрисовка облости под имя
        cv2.rectangle(frame, (left, bottom - 25), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX

        face_image = frame[top:bottom, left:right]
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
        frame[top:bottom, left:right] = face_image

        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)
        str = "FPS : %0.1f" % fps
        cv2.putText(frame, str, (0, 22), cv2.cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

        cv2.putText(frame, name, (left + 6, bottom - 6), font, 0.5, (255, 255, 255), 1)

    # Процесс обработки видео
    print("Обработка видео {} / {}".format(frame_number, length))
    output_movie.write(frame)

input_movie.release()
cv2.destroyAllWindows()
