import face_recognition
import cv2
import numpy as np
import time
import configparser

# Подключаем config-файл
config = configparser.ConfigParser()
config.read("settings.ini")

# Выбираем веб-камеру(по умолчанию идёт 0.Если спользуется n камер,то cv2.VideoCapture(n))
video_capture = cv2.VideoCapture(0)

# Загружаем изображение лица,которое будет известно системе
image_3 = face_recognition.load_image_file(config["settings"]["KNOW_FACE_3"])
image_3_face_encoding = face_recognition.face_encodings(image_3)[0]

# image_4 = face_recognition.load_image_file(config["settings"]["KNOW_FACE_4"])
# artyom_face_encoding = face_recognition.face_encodings(image_4)[0]

# Список известных системе лиц
known_face_encodings = [
    image_3_face_encoding,
    # image_4_face_encoding
]
# Список известных системе имён
known_face_names = [
    config["settings"]["KNOW_NAME_3"],

]
prevTime = 0
while True:

    # Берём кадр из видео
    ret, frame = video_capture.read()

    # Переводим из BGR в RGB
    rgb_frame = frame[:, :, ::-1]

    # Поиск лиц в кадре
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for (top, right, bottom, left), face_encoding in zip(face_locations, face_encodings):
        # key = cv2.waitKey(1)
        # if (key == ord('b')):
        # Блюр работает при нажатии на клавишу,обы работал всегда закоменьтить условие нижей
        # if cv2.waitKey(1) & 0xFF == ord('b'):
            # Блюр
        face_image = frame[top:bottom, left:right]
        face_image = cv2.GaussianBlur(face_image, (99, 99), 30)
        frame[top:bottom, left:right] = face_image
        # Известно ли системе лицо
        matches = face_recognition.compare_faces(known_face_encodings, face_encoding)

        name = "Unknown"

        # Если найденно совпадение с известными лицами,то name=известное лицо
        if True in matches:
            first_match_index = matches.index(True)
            name = known_face_names[first_match_index]

        face_distances = face_recognition.face_distance(known_face_encodings, face_encoding)
        best_match_index = np.argmin(face_distances)
        if matches[best_match_index]:
            name = known_face_names[best_match_index]

        # Отрисовка рамки вокруг лица
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 0, 255), 2)

        # Отрисовка облости под имя
        cv2.rectangle(frame, (left, bottom - 35), (right, bottom), (0, 0, 255), cv2.FILLED)
        font = cv2.FONT_HERSHEY_DUPLEX
        # Окно на весь экран
        cv2.startWindowThread()
        cv2.namedWindow("Faces detector", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Faces detector", cv2.WND_PROP_FULLSCREEN, cv2.cv2.WINDOW_FULLSCREEN)
        # Вывод имени
        cv2.putText(frame, name, (left + 6, bottom - 6), font, 1.0, (255, 255, 255), 1)
        # Отрисовка FPS
        curTime = time.time()
        sec = curTime - prevTime
        prevTime = curTime
        fps = 1 / (sec)
        str = "FPS : %0.1f" % fps
        cv2.putText(frame, str, (0, 22), cv2.cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0))

    # Отрисовка изображения
    cv2.imshow('Faces detector', frame)

    # Завершение работы
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

video_capture.release()
# cv2.destroyAllWindows()