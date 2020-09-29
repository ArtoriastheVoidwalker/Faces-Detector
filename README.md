**Для распознавания нового лица в видеофайлах и на веб-камерах
необходимо:**
 1) Добавить фотографию в папку 'img';
 2) В файле _settings.ini_ создать константу _KNOW_FACE_n_ = фото.расширение;
 3) В файле _settings.ini_ создать константу _KNOW_NAME_n_ = имя человека с фото;
 4) В файле _face_detected.py_ для нашего n прописываем следующий код:
 `image_n = face_recognition.load_image_file(config["settings"]["KNOW_FACE_n"])
  image_n_face_encoding = face_recognition.face_encodings(image_n)[0]`
 5) Далее заносим `image_n_face_encoding` в список известных лиц;
 `known_face_encodings = [
    image_n_face_encoding 
    ]`
 6) Затем добавим строчку `config["settings"]["KNOW_NAME_n"]` в список известных имён:
 `known_face_names = [
    config["settings"]["KNOW_NAME_n"]
   ]`          
 7) После этих действий система будет узнавать новое лицо.   