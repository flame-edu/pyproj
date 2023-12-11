from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QImage, QPixmap
from PyQt6.QtWidgets import QApplication, QWidget, QPushButton, QMainWindow, QLabel, QVBoxLayout, QLineEdit, QComboBox, QFileDialog, QCheckBox

import sys
from pathlib import Path as path
import cv2
import numpy as np
import importlib.util
import os
from threading import Thread


# основное окно программы (в принципе единственное)
class MainWindow(QMainWindow):
    
    def __init__(self):
        super(MainWindow, self).__init__()

        # параметры непосредственно программы (детектора)
        self.model_path = os.path.join(os.path.dirname(__file__), '/animal_detection/model/detect.tflite') # путь до модели
        self.label_path = os.path.join(os.path.dirname(__file__), '/animal_detection/model/labelmap.txt') # путь до карты лейблов
        self.min_threshold = float(0.5) # минимальное значение уверенности для распознавания

        self.label = QLabel()
        self.label.setText("Выберите тип источника")

        self.label2 = QLabel()
        self.label2.setText("Введите путь до изображения")

        self.checkBox = QCheckBox()
        self.checkBox.setText("Сохранить результат")

        self.input = QLineEdit()

        self.comboBox = QComboBox()
        self.comboBox.addItems(['Картинка', 'Видео'])
        self.comboBox.setCurrentText('Картинка')

        # self.imageBox = QLabel()
        self.image_window = ImageWindow()
        # self.image_output = cv2.imread('Dogtaleshope.jpg')
        # self.updateImage(self.image_output)

        self.buttonStart = QPushButton()
        self.buttonStart.setText('Произвести обнаружение')
        self.buttonStart.clicked.connect(self.threadDetectImage)

        self.buttonChooseFile = QPushButton()
        self.buttonChooseFile.setText('Выбрать файл')
        self.buttonChooseFile.clicked.connect(self.pickFile)

        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addWidget(self.comboBox)
        self.layout.addWidget(self.label2)
        self.layout.addWidget(self.input)
        self.layout.addWidget(self.checkBox)
        self.layout.addWidget(self.buttonChooseFile)
        self.layout.addWidget(self.buttonStart)        

        self.container = QWidget()
        self.container.setLayout(self.layout)

        self.setWindowTitle("Распознавание объектов")

        self.setCentralWidget(self.container)

        self.image_window.show()

    def threadDetectImage(self):
        # self.input.setText(str(self.comboBox.currentIndex()))
        if (self.comboBox.currentIndex() == 0):
            thread = Thread(target=self.detectImage)
            thread.start()
        elif (self.comboBox.currentIndex() == 1):
            thread = Thread(target=self.detectVideo)
            thread.start()
    
    # выбор файла
    def pickFile(self):
        filename, _filter = QFileDialog.getOpenFileName(self, "Выбрать файл")
        if filename:
            self.input.setText(filename)

    # обновление картинки в окне результата
    def updateImage(self, image):
        image_output = QImage(image.data, image.shape[1], image.shape[0], image.strides[0], QImage.Format.Format_BGR888)
        self.image_window.imageBox.setPixmap(QPixmap.fromImage(image_output))

    # обнаружение объектов на изображении
    def detectImage(self):
        # получение пути к картинке
        image_path_final = os.path.join(self.input.text())
        if not os.path.exists(image_path_final):
            return
        # чтение карты лейблов
        with open(self.label_path, 'r') as f:
            labels = [l.strip() for l in f.readlines()]
        # инициализация tensorflow lite
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
        else:
            from tensorflow.lite.python.interpreter import Interpreter
        # загрузка модели
        interpreter = Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        float_model = (input_details[0]['dtype'] == np.float32)
        input_mean = 127.5
        input_std = 127.5
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
        # чтение изображения из файла и преобразование для дальнейшей работы
        image = cv2.imread(image_path_final)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        image_height, image_width, _ = image.shape
        image_rsz = cv2.resize(image_rgb, (width, height))
        input_data = np.expand_dims(image_rsz, axis=0)
        # преобразование данных в формат float32 (модель без квантизации)
        if float_model:
            input_data = (np.float32(input_data) - input_mean) / input_std
        # запуск обнаружения
        interpreter.set_tensor(input_details[0]['index'], input_data)
        interpreter.invoke()
        # получение результата 
        boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0] # координаты bounding box'ов с обнаруженными классами
        classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0] # классы (объекты)
        scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0] # показатели "уверенности"

        # отрисовка box'ов для результатов лучше min_threshold на картинке
        for i in range(len(scores)):
            if (scores[i] > self.min_threshold) and (scores[i] <= 1.0):
                # координаты границ прямоугольника (с учетом границ изображения)
                y_min = int(max(1, (boxes[i][0] * image_height)))
                y_max = int(min(image_height, (boxes[i][2] * image_height)))
                x_min = int(max(1, (boxes[i][1] * image_width)))
                x_max = int(min(image_width, (boxes[i][3] * image_width)))
                # сам прямоугольник
                cv2.rectangle(image, (x_min, y_min), (x_max, y_max), (10, 255, 0), 2)
                # подпись лейбла
                object_name = labels[int(classes[i])]
                label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.7, 2)
                # поиск координаты с учетом границы изображения
                label_y_min = max(y_min, label_size[1] + 10)
                # добавление фона для текста и самого текста
                cv2.rectangle(image, (x_min, label_y_min - label_size[1] - 10), (x_min + label_size[0], label_y_min+base_line - 10), (255, 255, 255), cv2.FILLED)
                cv2.putText(image, label, (x_min, label_y_min - 7), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)

        # отрисовка картинки в окне результата
        self.updateImage(image)
        # сохранение в файл (если чекбокс включен)
        if (self.checkBox.isChecked() == True):
            filename = "result.png"
            cv2.imwrite(filename, image)

    # обнаружение объектов на видео
    def detectVideo(self):
        # получение пути к видеофайлу
        image_path_final = os.path.join(self.input.text())
        if not os.path.exists(image_path_final):
            return
        # чтение карты лейблов
        with open(self.label_path, 'r') as f:
            labels = [l.strip() for l in f.readlines()]
        # инициализация tensorflow lite
        pkg = importlib.util.find_spec('tflite_runtime')
        if pkg:
            from tflite_runtime.interpreter import Interpreter
        else:
            from tensorflow.lite.python.interpreter import Interpreter
        # загрузка модели
        interpreter = Interpreter(model_path=self.model_path)
        interpreter.allocate_tensors()
        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()
        height = input_details[0]['shape'][1]
        width = input_details[0]['shape'][2]
        float_model = (input_details[0]['dtype'] == np.float32)
        input_mean = 127.5
        input_std = 127.5
        boxes_idx, classes_idx, scores_idx = 1, 3, 0
        # чтение видеофайла по пути
        video = cv2.VideoCapture(image_path_final)
        # работа с видеофайлом (по кадрам в цикле)
        while (video.isOpened()):
            # чтение очередного кадра видео
            ret, frame = video.read()
            if not ret:
                break
            # обработка кадра
            image_height, image_width, _ = frame.shape
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frame_rsz = cv2.resize(frame_rgb, (width, height), interpolation=cv2.INTER_AREA)
            input_data = np.expand_dims(frame_rsz, axis=0)
            # преобразование в float32
            if float_model:
                input_data = (np.float32(input_data) - input_mean / input_std)
            # запуск обнаружения
            interpreter.set_tensor(input_details[0]['index'], input_data)
            interpreter.invoke()
            # получение результата
            boxes = interpreter.get_tensor(output_details[boxes_idx]['index'])[0]
            classes = interpreter.get_tensor(output_details[classes_idx]['index'])[0]
            scores = interpreter.get_tensor(output_details[scores_idx]['index'])[0]
            # отрисовка результатов на кадре (для видео min_threshold выставлен ниже)
            for i in range(len(scores)):
                if (scores[i] > 0.1) and (scores[i] <= 1.0):
                    # нахождение координат для прямоугольника
                    y_min = int(max(1, (boxes[i][0] * image_height)))
                    y_max = int(min(image_height, (boxes[i][2] * image_height)))
                    x_min = int(max(1, (boxes[i][1] * image_width)))
                    x_max = int(min(image_width, (boxes[i][3] * image_width)))
                    # отрисовка прямоугольника
                    cv2.rectangle(frame, (x_min, y_min), (x_max, y_max), (10, 255, 0), 4)
                    # подпись лейбла
                    object_name = labels[int(classes[i])]
                    label = '%s: %d%%' % (object_name, int(scores[i] * 100))
                    label_size, base_line = cv2.getTextSize(label, cv2.FONT_HERSHEY_PLAIN, 0.7, 2)
                    label_y_min = max(y_min, label_size[1] + 10)
                    # отрисовка фона и текста
                    cv2.rectangle(frame, (x_min, label_y_min - label_size[1] - 10), (x_min + label_size[0], label_y_min+base_line - 10), (255, 255, 255), cv2.FILLED)
                    cv2.putText(frame, label, (x_min, label_y_min - 7), cv2.FONT_HERSHEY_PLAIN, 0.7, (0, 0, 0), 1)
            # отрисовка результата в окне результата
            self.updateImage(frame)
            # задержка кадра (50мс)
            if cv2.waitKey(50) == ord('q'):
                break
        # закрытие видео
        video.release()

# окно с результатом
class ImageWindow(QWidget):
    def __init__(self):
        super().__init__()
        self.layout = QVBoxLayout()
        self.imageBox = QLabel()
        self.setLayout(self.layout)
        self.layout.addWidget(self.imageBox)
        self.setWindowTitle("Результат")


app = QApplication([])

window = MainWindow()
window.show()


app.exec()
