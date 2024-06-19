import sys
import socket
import time
import os
from pathlib import Path
import logging
import numpy as np
from PIL import Image
import cv2

from PyQt5.QtWidgets import QApplication, QWidget, QLabel, QVBoxLayout, QPushButton, QHBoxLayout
from PyQt5.QtGui import QPixmap, QPainter, QColor, QImage
from PyQt5.QtCore import Qt, QTimer

# Logging 설정
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Eye Tracker 관련 설정
screen_width = 1920
screen_height = 1080
max_gaze_points = 8
maxrecvsize = 250000

config_list = [
    'ENABLE_SEND_COUNTER',
    'ENABLE_SEND_TIME',
    'ENABLE_SEND_POG_FIX',
    'ENABLE_SEND_POG_BEST',
]

def setup_sockets():
    try:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.connect(("127.0.0.1", 4242))
        sock.settimeout(1.0)
        server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        server_sock.connect(("165.132.56.201", 5000))
        return sock, server_sock
    except ConnectionRefusedError:
        logger.error("Eye tracker connection refused!")
        exit(1)

def convert_position(x, y, img_width, img_height, offset_x, offset_y):
    # screen_x = int(x * img_width) + offset_x
    screen_x = int(x*screen_width)
    screen_y = int(y*screen_height)
    # screen_y = int(y * img_height) + offset_y
    return screen_x, screen_y

def configure_eye_tracker(sock, config_list):
    logger.info("Loading Eye Tracker configuration...")
    for c in config_list:
        msg = f'<SET ID="{c}" STATE="1" />\r\n'
        sock.send(msg.encode())
    time.sleep(1)
    sock.send(f'<SET ID="ENABLE_SEND_DATA" STATE="1" />\r\n'.encode())

def get_gaze_point(sock, img_width, img_height, offset_x, offset_y):
    try:
        msg = sock.recv(4096).decode()
        b_pogx = float(msg.split('BPOGX="')[1].split('"')[0])
        b_pogy = float(msg.split('BPOGY="')[1].split('"')[0])
        screen_x, screen_y = convert_position(b_pogx, b_pogy, img_width, img_height, offset_x, offset_y)
        return (screen_x, screen_y)
    except (IndexError, ValueError, TimeoutError):
        return None

def ensure_directory_exists(directory):
    if not os.path.exists(directory):
        os.makedirs(directory)

def send_to_server(image_path, gaze_points, server_sock, image_type, offset_x, offset_y, image_width, image_height, scale_factor):
    image = Image.open(image_path)
    resized_image = image.resize((1080, 1080))
    
    # 이미지가 1채널이면 3채널로 변환
    image_data_np = np.array(resized_image)
    if len(image_data_np.shape) == 2:  # Grayscale 이미지
        image_data_np = np.stack((image_data_np,) * 3, axis=-1)
    elif image_data_np.shape[2] == 1:  # 1채널 이미지
        image_data_np = np.concatenate((image_data_np, image_data_np, image_data_np), axis=2)
    image_data = image_data_np.tobytes()
    image_length = len(image_data)
    length_info = np.array([image_length], dtype=np.int32).tobytes()
    
    separator0 = "|TYP|"
    separator1 = "|IMG|"
    separator2 = "|GZP|"
    # print(gaze_points)
    # print(scale_factor)
    scaled_gaze_points = [(( x - offset_x ) / scale_factor / image_width * 1080 , ( y - offset_y ) / scale_factor / image_height * 1080 ) for x, y in gaze_points]
    # print(scaled_gaze_points)
    gaze_data = ','.join([f'{x},{y}' for x, y in scaled_gaze_points]).encode('utf-8')
    
    data_to_send = length_info + separator0.encode('utf-8') + image_type.encode('utf-8') + separator1.encode('utf-8') + image_data + separator2.encode('utf-8') + gaze_data
    # data_to_send = length_info + separator1.encode('utf-8') + image_data + separator2.encode('utf-8') + gaze_data
    
    server_sock.sendall(len(data_to_send).to_bytes(4, byteorder='big'))
    server_sock.sendall(data_to_send)

def revcall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def receive_from_server(server_sock, image_width, image_height):
    segmentation_mask = None
    
    data_length_bytes = revcall(server_sock, 4)
    # print(data_length_bytes)
    seg_length = int.from_bytes(data_length_bytes, byteorder='big')
    data = revcall(server_sock, seg_length)
    # print(len(data),'data_received_')
    
    data = data[4:]
    if data:
        segmentation_mask = np.frombuffer(data, dtype=np.uint8).reshape((1080, 1080, 3))
        segmentation_mask = cv2.resize(segmentation_mask, (image_width, image_height), interpolation=cv2.INTER_NEAREST)
        # print(segmentation_mask.shape)
        # segmentation_mask = np.transpose(segmentation_mask, (1, 0, 2))
        segmentation_mask[:, :, 0] = 0
        segmentation_mask[:, :, 2] = 0
        # 알파 채널 추가
        alpha_channel = np.ones((image_height, image_width), dtype=np.uint8) * 255
        non_value_indices = np.all(segmentation_mask == 0, axis=-1)
        alpha_channel[non_value_indices] = 0

        segmentation_mask = np.dstack((segmentation_mask, alpha_channel))
    return segmentation_mask

class EyeTrackerApp(QWidget):
    def __init__(self):
        super().__init__()
        self.image_index = 0
        self.image_dirs = {
            "cardiac": Path("./data/cardiac/original"),
            "chest": Path("./data/chest/original"),
            "t2spir": Path("./data/t2spir/original")
        }
        self.current_dir = self.image_dirs["cardiac"] # 열면 일단 default로 cardiac
        self.image_type = "cardiac" # 열면 일단 default로 cardiac
        self.image_paths = sorted(self.current_dir.glob("*.png"))
        self.gaze_points = []
        self.collecting_points = []
        self.gaze_buffer = []
        self.scale_factor = 1
        self.gaze_buffer_size = 10
        self.collecting = False
        self.current_image = None
        self.segmentation_mask = None
        self.sock, self.server_sock = setup_sockets()
        configure_eye_tracker(self.sock, config_list)
        self.initUI()
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_gaze_points)
        self.timer.start(16)  # Approx 60 FPS

    def initUI(self):
        self.setWindowTitle('Eye Tracker Visualization')
        self.setGeometry(100, 100, screen_width, screen_height)

        # Image display label
        self.label = QLabel(self)
        
        # Buttons for selecting image type
        self.btn_cardiac = QPushButton("Cardiac CT", self)
        self.btn_cardiac.clicked.connect(lambda: self.load_images("cardiac"))
        
        self.btn_chest = QPushButton("Chest Xray", self)
        self.btn_chest.clicked.connect(lambda: self.load_images("chest"))
        
        self.btn_t2spir = QPushButton("Abdomen MRI(T2Spir)", self)
        self.btn_t2spir.clicked.connect(lambda: self.load_images("t2spir"))
        
        # Next image button
        self.btn_next = QPushButton("Next Image", self)
        self.btn_next.clicked.connect(self.next_image)
        
        # Reset button (segmentation mask 지우고 현재 이미지에서 다시 시작)
        self.btn_reset = QPushButton("Clear", self)
        self.btn_reset.clicked.connect(self.reset_image)

        # Layouts
        self.layout_buttons = QHBoxLayout()
        self.layout_buttons.addWidget(self.btn_cardiac)
        self.layout_buttons.addWidget(self.btn_chest)
        self.layout_buttons.addWidget(self.btn_t2spir)
        self.layout_buttons.addWidget(self.btn_next)
        self.layout_buttons.addWidget(self.btn_reset)
        
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.label)
        self.layout.addLayout(self.layout_buttons)
        self.setLayout(self.layout)

        self.show()
        self.load_image()

    def load_images(self, image_type):
        self.current_dir = self.image_dirs[image_type]
        self.image_type = image_type
        self.image_paths = sorted(self.current_dir.glob("*.png"))
        self.image_index = 0
        self.load_image()

    def load_image(self):
        if self.image_index < len(self.image_paths):
            image_path = str(self.image_paths[self.image_index])
            self.original_pixmap = QPixmap(image_path)
            self.update_image_display()

    def update_image_display(self):
        if self.original_pixmap:
            scaled_pixmap = self.original_pixmap.scaled(screen_width, screen_height, Qt.KeepAspectRatio)

            self.current_image = QPixmap(screen_width, screen_height)
            self.current_image.fill(Qt.black)

            painter = QPainter(self.current_image)
            x_offset = (screen_width - scaled_pixmap.width()) // 2
            y_offset = (screen_height - scaled_pixmap.height()) // 2

            painter.drawPixmap(x_offset, y_offset, scaled_pixmap)
            painter.end()

            self.label.setPixmap(self.current_image)

            # 이미지 크기 및 오프셋 저장
            self.img_width = scaled_pixmap.width()
            self.img_height = scaled_pixmap.height()
            if self.img_width == screen_width: self.scale_factor = screen_width / self.img_width
            elif self.img_height == screen_height: self.scale_factor = screen_height / self.img_height
            self.offset_x = x_offset
            self.offset_y = y_offset


    def next_image(self):
        self.segmentation_mask = None
        self.image_index += 1
        if self.image_index >= len(self.image_paths):
            self.image_index = 0
        self.load_image()
        self.reset_gaze_points()

    def reset_image(self):
        self.segmentation_mask = None
        self.reset_gaze_points()
        self.update_image_display()

    def reset_gaze_points(self):
        self.gaze_points.clear()
        self.collecting_points.clear()
        self.collecting = False

    def update_gaze_points(self):
        gaze_point = get_gaze_point(self.sock, self.img_width, self.img_height, self.offset_x, self.offset_y)
        if gaze_point:
            self.gaze_buffer.append(gaze_point)
            
            # 일정 개수의 gaze 데이터를 수집하면 평균을 계산
            if len(self.gaze_buffer) >= self.gaze_buffer_size:
                avg_gaze_point = (
                    sum(x for x, y in self.gaze_buffer) // len(self.gaze_buffer),
                    sum(y for x, y in self.gaze_buffer) // len(self.gaze_buffer)
                )
                self.gaze_buffer.clear()
                
                if self.collecting:
                    self.collecting_points.append(avg_gaze_point)
                    if len(self.collecting_points) >= max_gaze_points:
                        send_to_server(str(self.image_paths[self.image_index]), self.collecting_points, self.server_sock, self.image_type, self.offset_x, self.offset_y, self.img_width, self.img_height, self.scale_factor)
                        print('sended!')
                        self.segmentation_mask = receive_from_server(self.server_sock, self.img_width, self.img_height)
                        self.reset_gaze_points()
                else:
                    self.gaze_points.append(avg_gaze_point)
                    if len(self.gaze_points) > max_gaze_points:
                        self.gaze_points.pop(0)

        self.update_overlay()

    
    def update_overlay(self):
        if self.current_image:
            image = self.current_image.copy()
            painter = QPainter(image)
            painter.setPen(QColor(255, 0, 0))
            for x, y in self.gaze_points:
                painter.drawEllipse(x, y, 5, 5)
            painter.setPen(QColor(0, 255, 0))
            for x, y in self.collecting_points:
                painter.drawEllipse(x, y, 5, 5)
            if self.segmentation_mask is not None:
                # print('segmentation mask is here!')
                # print(self.segmentation_mask.shape[1])
                # print(self.segmentation_mask.shape[0])
                mask_data = self.segmentation_mask.tobytes()
                mask = QImage(mask_data, self.segmentation_mask.shape[1], self.segmentation_mask.shape[0], QImage.Format_RGBA8888)
                painter.drawImage(self.offset_x, self.offset_y, mask)
            painter.end()
            self.label.setPixmap(image)

    
    def keyPressEvent(self, event):
        if event.key() == Qt.Key_S and not self.collecting:
            self.collecting = True
            self.collecting_points.clear()
            self.gaze_points.clear()

def main():
    app = QApplication(sys.argv)
    ex = EyeTrackerApp()
    sys.exit(app.exec_())

if __name__ == "__main__":
    main()
