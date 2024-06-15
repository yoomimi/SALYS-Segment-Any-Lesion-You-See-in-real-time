import socket
import torch
import cv2
import os
import numpy as np
from pathlib import Path
from box import Box
import lightning as L
import io
import signal
from config import cfg
from dataset import load_datasets
from model import MODELS
import threading
import re
from PIL import Image
import torch.nn.functional as F

SERVER_IP = "165.132.56.201"
SERVER_PORT = 5000
maxrecvsize = 20000000

# 종료 플래그 추가
exit_event = threading.Event()

def setup_server():
    server_sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server_sock.bind((SERVER_IP, SERVER_PORT))
    server_sock.listen(1)
    print(f"Server listening on {SERVER_IP}:{SERVER_PORT}")
    return server_sock

def load_model(cfg: Box):
    model = MODELS[cfg.model.name](cfg, inference=True)
    model.setup()
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model = model.to(device)
    return model

def create_gaze_image(gaze_points_array, image_shape):
    gaze_img = np.zeros(image_shape[:2], dtype=np.uint8)
    for (x, y) in gaze_points_array:
        cv2.circle(gaze_img, (x, y), 2, (255), -1)
    return gaze_img

def revcall(sock, n):
    data = bytearray()
    while len(data) < n:
        packet = sock.recv(n - len(data))
        if not packet:
            return None
        data.extend(packet)
    return data

def process_request(client_sock, model, fabric, cfg):
    original_image_path = "./data/T2SPIR/val/images/original/t2spir_1.nii_slice_12_obj_1.png_x_min=114_x_max=133_y_min=46_y_max=56.png"
    gaze_image_path = "./data/T2SPIR/val/images/gaze_images/t2spir_1.nii_slice_12_obj_1.png_x_min=114_x_max=133_y_min=46_y_max=56.png"
    while True:
        try:
            data_length_bytes = revcall(client_sock, 4)
            if not data_length_bytes or exit_event.is_set():
                break  # 클라이언트 연결이 종료되면 루프를 종료합니다.
            
            data_length = int.from_bytes(data_length_bytes, byteorder='big')
            data = revcall(client_sock, data_length)
            if not data or exit_event.is_set():
                break  # 클라이언트 연결이 종료되면 루프를 종료합니다.
            
            # 이미지와 응시 데이터 분리
            separator0 = "|TYP|".encode("utf-8")
            separator1 = "|IMG|".encode("utf-8")
            separator2 = "|GZP|".encode("utf-8")
            data = data[4:]
            separator_index0 = data.find(separator0)
            separator_index1 = data.find(separator1)
            separator_index2 = data.find(separator2)
            type_data = data[separator_index0 + len(separator0):separator_index1]
            image_data = data[separator_index1 + len(separator1):separator_index2]
            gaze_data = data[separator_index2:]
            
            if type_data == b'chest': 
                cfg.model.checkpoint = './out/training/base_config_chest/chest.pth'
            elif type_data == b'cardiac': 
                cfg.model.checkpoint = './out/training/base_config_chest/cardiac.pth'
            else: cfg.model.checkpoint = cfg.model.checkpoint 
            # 둘다 아니면 default인 t2spir의 path를 쓰게 됨
            
            fixed_img_shape = (1080, 1080, 3)
            imagenp = np.frombuffer(image_data, dtype=np.uint8).reshape(fixed_img_shape)
            imagePIL = Image.fromarray(imagenp)
            imagePIL.save(original_image_path)
            print('original image saved!')
            
            gaze_data = gaze_data.decode('utf-8')
            gaze_data = gaze_data.replace('|GZP|', '')
            gaze_items = gaze_data.split(',')
            gaze_ints = [max(int(float(item)), 0) for item in gaze_items]
            gaze_points = [(gaze_ints[i], gaze_ints[i + 1]) for i in range(0, len(gaze_ints), 2)]
            gaze_points_array = np.array(gaze_points)
            
            pos_xs_mean = np.mean(gaze_points_array[:, 0][gaze_points_array[:, 0] > 0])
            pos_ys_mean = np.mean(gaze_points_array[:, 1][gaze_points_array[:, 1] > 0])
            
            for z2 in range(gaze_points_array.shape[0]):
                if gaze_points_array[z2, 0] < 0:
                    gaze_points_array[z2, 0] = pos_xs_mean
                if gaze_points_array[z2, 1] < 0:
                    gaze_points_array[z2, 1] = pos_ys_mean
            
            gaze_data = gaze_points_array
            gaze_image = create_gaze_image(gaze_data, fixed_img_shape).reshape((1080,1080))
            cv2.imwrite(gaze_image_path, gaze_image)
            
            # Segmentation and response
            val_data = load_datasets(cfg, model.get_img_size())
            val_dataloader = fabric._setup_dataloader(val_data)
            with torch.no_grad():
                for iter, data in enumerate(val_dataloader):
                    images, prompt_input, _ = data
                    pred_masks, _ = model(images, prompt_input)
                    
                    pred_mask = pred_masks[0]
                    pred_mask = pred_mask.sigmoid()
                    resized_mask = F.interpolate(pred_mask[:, None, :, :], (1080, 1080))[0, 0] * 255
                    segmentation_mask = torch.stack([resized_mask, resized_mask, resized_mask], -1).detach().cpu().numpy().astype(np.uint8)
                    
                    segmentation_mask_path = "./data/T2SPIR/val/images/output/t2spir_1.nii_slice_12_obj_1.png_x_min=114_x_max=133_y_min=46_y_max=56.png"
                    
                    imageseg = Image.fromarray(segmentation_mask)
                    imageseg.save(segmentation_mask_path)
                    
                    segmentation_mask_length = len(segmentation_mask.tobytes())
                    seglength_info = np.array([segmentation_mask_length], dtype=np.int32).tobytes()
                    
                    data_to_send2 = seglength_info + segmentation_mask.tobytes()
                    client_sock.sendall(len(data_to_send2).to_bytes(4, byteorder='big'))
                    client_sock.sendall(data_to_send2)
        except Exception as e:
            print(f"Error processing request: {e}")
            break

def signal_handler(sig, frame):
    print("Signal received, shutting down...")
    exit_event.set()

def main():
    server_sock = setup_server()
    cfg = Box.from_yaml(filename="configs/base_config_t2spir.yaml")
    fabric = L.Fabric(accelerator="auto",
                      devices=cfg.num_devices,
                      strategy=L.fabric.strategies.DDPStrategy(start_method="popen", find_unused_parameters=True))
    model = load_model(cfg)

    # SIGINT 신호 핸들러 설정
    signal.signal(signal.SIGINT, signal_handler)
    
    try:
        while not exit_event.is_set():
            server_sock.settimeout(1.0)  # 1초 타임아웃
            try:
                client_sock, client_addr = server_sock.accept()
                print(f"Connection from {client_addr}")
                process_request(client_sock, model, fabric, cfg)
                client_sock.close()
            except socket.timeout:
                continue
    finally:
        server_sock.close()
        print("Server shut down")

if __name__ == "__main__":
    main()
