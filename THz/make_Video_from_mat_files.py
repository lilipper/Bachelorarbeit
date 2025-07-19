import sys
sys.path.insert(1, "src_code/")
import process_rdf as prdf
import torch
import matplotlib.pyplot as plt
import skimage
import numpy as np
import cv2
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_agg import FigureCanvasAgg

output_folder_norm = r"D:\700GHz\videos_normalized"
output_folder = r"D:\700GHz\videos"
output_folder_colormap = r"D:\700GHz\videos_colormap"
output_folder_beautiful = r"D:\700GHz\videos_beautiful"
input_path = r"D:\700GHz\mat_files"

if not os.path.exists(output_folder):
    os.makedirs(output_folder)
if not os.path.exists(output_folder_norm):
    os.makedirs(output_folder_norm)
if not os.path.exists(output_folder_colormap):
    os.makedirs(output_folder_colormap)
if not os.path.exists(output_folder_beautiful):
    os.makedirs(output_folder_beautiful)

def render_frame_with_matplotlib(array):
    dpi = 100
    h, w = array.shape
    figsize = (w / dpi, h / dpi)

    fig = Figure(figsize=figsize, dpi=dpi)
    canvas = FigureCanvasAgg(fig)
    ax = fig.add_axes([0, 0, 1, 1])
    ax.imshow(array, cmap='viridis', aspect='auto')
    ax.axis('off')

    canvas.draw()

    # → Bildgröße direkt vom Canvas holen
    buf = canvas.buffer_rgba()
    width, height = canvas.get_width_height()

    image = np.frombuffer(buf, dtype='uint8').reshape((height, width, 4))[:, :, :3]
    return image

def make_video_beautiful(filename_mat, fps=4, device='cpu'):
    os.makedirs(output_folder, exist_ok=True)

    # Daten laden
    complex_raw_data, parameters = prdf.read_mat(filename_mat, device=device)
    print("Complex raw data shape: ", complex_raw_data.shape, parameters)
    processed_data, max_val_abs = prdf.process_complex_data(complex_raw_data, int(parameters["NF"]), device=device)
    depth_layers = processed_data.shape[0]

    # Log-Min/Max bestimmen
    log_min = float("inf")
    log_max = float("-inf")
    for i in range(depth_layers):
        img = torch.flipud(torch.pow(torch.abs(processed_data[i]), 2) / max_val_abs).transpose(0, 1).detach().cpu().numpy()
        img_log = 10 * np.log10(img + 1e-9)
        log_min = min(log_min, img_log.min())
        log_max = max(log_max, img_log.max())

    # Beispielbild rendern, um Größe zu ermitteln
    test_img = torch.flipud(torch.pow(torch.abs(processed_data[0]), 2) / max_val_abs).transpose(0, 1).detach().cpu().numpy()
    test_log = 10 * np.log10(test_img + 1e-9)
    norm = (test_log - log_min) / (log_max - log_min)
    rendered = render_frame_with_matplotlib(norm)
    height, width, _ = rendered.shape

    # VideoWriter starten
    output_path = os.path.join(output_folder_beautiful, os.path.basename(filename_mat).replace('.mat', '_pretty.mp4'))
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)

    # Video schreiben
    for i in range(depth_layers):
        img = torch.flipud(torch.pow(torch.abs(processed_data[i]), 2) / max_val_abs).transpose(0, 1).detach().cpu().numpy()
        img_log = 10 * np.log10(img + 1e-9)
        img_norm = (img_log - log_min) / (log_max - log_min)
        frame = render_frame_with_matplotlib(img_norm)
        video_writer.write(cv2.cvtColor(frame, cv2.COLOR_RGB2BGR))

    video_writer.release()
    print(f"Schönes Video gespeichert unter: {output_path}")

def make_video(filename_mat):
    fps = 4
    device = 'cpu'
    complex_raw_data, parameters = prdf.read_mat(filename_mat, device=device)
    print("Complex raw data shape: ", complex_raw_data.shape, parameters)
    processed_data, max_val_abs = prdf.process_complex_data(complex_raw_data, int(parameters["NF"]), device=device)
    depth_layers = processed_data.shape[0]

    img = torch.pow(torch.abs(processed_data[0, ...]), 2) / max_val_abs
    img = torch.flipud(img).transpose(0, 1).detach().cpu().numpy()
    img_log = 10 * np.log10(img)
        

    height, width = img_log.shape
    

    # VideoWriter initialisieren
    output_path = fr"{output_folder}\{os.path.basename(filename_mat).replace('.mat', '.mp4')}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # Durch alle Tiefenebenen iterieren
    for depth_layer in range(depth_layers):
        img = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)/max_val_abs).transpose(0, 1).detach().cpu().numpy()
        img_log = 10 * np.log10(img)

        img_uint8 = img_log.astype(np.uint8)

        video_writer.write(img_uint8)

    video_writer.release()
    print(f"Video gespeichert unter: {output_path}")

def make_video_norm(filename_mat):
    fps = 4
    device = 'cpu'
    complex_raw_data, parameters = prdf.read_mat(filename_mat, device=device)
    print("Complex raw data shape: ", complex_raw_data.shape, parameters)
    processed_data, max_val_abs = prdf.process_complex_data(complex_raw_data, int(parameters["NF"]), device=device)
    depth_layers = processed_data.shape[0]

    log_min = float("inf")
    log_max = float("-inf")

    for depth_layer in range(depth_layers):
        img = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)/max_val_abs).transpose(0, 1).detach().cpu().numpy()
        img_log = 10 * np.log10(img)

        log_min = min(log_min, img_log.min())
        log_max = max(log_max, img_log.max())
    
    height, width = img_log.shape
    # VideoWriter initialisieren
    output_path = fr"{output_folder_norm}\{os.path.basename(filename_mat).replace('.mat', '.mp4')}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=False)

    # Durch alle Tiefenebenen iterieren
    for depth_layer in range(depth_layers):
        img = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)/max_val_abs).transpose(0, 1).detach().cpu().numpy()
        img_log = 10 * np.log10(img)

        img_norm = (img_log - log_min) / (log_max - log_min) * 255
        img_uint8 = img_norm.astype(np.uint8)

        video_writer.write(img_uint8)

    video_writer.release()
    print(f"Video gespeichert unter: {output_path}")

def make_video_color(filename_mat):
    fps = 4
    device = 'cpu'
    complex_raw_data, parameters = prdf.read_mat(filename_mat, device=device)
    print("Complex raw data shape: ", complex_raw_data.shape, parameters)
    processed_data, max_val_abs = prdf.process_complex_data(complex_raw_data, int(parameters["NF"]), device=device)
    depth_layers = processed_data.shape[0]

    log_min = float("inf")
    log_max = float("-inf")

    for depth_layer in range(depth_layers):
        img = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)/max_val_abs).transpose(0, 1).detach().cpu().numpy()
        img_log = 10 * np.log10(img)

        log_min = min(log_min, img_log.min())
        log_max = max(log_max, img_log.max())
    
    height, width = img_log.shape
    # VideoWriter initialisieren
    output_path = fr"{output_folder_colormap}\{os.path.basename(filename_mat).replace('.mat', '.mp4')}"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    video_writer = cv2.VideoWriter(output_path, fourcc, fps, (width, height), isColor=True)
    cmap = plt.get_cmap('viridis')

    # Durch alle Tiefenebenen iterieren
    for depth_layer in range(depth_layers):
        img = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)/max_val_abs).transpose(0, 1).detach().cpu().numpy()
        img_log = 10 * np.log10(img)

        img_norm = (img_log - log_min) / (log_max - log_min)
        img_colored = cmap(img_norm)[:, :, :3]  # nur RGB, ohne Alpha
        img_colored = (img_colored * 255).astype(np.uint8)
        img_bgr = cv2.cvtColor(img_colored, cv2.COLOR_RGB2BGR)
        video_writer.write(img_bgr)

    video_writer.release()
    print(f"Video gespeichert unter: {output_path}")


for foldername, _, dateinamen in os.walk(input_path):
    for dateiname in dateinamen:
        if dateiname.endswith(".mat"):
            mat_file_path = os.path.join(foldername, dateiname)
            print(f"Verarbeite Datei: {mat_file_path}")
            make_video_beautiful(mat_file_path)