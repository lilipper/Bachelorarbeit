import os
import sys
import numpy as np
import matplotlib.pyplot as plt
import torch
import scipy.io
import skimage
sys.path.insert(1, "src_code/")
import process_rdf as prdf
import imageio

# === Benutzereingabe ===
input_folder = input('Pfad zur .mat-Datei: ').strip()
output_folder = os.path.join(input_folder, 'USAF_images_2')
os.makedirs(output_folder, exist_ok=True)

device = 'cpu'
depth_layer = 701

def save_colormap_image(array, path):
    h, w = array.shape
    dpi = 100  # du kannst auch 200 oder 300 verwenden
    figsize = (w / dpi, h / dpi)
    
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(array, cmap='viridis')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)  # keine Ränder
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


# for file in os.listdir(input_folder):
#     if file.endswith('.mat'):
#         try:
#             filename_mat = os.path.join(input_folder, file)
#             complex_raw_data, parameters = prdf.read_mat(filename_mat, device=device)
            
#             processed_data, max_val_abs = prdf.process_complex_data(complex_raw_data, int(parameters["NF"]), device=device)
            
#             image_depth = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)/max_val_abs).transpose(0, 1).detach().cpu().numpy()

#             base_name = os.path.splitext(os.path.basename(filename_mat))[0]
#             skimage.io.imsave(fr'{output_folder}\z_{base_name}' + str(depth_layer) + '.tiff', image_depth, plugin="tifffile", check_contrast=False)
#             skimage.io.imsave(fr'{output_folder}\log_image_z_' + str(depth_layer) + '.tiff', 10*np.log10( image_depth), plugin="tifffile", check_contrast=False)
#         except Exception as e:
#             print(f"[ERROR] Fehler beim Verarbeiten von {file}: {e}")
#             continue



# === Funktion zum Speichern eines Colormap-Bildes ===
def save_colormap_image(array, path):
    h, w = array.shape
    dpi = 100
    figsize = (w / dpi, h / dpi)
    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    ax.imshow(array, cmap='viridis')
    ax.axis('off')
    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)

# === Schleife über alle .mat-Dateien ===
for file in os.listdir(input_folder):
    if file.endswith('.mat'):
        try:
            filename_mat = os.path.join(input_folder, file)
            complex_raw_data, parameters = prdf.read_mat(filename_mat, device=device)
            processed_data, max_val_abs = prdf.process_complex_data(
                complex_raw_data, int(parameters["NF"]), device=device)

            # === Bilddaten berechnen ===
            image_depth = torch.flipud(
                torch.pow(torch.abs(processed_data[depth_layer, ...]), 2) / (max_val_abs + 1e-8)
            ).transpose(0, 1).detach().cpu().numpy()

            log_image = 10 * np.log10(image_depth + 1e-10)
            base_name = os.path.splitext(os.path.basename(filename_mat))[0]

            # === Dateipfade ===
            out_tiff_linear = os.path.join(output_folder, f'z_{base_name}_{depth_layer}.tiff')
            out_tiff_log = os.path.join(output_folder, f'log_z_{base_name}_{depth_layer}.tiff')
            out_colormap_linear = os.path.join(output_folder, f'z_{base_name}_{depth_layer}_colormap.jpg')
            out_colormap_log = os.path.join(output_folder, f'log_z_{base_name}_{depth_layer}_colormap.jpg')

            # === Speichern ===
            imageio.imwrite(out_tiff_linear, image_depth.astype(np.float32))
            imageio.imwrite(out_tiff_log, log_image.astype(np.float32))
            save_colormap_image(image_depth, out_colormap_linear)
            save_colormap_image(log_image, out_colormap_log)

            print(f"[OK] Gespeichert: {base_name} (4 Bilder)")

        except Exception as e:
            print(f"[ERROR] Fehler beim Verarbeiten von {file}: {e}")
            continue
