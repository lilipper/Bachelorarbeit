import os
import scipy.io
import numpy as np
import torch
import imageio
from skimage.io import imsave
import matplotlib.pyplot as plt
from matplotlib import cm


# === Parameter ===
input_folder = input('Pfad zur .mat-Datei: ').strip()
output_folder = os.path.join(input_folder,'USAF_images')
os.makedirs(output_folder, exist_ok=True)

depth_layer = 701

for file in os.listdir(input_folder):
    if file.endswith('.mat'):
        try:
            input_file = os.path.join(input_folder, file)
            # === .mat-Datei laden ===
            data = scipy.io.loadmat(input_file)
            if 'data_complex_all_1' not in data:
                raise KeyError("'data_complex_all_1' nicht in .mat-Datei enthalten.")

            volume_np = data['data_complex_all_1']  # z. B. (1400, 292, 90)
            volume = torch.from_numpy(volume_np)

            if depth_layer >= volume.shape[0]:
                raise IndexError(f"Layer {depth_layer} existiert nicht (max {volume.shape[0]-1}).")

            # === Verarbeitung ===
            slice_complex = volume[depth_layer, ...]
            power_image = torch.abs(slice_complex) ** 2

            max_val_abs = torch.max(torch.abs(volume)) ** 2
            image_depth = power_image / (max_val_abs + 1e-8)

            # Flip & Transpose
            image_depth = torch.flipud(image_depth).transpose(0, 1)

            # NumPy konvertieren
            image_np = image_depth.detach().cpu().numpy()
            log_image_np = 10 * np.log10(image_np + 1e-10)

            p_low, p_high = np.percentile(log_image_np, [1, 99])
            log_stretched = np.clip(log_image_np, p_low, p_high)
            log_normalized = (log_stretched - p_low) / (p_high - p_low + 1e-8)
            image_for_colormap = image_np / (np.max(image_np) + 1e-8)

            # Matplotlib-Colormap anwenden (z. B. viridis)
            colored = cm.viridis(image_for_colormap)[:, :, :3]  # Nur RGB (Alpha weglassen)
            colored_uint8 = (colored * 255).astype(np.uint8)

            colored_norm = cm.viridis(log_normalized)[:, :, :3]  # nur RGB
            colored_uint8_norm = (colored_norm * 255).astype(np.uint8)


            # === Dateinamen vorbereiten ===
            base_name = os.path.splitext(os.path.basename(input_file))[0]
            out_linear_jpg = os.path.join(output_folder, f'z_{base_name}_{depth_layer}.jpg')
            out_linear_tiff = os.path.join(output_folder, f'z_{base_name}_{depth_layer}.tiff')
            out_log_jpg = os.path.join(output_folder, f'log_z_{base_name}_{depth_layer}.jpg')
            out_log_tiff = os.path.join(output_folder, f'log_z_{base_name}_{depth_layer}.tiff')
            colored_path = os.path.join(output_folder, f'z_colormap_{base_name}_{depth_layer}.jpg')
            colored_norm_path = os.path.join(output_folder, f'z_colormap_norm_{base_name}_{depth_layer}.jpg')
            
            # === Speichern als RGB-JPG ===
            imageio.imwrite(colored_norm_path, colored_uint8_norm)
            imageio.imwrite(colored_path, colored_uint8)
            # === Speichern: linear ===
            image_uint8 = (255 * image_np / np.max(image_np)).astype(np.uint8)
            imageio.imwrite(out_linear_jpg, image_uint8)
            imageio.imwrite(out_linear_tiff, image_np.astype(np.float32))

            # === Speichern: log ===
            log_scaled = (255 * (log_image_np - np.min(log_image_np)) / (np.ptp(log_image_np) + 1e-8)).astype(np.uint8)
            imageio.imwrite(out_log_jpg, log_scaled)
            imageio.imwrite(out_log_tiff, log_image_np.astype(np.float32))

            print(f"[OK] Gespeichert:\n  {out_linear_jpg}\n  {out_linear_tiff}\n  {out_log_jpg}\n  {out_log_tiff}")

        except Exception as e:
            print(f"[FEHLER] Datei konnte nicht verarbeitet werden: {input_file}")
            print(f"[INFO] {type(e).__name__}: {e}")
    