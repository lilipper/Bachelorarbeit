
import os
import sys

import skimage
sys.path.insert(1, "src_code/")
import process_rdf as prdf
import torch

for scan in os.listdir("D:/700GHz/mat_files/train/"):
    if scan.endswith(".mat"):
        filename_mat = os.path.join("D:/700GHz/mat_files/train/", scan)
        device = 'cpu'
        complex_raw_data, parameters = prdf.read_mat(filename_mat, device=device)
        processed_data, max_val_abs = prdf.process_complex_data(complex_raw_data, int(parameters["NF"]), device=device)
        depth_layer = 700
        image_depth = torch.flipud(torch.pow(torch.abs(processed_data[depth_layer, ...]), 2)/max_val_abs).transpose(0, 1).detach().cpu().numpy()
        skimage.io.imsave('THz/USAF_images/tiff/image_z_' + os.path.basename(filename_mat).split('.')[0] + str(depth_layer) + '.tiff', image_depth, plugin="tifffile", check_contrast=False)
