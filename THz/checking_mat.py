
import sys
sys.path.insert(1, "src_code/")
import process_rdf as prdf
import torch
import os

folders = [r'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/test', r'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/thz_dataset/train']


for folder in folders:
    error = []
    print(f"Processing folder: {folder}")
    print(f"Number of files: {len([f for f in os.listdir(folder) if f.endswith('.mat')])}")
    for root, fol, files in os.walk(folder):
        for file in files:
            if file.endswith(".mat"):
                try:
                    filename_mat = os.path.join(root, file)
                    device = 'cpu'
                    complex_raw_data, parameters = prdf.read_mat(filename_mat, device=device)
                except Exception as e:
                    error.append((filename_mat, str(e)))

    print("Errors found:")
    print(f"Number of Errorfiles: {len(error)}")
    