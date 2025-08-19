import json
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from process_rdf import process_complex_data, read_mat
import tifffile as tiff



class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(r'/pfs/work9/workspace/scratch/ma_lilipper-lippert_bachelorthesis_ws/Bachelorarbeit/train_prompt_one_target.json', 'r') as f:
            json_data = json.load(f)
            for line in json_data.get("prompts", []):
                if not line:  # <- überspringt leere Zeilen
                    continue
                self.data.append(line)

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        data_complex_all, parameters = read_mat(source_filename)
        source, max_val = process_complex_data(data_complex_all, int(parameters["NF"]), device="cpu")
        vol = torch.abs(source) ** 2
        vol = vol / (max_val + 1e-12)   # [T,H,W], float32

        # 4) Flip entlang Höhe (dim=1) – entspricht torch.flipud
        vol = torch.flip(vol, dims=[1])     # [T,H,W]

        # 5) In Form [B,C,T,H,W] bringen
        source = vol.unsqueeze(0).contiguous().float()
        target = cv2.imread(target_filename, cv2.IMREAD_UNCHANGED)

        # Sicherstellen, dass Bild geladen wurde
        if target is None:
            raise ValueError(f"Konnte Datei {target_filename} nicht laden.")

        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)


dataset = MyDataset()
sample = dataset[0]
print(f"length: {len(dataset)}")
print("Prompt:", sample["txt"])
print("Hint shape:", sample["hint"].shape)
print("Image shape:", sample["jpg"].shape)