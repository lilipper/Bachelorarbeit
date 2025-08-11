import json
import cv2
import numpy as np

from torch.utils.data import Dataset
from process_rdf import process_complex_data, read_mat



class MyDataset(Dataset):
    def __init__(self):
        self.data = []
        with open(r'.\thz_dataset\train\prompt.json', 'rt') as f:
            for line in f:
                self.data.append(json.loads(line))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        item = self.data[idx]

        source_filename = item['source']
        target_filename = item['target']
        prompt = item['prompt']

        data_complex_all, parameters = read_mat(source_filename)
        source, max_val = process_complex_data(data_complex_all, int(parameters["NF"]), device="cpu")
        target = cv2.imread(target_filename)

        # Do not forget that OpenCV read images in BGR order.
        # TODO: Nessecary?
        source = cv2.cvtColor(source, cv2.COLOR_BGR2RGB)
        target = cv2.cvtColor(target, cv2.COLOR_BGR2RGB)

        # Normalize source images to [0, 1].
        source = source.astype(np.float32) / 255.0

        # Normalize target images to [-1, 1].
        target = (target.astype(np.float32) / 127.5) - 1.0

        return dict(jpg=target, txt=prompt, hint=source)
