import pandas as pd
import numpy as np
import cv2 as cv
import torch
from torch.utils.data import Dataset

class getData(Dataset):
    def __init__(self, f='train.csv'):
        super(getData).__init__()
        self.src, self.trg = [], []
        self.getCSV(f)
        # self.src, self.trg = np.array(self.src), np.array(self.trg)

    def __getitem__(self, index):
        vec, label = self.src[index], self.trg[index]
        return torch.tensor(vec, dtype=torch.float32), torch.tensor(label, dtype=torch.float32)

    def __len__(self):
        return len(self.trg)

    def getCSV(self, d):
        data = pd.read_csv(d)
        for row in range(len(data)):
            # print(data['name'][row])
            self.src.append(cv.imread(data['filename'][row]))
            self.trg.append([data['xmin'][row], data['ymin'][row], data['xmax'][row], data['ymax'][row], data['name'][row]])

            # return np.array(src), np.array(trg)
