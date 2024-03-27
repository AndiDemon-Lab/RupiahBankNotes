import pandas as pd
import numpy as np
import cv2 as cv
import torch
import torchvision
from torch.utils.data import Dataset
# from sklearn.preprocessing import LabelEncoder

class getData(Dataset):
    def __init__(self, f='train.csv'):
        super(getData).__init__()
        self.src, self.trg = [], []
        self.getCSV(f)

    def __getitem__(self, index):
        img_tensor = torch.tensor(self.src[index], dtype=torch.float32)
        boxes = torch.tensor(self.trg[index][0]).clone().detach()
        print(boxes.shape)
        # .reshape(-1, 4)
        labels = torch.tensor(self.trg[index][1], dtype=torch.int64)
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        return img_tensor, target

    def __len__(self):
        return len(self.trg)

    # def getCSV(self, d):
    #     data = pd.read_csv(d)
    #     targets = []
    #     for row in range(len(data)):
    #         xmin, ymin, xmax, ymax = data['xmin'][row], data['ymin'][row], data['xmax'][row], data['ymax'][row]
    #         img = cv.imread(data['filename'][row])
    #         img = cv.cvtColor(img, cv.COLOR_BGR2RGB)
    #         height, width, _ = img.shape
    #         boxes = torch.tensor([xmin, ymin, xmax, ymax], dtype=torch.float32)
    #         labels = torch.tensor(data['name'][row], dtype=torch.int64)
    #         targets.append({'boxes': boxes / torch.tensor([width, height, width, height], dtype=torch.float32),
    #                         'labels': labels})
    #     self.trg = targets
    def getCSV(self, d):
        data = pd.read_csv(d)
        for row in range(len(data)):
            img = cv.imread(data['filename'][row])
            boxes = torch.tensor([data['xmin'][row], data['ymin'][row], data['xmax'][row], data['ymax'][row]])
            labels = torch.tensor(data['name'][row])
            self.src.append(img)
            self.trg.append((boxes, labels))

