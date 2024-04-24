import pandas as pd
import numpy as np
import cv2 as cv
import torch
import torchvision
from torch.utils.data import Dataset
# from sklearn.preprocessing import LabelEncoder

class getData(Dataset):
    def __init__(self, f='train.csv'):
        # super(getData).__init__()
        self.src, self.trg = [], []

        data = pd.read_csv(f)
        for row in range(len(data)):
            # print("xmin = ",data['xmin'][row],", label = ", data['name'][row])
            box = [data['xmin'][row], data['ymin'][row], data['xmax'][row], data['ymax'][row]]
            lab = [data['name'][row]]
            # print(type(lab[0]))

            lab = list(map(self.mapping, lab))
            img = cv.imread(data['filename'][row])
            img = (img - np.min(img))/np.ptp(img)
            self.src.append(img)
            self.trg.append([box, lab])

        # self.trg = np.array(self.trg)
        # print(self.trg.shape)
    def mapping(self, x):
        if x == 1000:
            return 0
        elif x == 2000:
            return 1
        elif x == 5000:
            return 2
        elif x == 10000:
            return 3
        elif x == 20000:
            return 4
        elif x == 50000:
            return 5
        elif x == 75000:
            return 6
        elif x == 100000:
            return 7

    def __getitem__(self, index):
        # boxes = {'boxes':[], 'labels':[]}
        # trg['boxes'] = torch.tensor(self.trg['boxes'][index], dtype=torch.float32)
        # # print(boxes.shape)
        # # .reshape(-1, 4)
        # trg['labels'] = torch.tensor(self.trg['labels'][index], dtype=torch.int64)
        return torch.tensor(self.src[index], dtype=torch.float32), self.trg[index]

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
        src, trg = [],[]
        data = pd.read_csv(d)
        for row in range(len(data)):
            img = cv.imread(data['filename'][row])
            img = (img - np.min(img))/np.ptp(img)
            boxes = torch.tensor([data['xmin'][row], data['ymin'][row], data['xmax'][row], data['ymax'][row]])
            labels = torch.tensor(data['name'][row])
            src.append(img)
            trg.append([boxes, labels])

        print(np.array(src).shape)

        return src, trg

