#Import Library

import pandas as pd
import numpy as np
import cv2 as cv
import torch
import torchvision
from torch.utils.data import Dataset
import torchvision.transforms as T
from skimage.feature import hog

class getData(Dataset):
    def __init__(self, f='train.csv'):
        self.src, self.trg, self.original = [], [], []

        self.transform = T.Compose([
            T.ToPILImage(),
            T.ToTensor()
        ])

        data = pd.read_csv(f)
        for row in range(len(data)):
            box = [data['xmin'][row], data['ymin'][row], data['xmax'][row], data['ymax'][row]]
            lab = [data['name'][row]]

            lab = list(map(self.mapping, lab))
            img = cv.imread(data['filename'][row])
            self.original.append(img.copy())
            img = (img - np.min(img))/np.ptp(img)

            # self.original.append(img.copy())
            # Convert image to uint8 (8-bit) before converting to HSV
            img = (img * 255).astype(np.uint8)

            # img_gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
            # img_hog = self.extract_hog_features(img_gray)
            # img_hog = np.stack([img_hog] * 3, axis=-1)


            # Convert image to HSV
            img = cv.cvtColor(img, cv.COLOR_BGR2HSV)

            # img = img + img_hog
            #
            # # Extract HOG features from the V channel
            # hog_features = self.extract_hog_features(img_hsv[:, :, 2])  # V channel
            #
            # # Reshape HOG features to match original image size
            # hog_image = hog_features.reshape(img_hsv[:, :, 2].shape)
            #
            # # Stack HOG with H and S channels to form a 3-channel image
            # combined_image = np.stack([img_hsv[:, :, 0], img_hsv[:, :, 1], hog_image], axis=-1)

            self.src.append(img)
            self.trg.append([box, lab])

    def extract_hog_features(self, img):
        # Use skimage's hog function to extract HOG features
        features, hog_image = hog(img, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True)
        return hog_image

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

    def __len__(self):
        return len(self.trg)

    def __getitem__(self, index):
        img = self.src[index]
        img = self.transform(img)
        return torch.tensor(img, dtype=torch.float32), self.trg[index]
