import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from utils.data import getData

import torch
import torch.nn.functional as F
from torchvision.ops import box_iou

# def calculate_mAP(pred, trg, iou_threshold=0.3):
#     # Calculate IoU between predicted boxes and target boxes
#     iou = box_iou(pred, targets)
#
#     # Calculate precision and recall
#     precision = []
#     recall = []
#     for i in range(len(pred)):
#         true_positives = (iou[i] >= iou_threshold).sum().item()
#         false_positives = len(pred) - true_positives
#         false_negatives = len(trg) - true_positives
#
#         precision.append(true_positives / (true_positives + false_positives))
#         recall.append(true_positives / (true_positives + false_negatives))
#
#     # Calculate average precision
#     ap = torch.tensor(precision).mean()
#
#     return ap

def main():
    BATCH_SIZE = 32
    EPOCH = 100
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005
    STEP_SIZE = 3
    GAMMA = 0.1

    train_loader = DataLoader(getData(f='train.csv'), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(getData(f='test.csv'), batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=STEP_SIZE, gamma=GAMMA)

    # mAP = calculate_mAP

    loss_fn = nn.MSELoss() #use mAP(?)
    for epoch in range(EPOCH):
        for batch, (src, trg) in enumerate(train_loader):
            print(trg.shape)
            loss = 0
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)

            loss = loss_fn(pred, trg)
            accuracy = accuracy_score(pred, trg)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('train loss :', loss, 'train acc', accuracy)

        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            loss = 0
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)  # .to(device)

            loss = loss_fn(pred, trg)
            accuracy = accuracy_score(pred, trg)

            print('test loss :', loss, 'test acc', accuracy)


if __name__ == '__main__':
    main()