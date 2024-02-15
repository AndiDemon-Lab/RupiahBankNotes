import numpy as np
import torch
import torchvision
import torch.nn as nn
from torch.utils.data import DataLoader

from utils.data import getData

#sklearn confusion matrix, accuracy, precision, recall, f1

def main():
    BATCH_SIZE = 16
    EPOCH = 100
    LEARNING_RATE = 0.005

    train_loader = DataLoader(getData(f='train.csv'), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(getData(f='test.csv'), batch_size=BATCH_SIZE, shuffle=True)

    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=0.9, weight_decay=0.0005)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

    loss_fn = nn.MSELoss()
    for epoch in range(EPOCH):
        model.train()
        for batch, (src, trg) in enumerate(train_loader):
            loss = 0
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)#.to(device)

            loss = loss_fn(pred, trg)

            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('train loss :', loss, 'train acc')

        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            loss = 0
            src = torch.permute(src, (0, 3, 1, 2))

            pred = model(src)  # .to(device)

            loss = loss_fn(pred, trg)


            print('test loss :', loss, 'train acc')


if __name__ == '__main__':
    main()