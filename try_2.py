import torch
import torchvision
import torchvision.transforms as transforms
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
# from sklearn.metrics import accuracy_score
from torchvision.models.detection import FasterRCNN_ResNet50_FPN_Weights
from utils.data_2 import getData
import torch.nn.functional as F


def getModel(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(weights=FasterRCNN_ResNet50_FPN_Weights.DEFAULT)
    # model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = nn.Sequential(
        nn.Linear(in_features, num_classes + 1),
        nn.Softmax(dim=1)
    )
    return model
def main():
    BATCH_SIZE = 16
    EPOCH = 100
    LEARNING_RATE = 0.005
    MOMENTUM = 0.9
    WEIGHT_DECAY = 0.0005

    train_loader = DataLoader(getData(f='./train.csv'), batch_size=BATCH_SIZE, shuffle=True)
    test_loader = DataLoader(getData(f='./test.csv'), batch_size=BATCH_SIZE, shuffle=False)



    model = getModel(num_classes=8)
    optimizer = torch.optim.SGD(model.parameters(), lr=LEARNING_RATE, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    # for epoch in range(EPOCH):
    #     model.train()
    #     for batch, (src, trg) in enumerate(train_loader):
    #         targets = [{k: v[batch] for k, v in t.items()} for t in zip(trg, src.shape[0] * [trg[0]])]
    #
    #         loss_dict = model(src, targets=targets)
    #
    #         losses = sum(loss for loss in loss_dict.values())
    #         losses.backward()
    #         optimizer.step()
    #         optimizer.zero_grad()
    #
    #     print('train loss :', losses.item())

    for epoch in range(EPOCH):
        model.train()
        for data in train_loader:
            print(data)
            imgs = []
            target = []
            for d in data:
                imgs.append(d[0])
                targ = {}
                targ["boxes"] = d[1]["boxes"]
                targ["labels"] = d[1]["labels"]
                target.append(targ)

            loss_dict = model(src, targets)
            losses = sum(loss for loss in loss_dict.values())
            losses.backward()
            optimizer.step()
            optimizer.zero_grad()

            print('train loss :', losses.item())

        # for batch, (src, trg) in enumerate(train_loader):
        #     src = torch.permute(src, (0, 3, 1, 2))
        #     targets = []
        #     for i in range(len(trg["boxes"])):
        #         d = {}
        #         d["boxes"] = torch.tensor(trg["boxes"][i])
        #         d["labels"] = torch.tensor(trg["labels"][i])
        #         targets.append(d)
        #     loss_dict = model(src, targets)
        #
        #     print(loss_dict)
        #
        #     losses = sum(loss for loss in loss_dict.values())
        #     losses.backward()
        #     optimizer.step()
        #     optimizer.zero_grad()
        #
        #     print('train loss :', losses.item())

        model.eval()
        for batch, (src, trg) in enumerate(test_loader):
            targets = [{'boxes': trg[i]['boxes'], 'labels': trg[i]['labels']} for i in range(len(trg))]

            with torch.no_grad():
                loss_dict = model(src, targets)

            losses = sum(loss for loss in loss_dict.values())
            print('test loss :', losses.item())



if __name__ == "__main__":
    main()
