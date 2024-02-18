import torch
from torch import nn
from torchmetrics import Accuracy
import torchvision
from torch.utils.data import DataLoader
from utils.utils_chihuo import train_model
from utils.utils_chihuo import get_dataloader

net = torchvision.models.resnet50(pretrained=True)
loss_fn = nn.BCEWithLogitsLoss()
optimizer= torch.optim.Adam(net.parameters(),lr = 0.01)   
metrics_dict = {"acc":Accuracy()}
train_dataloader, test_dataloader = get_dataloader(batch_size=64)

dfhistory = train_model(net,
    optimizer,
    loss_fn,
    metrics_dict,
    train_data = train_dataloader,
    val_data= test_dataloader,
    epochs=10,
    patience=5,
    monitor="val_acc", 
    mode="max")
