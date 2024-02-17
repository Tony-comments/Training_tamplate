
from torch import nn, optim
import torch
from torch.optim.lr_scheduler import CosineAnnealingLR
import Ranger
from tqdm import tqdm

def train(net, loss, train_dataloader, valid_dataloader, device, batch_size, num_epoch, lr, lr_min, optim='sgd', init=True, scheduler_type='Cosine'):
    def init_xavier(m): #参数初始化
        #if type(m) == nn.Linear or type(m) == nn.Conv2d:
        if type(m) == nn.Linear:
            nn.init.xavier_normal_(m.weight)

    if init:
        net.apply(init_xavier)

    print('training on:', device)
    net.to(device)
    #优化器选择
    if optim == 'sgd': 
        optimizer = torch.optim.SGD((param for param in net.parameters() if param.requires_grad), lr=lr,
                                    weight_decay=0)
    elif optim == 'adam':
        optimizer = torch.optim.Adam((param for param in net.parameters() if param.requires_grad), lr=lr,
                                     weight_decay=0)
    elif optim == 'adamW':
        optimizer = torch.optim.AdamW((param for param in net.parameters() if param.requires_grad), lr=lr,
                                      weight_decay=0)
    elif optim == 'ranger':
        optimizer = Ranger((param for param in net.parameters() if param.requires_grad), lr=lr,
                           weight_decay=0)
    if scheduler_type == 'Cosine':
        scheduler = CosineAnnealingLR(optimizer, T_max=num_epoch, eta_min=lr_min)
    #用来保存每个epoch的Loss和acc以便最后画图
    train_losses = []
    train_acces = []
    eval_acces = []
    best_acc = 0.0
    #训练
    for epoch in range(num_epoch):

        print("——————第 {} 轮训练开始——————".format(epoch + 1))

        # 训练开始
        net.train()
        train_acc = 0
        for batch in tqdm(train_dataloader, desc='训练'):
            imgs, targets = batch
            imgs = imgs.to(device)
            targets = targets.to(device)
            output = net(imgs)

            Loss = loss(output, targets)
          
            optimizer.zero_grad()
            Loss.backward()
            optimizer.step()

            _, pred = output.max(1)
            num_correct = (pred == targets).sum().item()
            acc = num_correct / (batch_size)
            train_acc += acc
        scheduler.step()
        print("epoch: {}, Loss: {}, Acc: {}".format(epoch, Loss.item(), train_acc / len(train_dataloader)))
        train_acces.append(train_acc / len(train_dataloader))
        train_losses.append(Loss.item())

        # 测试步骤开始
        net.eval()
        eval_loss = 0
        eval_acc = 0
        with torch.no_grad():
            for imgs, targets in valid_dataloader:
                imgs = imgs.to(device)
                targets = targets.to(device)
                output = net(imgs)
                Loss = loss(output, targets)
                _, pred = output.max(1)
                num_correct = (pred == targets).sum().item()
                eval_loss += Loss
                acc = num_correct / imgs.shape[0]
                eval_acc += acc

            eval_losses = eval_loss / (len(valid_dataloader))
            eval_acc = eval_acc / (len(valid_dataloader))
            if eval_acc > best_acc:
                best_acc = eval_acc
                torch.save(net.state_dict(),'best_acc.pth')
            eval_acces.append(eval_acc)
            print("整体验证集上的Loss: {}".format(eval_losses))
            print("整体验证集上的正确率: {}".format(eval_acc))
    return train_losses, train_acces, eval_acces