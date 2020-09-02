import os
from tqdm import tqdm
import torch
import argparse
import shutil
import numpy as np
import torch.nn.functional as F
from torchvision import transforms
from torch.utils.data import DataLoader
from libs.data import ICUDataset, ToTensor, GenerateHeatmap
from libs.loss import JointsMSELoss, Debug_JointsMSELoss, Hard_mining_JointsMSELoss, KpLoss
import extramodel.densenet as densenet
import extramodel.resnet as resnet
import copy
import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
import itertools
from libs.metrics import  compute_mse
import cv2


def train_epoch(net, epoch, dataLoader, optimizer, loss_list):
    net.train()
    dataprocess = tqdm(dataLoader)
    for batch_item in dataprocess:
        images, label = batch_item['image'], batch_item["label"][:, -3:]
        if torch.cuda.is_available():
            images, label = images.cuda(), label.cuda()

        optimizer.zero_grad()
        output = net(images)

        loss = torch.mean(torch.abs((label - output)))

        loss.backward()
        optimizer.step()
        dataprocess.set_description_str("epoch:{}".format(epoch))
        dataprocess.set_postfix_str("loss:{:.4f}".format(loss.item()))
        loss_list.append(loss.float())


def test(net, epoch, dataLoader, parser, test_results_list):
    net.eval()
    cor_abs_mean_error_list, angle_pred_list, angle_gt_list = [], [], []
    for batch_num, batch_item in enumerate(dataLoader):
        with torch.no_grad():
            images, label = batch_item['image'].cuda(), batch_item['label'][:,-3:].cuda()
            out = net(images)
            out = out.cpu().detach().numpy()
            label = label.cpu().detach().numpy()

            for i in range(out.shape[0]):
                angle_pred_list.append(out[i])
                angle_gt_list.append(label[i])

    angle_abs_mean_error = np.mean(np.abs(np.array(angle_pred_list) - np.array(angle_gt_list)), 0)
    print('Abs Error: %.4f, %.4f, %.4f' % (angle_abs_mean_error[0] * 90, angle_abs_mean_error[1] * 90, angle_abs_mean_error[2] * 90))

def adjust_lr(optimizer, epoch):
    if epoch == 0:
        return
    if (epoch % 30 == 0) & (epoch != 0):
        lr = optimizer.param_groups[0]['lr'] * 0.1
        for param_group in optimizer.param_groups:
            param_group['lr'] = lr


def main(parser):
    if os.path.exists(parser["save_path"]):
        shutil.rmtree(parser["save_path"])
    os.makedirs(parser["save_path"], exist_ok=True)

    if os.path.exists(parser["visual_dir"]):
        shutil.rmtree(parser["visual_dir"])
    os.makedirs(parser["visual_dir"], exist_ok=True)

    kwargs = {'num_workers': 8, 'pin_memory': False} if torch.cuda.is_available() else {}

    training_dataset = ICUDataset(data_dir=parser["data_dir"], size=parser["img_size"],
                                  transform=transforms.Compose([GenerateHeatmap(parser["img_size"], parser["stride"], parser["sigma"]),
                                                                ToTensor()]))
    training_data_batch = DataLoader(training_dataset, batch_size=parser["batch_size"],
                                     shuffle=True, drop_last=True, ** kwargs)
    val_dataset = ICUDataset(data_dir=parser["data_dir"], phase='test', size=parser["img_size"],
                             transform=transforms.Compose([GenerateHeatmap(parser["img_size"], parser["stride"], parser["sigma"]),
                                                           ToTensor()]))
    val_data_batch = DataLoader(val_dataset, batch_size=parser["test_batch_size"], ** kwargs)

    if parser["net"] == "dense121":
        net = densenet.densenet121(pretrained=True, num_classes=3)
    if parser["net"] == "dense161":
        net = densenet.densenet161(pretrained=True, num_classes=3)
    if parser["net"] == "dense169":
        net = densenet.densenet169(pretrained=True, num_classes=3)
    if parser["net"] == "dense201":
        net = densenet.densenet201(pretrained=True, num_classes=3)
    if parser["net"] == "res18":
        net = resnet.resnet18(pretrained=True, num_classes=3)
    if parser["net"] == "res50":
        net = resnet.resnet50(pretrained=True, num_classes=3)
    if parser["net"] == "res101":
        net = resnet.resnet101(pretrained=True, num_classes=3)
    if parser["net"] == "res152":
        net = resnet.resnet152(pretrained=True, num_classes=3)

    if torch.cuda.is_available():
        net = net.cuda()
        net = torch.nn.DataParallel(net)

    optimizer = torch.optim.SGD(net.parameters(), lr=parser["lr"], momentum=0.9, weight_decay=parser["weight_decay"])
#    optimizer = torch.optim.Adam(net.parameters(), lr=3e-3, weight_decay=parser["weight_decay"])

    loss_list, test_results_list = [], []
    for epoch in range(parser["epochs"]):
        adjust_lr(optimizer, epoch)
        train_epoch(net, epoch, training_data_batch, optimizer, loss_list)
        test(net, epoch, val_data_batch, parser, test_results_list)

    plt.plot(np.linspace(0, parser["epochs"], len(loss_list)), loss_list)
    plt.savefig(os.path.join('./', 'angle_training_loss.png'))
    torch.save(net, os.path.join(os.getcwd(), parser["save_path"], "angle_baseline.pth"))
    test_results = np.array(test_results_list)

    for k, v in parser.items():
        print('{}: {}'.format(k, v))
    print()
    test_results = np.mean(test_results[-10:, :], 0)
    print('min_abs_angle: %.4f, %.4f, %.4f' % (test_results[0]*90, test_results[1]*90, test_results[2]*90))


if __name__ == "__main__":
    os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
    platform = 'sever'
    parser = {}
    parser["epochs"] = 90
    parser["weight_decay"] = 1e-4
    parser["lr"] = 1e-2
    parser["save_path"] = "logs"
    parser["visual_dir"] = ""
    parser["img_size"] = [512, 256]
    parser["sigma"] = 0.01
    parser["stride"] = 4

    parser["data_dir"] = ""
    parser["batch_size"] = 32
    parser["test_batch_size"] = 32
    parser["net"] = "dense169"

    if platform == 'win':
        parser["data_dir"] = ""
        parser["net_growth_rate"] = 1
        parser["batch_size"] = 2
        parser["test_batch_size"] = 2
    main(parser)
