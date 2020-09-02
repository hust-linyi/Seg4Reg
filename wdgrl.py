"""
Implements RevGrad:
Unsupervised Domain Adaptation by Backpropagation, Ganin & Lemptsky (2014)
Domain-adversarial training of neural networks, Ganin et al. (2016)
"""
import argparse

import torch
from torch import nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
from torchvision.datasets import MNIST
from torchvision.transforms import Compose, ToTensor
from tqdm import tqdm
from torch.autograd import Function
import cv2
import numpy as np
import pandas as pd
import os
import extramodel.densenet as densenet
import extramodel.resnet as resnet
from torchvision import transforms
from libs.black_list import black_list
from torch.autograd import grad

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def resize_and_padding(img, size):
    h, w = img.shape[0], img.shape[1]
    new_h, new_w = size[0], int(size[0] * w / h)
    img = cv2.resize(img, (new_w, new_h))
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    img = clahe.apply(img[:,:,0])
    img = np.tile(img[:,:,np.newaxis], [1,1,3])

    if size[1] > new_w:
        pad_l = (size[1] - new_w) // 2
        pad_r = size[1] - pad_l - new_w
        img_pad = np.pad(img, ((0, 0), (pad_l, pad_r), (0, 0)), 'constant')

    else:
        crop_l = (new_w - size[1]) // 2
        img_crop = img[:, crop_l:crop_l + size[1], :]
        return img_crop
    return img_pad


def read_images(self, data_dir, nameFile, phase):
    """
    """
    dataNames = pd.read_csv(os.path.join(data_dir, "labels", phase, nameFile), header=None).values

    dataImages = []
    for i in range(dataNames.shape[0]):
        if dataNames[i][0].endswith('.jpg'):
            dataname = dataNames[i][0]
        else:
            dataname = dataNames[i][0] + '.jpg'

        scan = cv2.imread(os.path.join(data_dir, "data", phase, dataname))
        scan = resize_and_padding(scan, size=self.size)
#        seg = cv2.imread(os.path.join(data_dir, "seg", phase, dataname))
        seg = cv2.imread(os.path.join(data_dir, "seg", phase + '_3c', dataname))
        seg = np.max(seg, 2)
        seg = cv2.resize(seg, (scan.shape[1], scan.shape[0]))
#        scan = np.concatenate(
#            [scan[:, :, 0][:, :, np.newaxis], seg[:, :, np.newaxis], np.zeros_like(seg[:, :, np.newaxis])], 2)
#         scan = np.tile(seg[:,:,np.newaxis], [1,1,3])
#         cv2.imshow('scan', scan)
#         cv2.waitKey(0)
        scan = np.transpose(scan, [2,0,1])
        dataImages.append(scan)
    dataImages = np.stack(dataImages, axis=0)
    return dataImages

class ToTensor(object):
    """
    label: 0-th index is nb_spot,others are location of spots (X,Y)
    """

    def __call__(self, sample):
        image = sample[0]
        image = image.astype(np.float32) / 255.
        debug = False
        if debug:
            image = (image * 255).astype(np.uint8)
            image = image.transpose(1,2,0)
            cv2.imshow('img', image)
            cv2.waitKey(0)
        return {'image': torch.from_numpy(image.copy())}

class AasceDataTest(Dataset):
    def __init__(self, imageFile='filenames.csv', annotatFile='landmarks.csv',
                 phase='training', data_dir="", size=1024, transform=None):
        super(AasceDataTest, self).__init__()
        self.size = size

        self.dataImages = read_images(self, data_dir, imageFile, phase)

        self.phase = phase

        self.filenames = pd.read_csv(os.path.join(data_dir, 'labels', phase, imageFile), header=None).values


        if phase == 'training':
            self.angles = pd.read_csv(os.path.join(data_dir, 'labels', phase, '.csv'), header=None).values
            self.cla_gt = pd.read_csv(os.path.join(data_dir, 'labels', phase, '.csv'), header=None).values

            clean_labels, clean_data, clean_angles = [], [], []
            clean_cla = []
            for f in range(len(self.filenames)):
                if not self.filenames[f][0][:-4] in black_list[phase]:
                    clean_data.append(self.dataImages[f])
                    clean_angles.append(self.angles[f])
                    clean_cla.append(self.cla_gt[f])

            self.dataImages = np.stack(clean_data, axis=0)
            self.angles = np.stack(clean_angles, axis=0)
            self.cla_gt = np.stack(clean_cla, axis=0)

        self.transform = transform


    def __len__(self):
        return self.dataImages.shape[0]

    def __getitem__(self, idx):
        sample = [self.dataImages[idx].copy()]
        if self.transform:
            sample = self.transform(sample)
        if self.phase == 'training':
            sample["angles"] = torch.from_numpy(self.angles[idx].copy() / 90.).float()
            sample["cla"] = torch.tensor(self.cla_gt[idx].copy()).long()
        return sample


def gradient_penalty(critic, h_s, h_t):
    # based on: https://github.com/caogang/wgan-gp/blob/master/gan_cifar10.py#L116
    alpha = torch.rand(h_s.size(0), 1).to(device)
    differences = h_t - h_s
    interpolates = h_s + (alpha * differences)
    interpolates = torch.stack([interpolates, h_s, h_t]).requires_grad_()

    preds = critic(interpolates)
    gradients = grad(preds, interpolates,
                     grad_outputs=torch.ones_like(preds),
                     retain_graph=True, create_graph=True)[0]
    gradient_norm = gradients.norm(2, dim=1)
    gradient_penalty = ((gradient_norm - 1)**2).mean()
    return gradient_penalty


def set_requires_grad(model, requires_grad=True):
    for param in model.parameters():
        param.requires_grad = requires_grad


def main(args):
    if args.net == "dense121":
        model = densenet.densenet121(pretrained=False, num_classes=3)
    if args.net == "dense161":
        model = densenet.densenet161(pretrained=False, num_classes=3)
    if args.net == "dense169":
        model = densenet.densenet169(pretrained=False, num_classes=3)
    if args.net == "dense201":
        model = densenet.densenet201(pretrained=False, num_classes=3)
    if args.net == "res18":
        model = resnet.resnet18(pretrained=False, num_classes=3)
    if args.net == "res50":
        model = resnet.resnet50(pretrained=False, num_classes=3)
    if args.net == "res101":
        model = resnet.resnet101(pretrained=False, num_classes=3)
    if args.net == "res152":
        model = resnet.resnet152(pretrained=False, num_classes=3)

    feature_extractor = model.features
    clf = model.classifier_lin

    if torch.cuda.is_available():
        model = model.cuda()
        model = torch.nn.DataParallel(model)

    # model.load_state_dict(torch.load(args.checkpoint).state_dict())

    discriminator = nn.Sequential(
        nn.Linear(1664, 512),
        nn.ReLU(),
        nn.Linear(512, 64),
        nn.ReLU(),
        nn.Linear(64, 1)
    ).to(device)

    half_batch = args.batch_size // 2
    kwargs = {'num_workers': 8, 'pin_memory': False} if torch.cuda.is_available() else {}

    source_dataset = AasceDataTest(data_dir=args.data_dir, phase='training', size=args.img_size,
                                transform=transforms.Compose([ToTensor()]))
    source_loader = DataLoader(source_dataset, batch_size=half_batch, shuffle=True, drop_last=True, **kwargs)

    target_dataset = AasceDataTest(data_dir=args.data_dir, phase='test', size=args.img_size,
                                transform=transforms.Compose([ToTensor()]))
    target_loader = DataLoader(target_dataset, batch_size=half_batch, shuffle=True, drop_last=True, **kwargs)

    critic_optim = torch.optim.Adam(discriminator.parameters(), lr=1e-4)
    clf_optim = torch.optim.Adam(model.parameters(), lr=1e-4)

    for epoch in range(1, args.epochs+1):
        batches = zip(source_loader, target_loader)
        n_batches = min(len(source_loader), len(target_loader))

        total_loss = 0
        for source_batch, target_batch in tqdm(batches, leave=False, total=n_batches):
            source_x, source_labels = source_batch['image'], source_batch['angles']
            target_x = target_batch['image']

            if torch.cuda.is_available():
                source_x, source_labels, target_x = source_x.cuda(), source_labels.cuda(), target_x.cuda()

            set_requires_grad(feature_extractor, requires_grad=False)
            set_requires_grad(discriminator, requires_grad=True)

            with torch.no_grad():
                h_s = feature_extractor(source_x)
                h_s = F.adaptive_avg_pool2d(h_s, (1, 1)).data.view(h_s.size(0), -1)
                h_t = feature_extractor(target_x)
                h_t = F.adaptive_avg_pool2d(h_t, (1, 1)).data.view(h_t.size(0), -1)

            for _ in range(args.k_critic):
                gp = gradient_penalty(discriminator, h_s, h_t)

                critic_s = discriminator(h_s)
                critic_t = discriminator(h_t)
                wasserstein_distance = critic_s.mean() - critic_t.mean()

                critic_cost = -wasserstein_distance + args.gamma*gp

                critic_optim.zero_grad()
                critic_cost.backward()
                critic_optim.step()

                total_loss += critic_cost.item()

            # Train classifier
            set_requires_grad(feature_extractor, requires_grad=True)
            set_requires_grad(discriminator, requires_grad=False)

            for _ in range(args.k_clf):
                source_features = feature_extractor(source_x)
                source_features = F.adaptive_avg_pool2d(source_features, (1, 1)).data.view(source_features.size(0), -1)
                target_features = feature_extractor(target_x)
                target_features = F.adaptive_avg_pool2d(target_features, (1, 1)).data.view(target_features.size(0), -1)

                source_preds = clf(source_features)
                source_preds = nn.Sigmoid()(source_preds)
                clf_loss = torch.mean(torch.abs(source_preds - source_labels) / (source_preds + source_labels))
                wasserstein_distance = discriminator(source_features).mean() - discriminator(target_features).mean()

                loss = clf_loss + args.wd_clf * wasserstein_distance
                clf_optim.zero_grad()
                loss.backward()
                clf_optim.step()

        mean_loss = total_loss / (args.iterations * args.k_critic)
        tqdm.write('EPOCH: %03d, domain_loss: %.4f' % (epoch, mean_loss))
        torch.save(model, os.path.join(os.getcwd(), args.save_path, "angle_domain_adaptation.pth"))


if __name__ == '__main__':
    arg_parser = argparse.ArgumentParser(description='Domain adaptation using RevGrad')
    arg_parser.add_argument('--checkpoint', type=str, default='')
    arg_parser.add_argument('--batch-size', type=int, default=16)
    arg_parser.add_argument('--epochs', type=int, default=15)
    arg_parser.add_argument('--data-dir', type=str, default='')
    # arg_parser.add_argument('--data-dir', type=str, default='/data/home/ianylin/data/aasce')
    arg_parser.add_argument('--img-size', type=list, default=[64, 32])
    arg_parser.add_argument('--net', type=str, default='dense169')
    arg_parser.add_argument('--save-path', type=str, default='logs')
    arg_parser.add_argument('--k-critic', type=int, default=5)
    arg_parser.add_argument('--k-clf', type=int, default=1)
    arg_parser.add_argument('--gamma', type=float, default=10)
    arg_parser.add_argument('--wd-clf', type=float, default=1)

    args = arg_parser.parse_args()
    main(args)
