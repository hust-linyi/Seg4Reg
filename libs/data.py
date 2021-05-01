import os
import torch
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
import cv2
from libs.black_list import black_list


def resize_and_padding(img, size, label):
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
        label = (label * new_w + pad_l) / size[1]

    else:
        crop_l = (new_w - size[1]) // 2
        img_crop = img[:, crop_l:crop_l + size[1], :]
        label = (label * new_w - crop_l) / size[1]
        return img_crop, label
    return img_pad, label


def read_images(self, data_dir, nameFile, phase):
    """
    """
    dataNames = pd.read_csv(os.path.join(data_dir, "labels", phase, nameFile), header=None).values

    dataImages = []
    for i in range(dataNames.shape[0]):
        scan = cv2.imread(os.path.join(data_dir, "data", phase, dataNames[i][0]))
        # H, W = scan.shape[0], scan.shape[1]
        label = self.labels[i][:68]
        scan, label = resize_and_padding(scan, size=self.size, label=label)
        self.labels[i][:68] = label
#        seg = cv2.imread(os.path.join(data_dir, "seg", phase, dataNames[i][0]))
#        seg = np.max(seg, 2)

#        seg = cv2.imread(os.path.join('', phase, dataNames[i][0]), 0)
#        if seg is None:
#            seg = cv2.imread(os.path.join(data_dir, "seg", phase, dataNames[i][0]))
#            seg = np.max(seg, 2)

#        seg = cv2.resize(seg, (scan.shape[1], scan.shape[0]))
#        scan = np.tile(seg[:,:,np.newaxis], [1,1,3])
#        scan = np.concatenate(
#            [scan[:, :, 0][:, :, np.newaxis], seg[:, :, np.newaxis], np.zeros_like(seg[:, :, np.newaxis])], 2)
        scan = np.transpose(scan, [2,0,1])
        dataImages.append(scan)


    dataImages = np.stack(dataImages, axis=0)
    return dataImages


def _putGaussianMap(center, crop_size_y, crop_size_x, stride, sigma):
    """
    :param center:
    :return:
    """
    grid_y = crop_size_y // stride
    grid_x = crop_size_x // stride
    start = stride / 2.0 - 0.5
    y_range = [i for i in range(grid_y)]
    x_range = [i for i in range(grid_x)]
    xx, yy = np.meshgrid(x_range, y_range)
    # label is already normalized
    xx = (xx * stride + start) / crop_size_x
    yy = (yy * stride + start) / crop_size_y
    d2 = (xx - center[0]) ** 2 + (yy - center[1]) ** 2
    exponent = d2 / 2.0 / sigma / sigma
    heatmap = np.exp(-exponent)
    y = center[1]
    x = center[0]
    if y>=grid_y:
        y= grid_y-1
    if x>= grid_x:
        x= grid_x-1
    heatmap[int(y), int(x)] = 1
    return heatmap

def _putGaussianMaps(keypoints, crop_size_y, crop_size_x, stride, sigma):
    """
    :param keypoints: (15,2)
    :param crop_size_y: int
    :param crop_size_x: int
    :param stride: int
    :param sigma: float
    :return:
    """
    all_keypoints = keypoints.copy()
    point_num = int(all_keypoints.shape[0] / 2)
    heatmaps_this_img = []
    for c in range(4):
        heatmap_cornor = []
        for b in range(17):
            heatmap = _putGaussianMap([all_keypoints[4*b+c], all_keypoints[4*b+c + 68]], crop_size_y, crop_size_x, stride, sigma)
            heatmap = heatmap[np.newaxis, ...]
            heatmap_cornor.append(heatmap)
        heatmap_cornor = np.concatenate(heatmap_cornor, axis=0)
        heatmap_cornor = np.max(heatmap_cornor, 0, keepdims=True)
        heatmaps_this_img.append(heatmap_cornor)
    heatmaps_this_img = np.concatenate(heatmaps_this_img, 0)

    return heatmaps_this_img.astype(np.float32)


class ICUDataset(Dataset):

    def __init__(self, imageFile='filenames.csv', annotatFile='landmarks.csv',
                 phase='training', data_dir="E:/data/boostnet_labeldata/", size=1024, transform=None):
        super(ICUDataset, self).__init__()
        self.size = size

        self.labels = pd.read_csv(os.path.join(data_dir, 'labels', phase, annotatFile), header=None).values

        self.angles = pd.read_csv(os.path.join(data_dir, 'labels', phase, 'angles.csv'), header=None).values
#        self.angles = pd.read_csv(os.path.join(data_dir, 'labels', phase, 'angles_lin.csv'), header=None).values
#        if phase == 'test':
#            self.angles = pd.read_csv(os.path.join(data_dir, 'labels', phase, 'angles.csv'), header=None).values

        self.dataImages = read_images(self, data_dir, imageFile, phase)

        self.transform = transform

        filenames = pd.read_csv(os.path.join(data_dir, 'labels', phase, imageFile), header=None).values

        clean_labels, clean_data, clean_angles = [], [], []
        for f in range(len(filenames)):
            if not filenames[f][0][:-4] in black_list[phase]:
                clean_labels.append(self.labels[f])
                clean_data.append(self.dataImages[f])
                clean_angles.append(self.angles[f])

        self.dataImages = np.stack(clean_data, axis=0)
        self.labels = np.stack(clean_labels, axis=0)
        self.angles = np.stack(clean_angles, axis=0)

        print()

    def __len__(self):
        return self.labels.shape[0]

    def __getitem__(self, idx):
        sample = [self.dataImages[idx].copy(), self.labels[idx].copy(), self.angles[idx].copy()]
        if self.transform:
            sample = self.transform(sample)
        return sample


class GenerateHeatmap(object):
    def __init__(self, size, stride, sigma):
        super(GenerateHeatmap, self).__init__()
        self.size = size
        self.stride = stride
        self.sigma = sigma

    def __call__(self, sample):
        heatmap = _putGaussianMaps(keypoints=sample[1], crop_size_x=self.size[1],
                                   crop_size_y=self.size[0], stride=self.stride, sigma=self.sigma)
        return sample[0], heatmap, sample[1], sample[2]


class ToTensor(object):
    """
    label: 0-th index is nb_spot,others are location of spots (X,Y)
    """

    def __call__(self, sample):
        image, heatmap, label, angle = sample
        angle = angle / 90.
        label = np.append(label, angle).astype(np.float32)
        image = image.astype(np.float32) / 255.

        debug = False
        if debug:
            test_img = image.copy()
            test_img = test_img.transpose(1, 2, 0).copy()
            test_img = cv2.resize(test_img, (test_img.shape[1]//4,test_img.shape[0]//4))
            test_img1 = test_img.copy()
            test_img2 = test_img.copy()
            test_img3 = test_img.copy()
            test_img4 = test_img.copy()

            test_img1[:,:,1] += heatmap[0]
            test_img2[:, :, 1] += heatmap[1]
            test_img3[:, :, 1] += heatmap[2]
            test_img4[:, :, 1] += heatmap[3]
            test_img1 = cv2.resize(test_img1, (256,512))
            test_img2 = cv2.resize(test_img2, (256,512))
            test_img3 = cv2.resize(test_img3, (256,512))
            test_img4 = cv2.resize(test_img4, (256,512))

            cv2.imshow('test_img1', (test_img1 * 255).astype(np.uint8))
            cv2.imshow('test_img2', (test_img2 * 255).astype(np.uint8))
            cv2.imshow('test_img3', (test_img3 * 255).astype(np.uint8))
            cv2.imshow('test_img4', (test_img4 * 255).astype(np.uint8))

            cv2.imshow('test_img', cv2.resize(test_img, (256,512)))
            cv2.waitKey(0)

        return {'image': torch.from_numpy(image.copy()),
                'heatmap': torch.from_numpy(heatmap),
                'label': torch.from_numpy(label)}
