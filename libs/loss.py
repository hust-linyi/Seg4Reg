import torch
import torch.nn as nn
import torch.nn.functional as F


class JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss()
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)
        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()
            if self.use_target_weight:
                loss += 0.5 * self.criterion(
                    heatmap_pred.mul(target_weight[:, idx]),
                    heatmap_gt.mul(target_weight[:, idx])
                )
            else:
                loss += 0.5 * self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class Debug_JointsMSELoss(nn.Module):
    def __init__(self, use_target_weight=True):
        super(Debug_JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss(reduction='sum')
        self.use_target_weight = use_target_weight

    def forward(self, output, target, target_weight):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            target_weight = (heatmap_pred > 0).float()


            self.use_target_weight = True

            if self.use_target_weight:
                loss += self.criterion(
                    heatmap_pred.mul(target_weight),
                    heatmap_gt.mul(target_weight)) / torch.sum(target_weight)
            else:
                loss += self.criterion(heatmap_pred, heatmap_gt)

        return loss / num_joints


class Hard_mining_JointsMSELoss(nn.Module):
    def __init__(self):
        super(Hard_mining_JointsMSELoss, self).__init__()
        self.criterion = nn.MSELoss()

    def forward(self, output, target):
        batch_size = output.size(0)
        num_joints = output.size(1)
        heatmaps_pred = output.reshape((batch_size, num_joints, -1)).split(1, 1)
        heatmaps_gt = target.reshape((batch_size, num_joints, -1)).split(1, 1)

        loss = 0

        for idx in range(num_joints):
            heatmap_pred = heatmaps_pred[idx].squeeze()
            heatmap_gt = heatmaps_gt[idx].squeeze()

            # hard mining
            pos_idx = heatmap_gt > 0
            neg_idx = heatmap_gt == 0

            pos_pred = heatmap_pred[pos_idx]
            neg_pred = heatmap_pred[neg_idx]
            pos_gt = heatmap_gt[pos_idx]
            neg_gt = heatmap_gt[neg_idx]

            # hard mining
            if len(neg_pred):
                _, idcs = torch.topk(neg_pred, min(len(neg_pred), len(pos_pred), 1))
                neg_pred = neg_pred[idcs]
                neg_gt = neg_gt[idcs]
                loss += self.criterion(pos_pred, pos_gt)
                loss += self.criterion(neg_pred, neg_gt)
            else:
                loss += self.criterion(pos_pred, pos_gt)

        return loss / num_joints

class KpLoss(nn.Module):

    def __init__(self):
        super(KpLoss, self).__init__()

    def forward(self, preds, gt):
        pos_inds = gt.eq(1)
        neg_inds = gt.lt(1)

        neg_weights = torch.pow(1 - gt[neg_inds], 4)

        pos_pred = preds[pos_inds]
        neg_pred = preds[neg_inds]

        pos_loss = torch.log(pos_pred) * torch.pow(1 - pos_pred, 2)
        neg_loss = torch.log(1 - neg_pred) * torch.pow(neg_pred, 2) * neg_weights

        pos_loss = pos_loss.mean()
        neg_loss = neg_loss.mean()
        if pos_pred.nelement() == 0:
            loss = - neg_loss
        else:
            loss = -(pos_loss + neg_loss)
        return loss