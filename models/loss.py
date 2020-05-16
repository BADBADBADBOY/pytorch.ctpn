"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: loss.py
@time: 2020/4/5 9:28

"""

import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
import numpy as np
import math
import pdb 
from torch.autograd import Variable

LossCL = nn.CrossEntropyLoss().cuda()

def smooth_l1_loss(inputs,
                   targets,
                   beta=1. / 9,
                   size_average=True):
    """
    https://github.com/facebookresearch/maskrcnn-benchmark
    """
    diff = torch.abs(inputs - targets)
    loss = torch.where(
        diff < beta,
        0.5 * diff ** 2 / beta,
        diff - 0.5 * beta
    )
    if size_average:
        return loss.mean()
    return loss.sum()


def balanced_l1_loss(inputs,
                     targets,
                     beta=1. / 9,
                     alpha=0.5,
                     gamma=1.5,
                     size_average=True):
    """Balanced L1 Loss

    arXiv: https://arxiv.org/pdf/1904.02701.pdf (CVPR 2019)
    """
    assert beta > 0
    assert inputs.size() == targets.size() and targets.numel() > 0

    diff = torch.abs(inputs - targets)
    b = np.e ** (gamma / alpha) - 1
    loss = torch.where(
        diff < beta, alpha / b *
        (b * diff + 1) * torch.log(b * diff / beta + 1) - alpha * diff,
        gamma * diff + gamma / b - alpha * beta)

    if size_average:
        return loss.mean()
    return loss.sum()

def hard_negative_mining(loss, labels, neg_pos_ratio):
    """
    It used to suppress the presence of a large number of negative prediction.
    It works on image level not batch level.
    For any example/image, it keeps all the positive predictions and
     cut the number of negative predictions to make sure the ratio
     between the negative examples and positive examples is no more
     the given ratio for an image.
    Args:
        loss (N, num_anchors): the loss for each example.
        labels (N, num_anchors): the labels.
        neg_pos_ratio:  the ratio between the negative examples and positive examples.
    """
    
    pos_mask = labels > 0
    num_pos = pos_mask.long().sum(dim=1, keepdim=True)
    num_neg = num_pos * neg_pos_ratio

    loss[pos_mask] = -math.inf
    _, indexes = loss.sort(dim=1, descending=True)
    _, orders = indexes.sort(dim=1)
    neg_mask = orders < num_neg
    return pos_mask | neg_mask


class CTPNLoss(nn.Module):
    def __init__(self, neg_pos_ratio=3):
        """Implement SSD Multibox Loss.
        Basically, Multibox loss combines classification loss
         and Smooth L1 regression loss.
        """
        super(CTPNLoss, self).__init__()
        self.neg_pos_ratio = neg_pos_ratio

    def forward(self, confidence, predicted_locations, labels, gt_locations):
        # cls_out, reg_out, anchor_gt_labels, anchor_gt_locations

        """


        Compute classification loss and smooth l1 loss.
        Args:
            confidence (batch_size, num_anchors, num_classes): class predictions.
            locations (batch_size, num_anchors, 4*seq_len): predicted locations.
            labels (batch_size, num_anchors): real labels of all the anchors.
            boxes (batch_size, num_anchors, 4*seq_len): real boxes corresponding all the anchors.


        """
#         pdb.set_trace()
        num_classes = 2
        batch_size = confidence.shape[0]

        confidence = confidence.contiguous().view(batch_size,-1,2)
        predicted_locations = predicted_locations.contiguous().view(batch_size,-1,4)
        labels = labels.view(batch_size,-1)
        gt_locations = gt_locations.view(batch_size,-1,4)
        
        mask_1 = labels>=0
        confidence = confidence[mask_1, :].unsqueeze(0)
        labels = labels[mask_1].unsqueeze(0)
        
        predicted_locations = predicted_locations[mask_1, :]
        gt_locations = gt_locations[mask_1, :]
        
        with torch.no_grad():
            loss = -F.log_softmax(confidence, dim=2)[:, :, 0]
            mask = hard_negative_mining(loss, labels, self.neg_pos_ratio)

        # pdb.set_trace()

        confidence = confidence[mask, :].view(-1, num_classes)
        labels = labels[mask].long()
        
        predicted_locations = predicted_locations[mask.squeeze(0), :]
        gt_locations = gt_locations[mask.squeeze(0), :]
        
        classification_loss = F.cross_entropy(confidence,labels, reduction='mean') if labels.numel() > 0 else Variable(torch.tensor(0.0).cuda(), requires_grad=True)
        loss_cls = torch.clamp(classification_loss, 0, 5) if classification_loss.numel() > 0 else Variable(torch.tensor(0.0).cuda(), requires_grad=True)
        
        pos_mask = labels > 0
        predicted_locations = predicted_locations[pos_mask, :].reshape(-1, 4)
        gt_locations = gt_locations[pos_mask, :].reshape(-1, 4)
        
        gt_locations = torch.cat((gt_locations[:, 1].unsqueeze(1), gt_locations[:, 3].unsqueeze(1)), 1)
        predicted_locations = torch.cat((predicted_locations[:, 1].unsqueeze(1), predicted_locations[:, 3].unsqueeze(1)), 1)
        
        loss_ver = smooth_l1_loss(predicted_locations, gt_locations) if gt_locations.numel() > 0 else Variable(torch.tensor(0.0).cuda(), requires_grad=True)
        loss_ver = torch.clamp(loss_ver, 0, 5) if loss_ver.numel() > 0 else Variable(torch.tensor(0.0).cuda(), requires_grad=True)
             
         
        
        loss_tatal = loss_ver + loss_cls
        
        loss_refine = torch.tensor(0.)
        return loss_tatal , loss_cls, loss_ver, loss_refine 
    




class ctpn_loss(nn.Module):
    
    def __init__(self, sigma):
        super(ctpn_loss, self).__init__()
        self.sigma = sigma

    def forward(self, pre_score, pre_reg, rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights,
                rpn_bbox_outside_weights, side_refinement):
        pre_score = pre_score.contiguous().view(-1, 2)
        pre_reg = pre_reg.contiguous().view(-1, 4)
        side_refinement = side_refinement.contiguous().view(-1)
        
        rpn_labels = rpn_labels.view(-1)
        rpn_bbox_targets = rpn_bbox_targets.view(-1, 4)
        rpn_bbox_inside_weights = rpn_bbox_inside_weights.view(-1, 4)
        rpn_bbox_outside_weights = rpn_bbox_outside_weights.view(-1, 4)

        fg_keep = (rpn_labels == 1)
        rpn_keep = (rpn_labels != -1)
        pre_score = pre_score[rpn_keep]
        rpn_labels = rpn_labels[rpn_keep].long()
        
        loss_cls = LossCL(pre_score, rpn_labels)
        loss_cls = torch.clamp(torch.mean(loss_cls), 0, 10) if loss_cls.numel() > 0 else torch.tensor(0.0)

        #         pre_reg = pre_reg[rpn_keep]
        #         rpn_bbox_targets = rpn_bbox_targets[rpn_keep]
        #         rpn_bbox_inside_weights = rpn_bbox_inside_weights[rpn_keep]
        #         rpn_bbox_outside_weights = rpn_bbox_outside_weights[rpn_keep]

        #         diff = torch.abs((rpn_bbox_targets - pre_reg)*rpn_bbox_inside_weights)
        #         less_one = (diff<1.0/self.sigma).float()
        #         loss_reg = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1- less_one) * (diff - 0.5/self.sigma)
        #         loss_reg = loss_reg*rpn_bbox_outside_weights
        #         loss_reg = torch.sum(loss_reg, 1)
        #         loss_reg = torch.sum(loss_reg)/(torch.sum(fg_keep).float()+1)
        #         loss_reg = torch.mean(loss_reg) if loss_reg.numel() > 0 else torch.tensor(0.0)
        #         loss_reg = torch.clamp(loss_reg, 0, 5) if loss_reg.numel() > 0 else torch.tensor(0.0)


        pre_reg = pre_reg[fg_keep]
        rpn_bbox_targets_gt = rpn_bbox_targets[fg_keep]

        rpn_bbox_targets_gt = torch.cat(
            (rpn_bbox_targets_gt[:, 1].unsqueeze(1), rpn_bbox_targets_gt[:, 3].unsqueeze(1)), 1)
        pre_reg = torch.cat((pre_reg[:, 1].unsqueeze(1), pre_reg[:, 3].unsqueeze(1)), 1)

        loss_reg = smooth_l1_loss(pre_reg, rpn_bbox_targets_gt) if rpn_bbox_targets_gt.numel() > 0 else torch.tensor(
            0.0)
        loss_reg = torch.clamp(loss_reg, 0, 5) if loss_reg.numel() > 0 else torch.tensor(0.0)

        side_refinement = side_refinement[fg_keep]
        side_refinement_gt = rpn_bbox_targets[fg_keep][:, 0]

        loss_refine = smooth_l1_loss(side_refinement,
                                       side_refinement_gt) if side_refinement_gt.numel() > 0 else torch.tensor(0.0)
        loss_refine = torch.clamp(loss_refine, 0, 5) if loss_refine.numel() > 0 else torch.tensor(0.0)

        loss_tatal = loss_cls + loss_reg #+ 2 * loss_refine

        return loss_tatal, loss_cls, loss_reg, loss_refine