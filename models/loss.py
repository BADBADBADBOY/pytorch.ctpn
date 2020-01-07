import torch
import torch.nn as nn
import pdb
import torch.nn.functional as F
LossCL = nn.CrossEntropyLoss().cuda()
class ctpn_loss(nn.Module):
    def __init__(self,sigma):
        super(ctpn_loss,self).__init__()
        self.sigma = sigma
    def forward(self,pre_score,pre_reg,rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights):
        pre_score = pre_score.contiguous().view(-1,2)
        pre_reg = pre_reg.contiguous().view(-1,4)
        rpn_labels = rpn_labels.view(-1)
        rpn_bbox_targets = rpn_bbox_targets.view(-1,4)
        rpn_bbox_inside_weights = rpn_bbox_inside_weights.view(-1,4)
        rpn_bbox_outside_weights = rpn_bbox_outside_weights.view(-1,4)
        
        fg_keep = (rpn_labels==1)
        rpn_keep = (rpn_labels!=-1)
        pre_score = pre_score[rpn_keep]
        rpn_labels = rpn_labels[rpn_keep].long()
#         loss_cls = F.nll_loss(F.log_softmax(pre_score, dim=-1), rpn_labels)
        loss_cls = LossCL(pre_score,rpn_labels)
        loss_cls = torch.clamp(torch.mean(loss_cls), 0, 10) if loss_cls.numel() > 0 else torch.tensor(0.0)
        
        pre_reg = pre_reg[rpn_keep]
        rpn_bbox_targets = rpn_bbox_targets[rpn_keep]
        rpn_bbox_inside_weights = rpn_bbox_inside_weights[rpn_keep]
        rpn_bbox_outside_weights = rpn_bbox_outside_weights[rpn_keep]
        
        diff = torch.abs((rpn_bbox_targets - pre_reg)*rpn_bbox_inside_weights)
        less_one = (diff<1.0/self.sigma).float()
        loss_reg = less_one * 0.5 * diff ** 2 * self.sigma + torch.abs(1- less_one) * (diff - 0.5/self.sigma)
        loss_reg = loss_reg*rpn_bbox_outside_weights
        loss_reg = torch.sum(loss_reg, 1)
        loss_reg = torch.sum(loss_reg)/(torch.sum(fg_keep).float()+1)
        loss_reg = torch.mean(loss_reg) if loss_reg.numel() > 0 else torch.tensor(0.0)
        loss_reg = torch.clamp(loss_reg, 0, 1) if loss_cls.numel() > 0 else torch.tensor(0.0)
        
        loss_tatal = loss_cls+loss_reg
        return loss_tatal,loss_cls,loss_reg