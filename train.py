import os
os.environ['CUDA_VISIBLE_DEVICES'] = '1'
import torch
from torch import optim
import numpy as np
from torchvision.transforms import transforms
from dataloader.data_provider import get_batch,DATA_FOLDER
from models.loss import ctpn_loss
from models.ctpn import CTPN_Model
from utils.rpn_msr.anchor_target_layer import anchor_target_layer

def toTensorImage(image,is_cuda=True):
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image).unsqueeze(0)
    if(is_cuda is True):
        image = image.cuda()
    return image

def toTensor(item,is_cuda=True):
    item = torch.Tensor(item)
    if(is_cuda is True):
        item = item.cuda()
    return item

random_seed = 2020
torch.random.manual_seed(random_seed)
np.random.seed(random_seed)

gen = get_batch(num_workers=2, vis=False)

model = CTPN_Model().cuda()
critetion = ctpn_loss(sigma=9)
optimizer = optim.SGD(model.parameters(), lr=0.001, momentum=0.9)
scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=50, gamma=0.1)

pre_model_dict = torch.load('./vgg16.model')
model.load_state_dict(pre_model_dict)

model.train()

epochs =200

image_nums = len(os.listdir(os.path.join(DATA_FOLDER,'image')))

for epoch in range(epochs):
    scheduler.step()
    loss_total_list = []
    loss_cls_list = []
    loss_ver_list = []
    for i in range(image_nums):
        image_ori, bbox, im_info = next(gen)
        image = toTensorImage(image_ori[0])
        optimizer.zero_grad()

        score_pre,vertical_pred = model(image)

        score_pre =score_pre.permute(0,2,3,1)
        vertical_pred =vertical_pred.permute(0,2,3,1)
        gt_boxes = np.array(bbox)
        rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights = anchor_target_layer(image_ori[0],score_pre, gt_boxes, im_info)
        rpn_labels = toTensor(rpn_labels)
        rpn_bbox_targets = toTensor(rpn_bbox_targets)
        rpn_bbox_inside_weights = toTensor(rpn_bbox_inside_weights)
        rpn_bbox_outside_weights = toTensor(rpn_bbox_outside_weights)

        loss_tatal,loss_cls,loss_ver = critetion(score_pre,vertical_pred,rpn_labels, rpn_bbox_targets, rpn_bbox_inside_weights, rpn_bbox_outside_weights)

        loss_tatal.backward()
        optimizer.step()
        loss_total_list.append(loss_tatal.item())
        loss_cls_list.append(loss_cls.item())
        loss_ver_list.append(loss_ver.item())
        if(i%50==0):
            print("{}/{}/{}/{}    loss_total:{}-------loss_cls:{}--------loss_ver:{}---------Lr:{}".format(epoch,epochs,i,image_nums,loss_tatal.item(),loss_cls.item(),loss_ver.item(),scheduler.get_lr()[0]))
            
    print("********epoch_loss_total:{}********epoch_loss_cls:{}********epoch_loss_ver:{}********Lr:{}".format(np.mean(loss_total_list),np.mean(loss_cls_list),np.mean(loss_ver_list),scheduler.get_lr()[0]))
    torch.save(model.state_dict(),'./model_save/ctpn_'+str(epoch)+'.pth')
        
    
    
    