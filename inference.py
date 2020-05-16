"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: inference.py
@time: 2020/4/5 9:25

"""
import os

os.environ['CUDA_VISIBLE_DEVICES'] = '0'
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
from torchvision.transforms import transforms
from models.ctpn import *
import time
from tqdm import tqdm
from PIL import Image

def Add_Padding(image,top, bottom, left, right, color):
    padded_image = cv2.copyMakeBorder(image, top, bottom,
                                      left, right, cv2.BORDER_CONSTANT, value=color)
    return padded_image

def rotate(img, angle):
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
    return img_rotation

def resize_image(img,max_size=1200,color=(0,0,0)):
    
    img_size = img.shape
    im_size_max = np.max(img_size[0:2])
    im_scale = float(max_size) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w_w, new_h_h), interpolation=cv2.INTER_LINEAR)
           
    return re_im, (im_scale*(new_h_h/new_h),im_scale*(new_w_w/new_w))


def toTensorImage(image, is_cuda=True):
    image = transforms.ToTensor()(image)
#     image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image).unsqueeze(0)
    image = (image).unsqueeze(0)
    if (is_cuda is True):
        image = image.cuda()
    return image


class DetectImg():
    
    def load_model(self, model_file,base_model,detect_type):
        model_dict = torch.load(model_file)
        model = CTPN_Model(base_model,pretrained=False).cuda()
        model.load_state_dict(model_dict)
        self.model = model
        self.detect_type = detect_type
        self.model.eval()

    def detect(self, img_file):
        img = Image.open(img_file).convert('RGB')
        img = np.array(img)
        img_ori, (rh, rw) = resize_image(img)
        h, w, c = img_ori.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        img = toTensorImage(img_ori)
        with torch.no_grad():
            pre_score, pre_reg, refine_ment = self.model(img)
        score = pre_score.reshape((pre_score.shape[0], 10, 2, pre_score.shape[2], pre_score.shape[3])).squeeze(
            0).permute(0, 2, 3, 1).reshape((-1, 2))
        score = F.softmax(score, dim=1)
        score = score.reshape((10, pre_reg.shape[2], -1, 2))

        pre_score = score.permute(1, 2, 0, 3).reshape(pre_reg.shape[2], pre_reg.shape[3], -1).unsqueeze(
            0).cpu().detach().numpy()
        pre_reg = pre_reg.permute(0, 2, 3, 1).cpu().detach().numpy()
        refine_ment = refine_ment.permute(0, 2, 3, 1).cpu().detach().numpy()

        textsegs, _ = proposal_layer(pre_score, pre_reg, refine_ment, im_info)
        scores = textsegs[:, 0]
        textsegs = textsegs[:, 1:5]

        textdetector = TextDetector(DETECT_MODE = self.detect_type)
        boxes, text_proposals = textdetector.detect(textsegs, scores[:, np.newaxis], img_ori.shape[:2])
        boxes = np.array(boxes, dtype=np.int)
        text_proposals = text_proposals.astype(np.int)
        return boxes, text_proposals, rh, rw


def show_img(save_path, im_file, boxes, text_proposals):
    img_ori = cv2.imread(im_file)
    img_ori, (rh, rw) = resize_image(img_ori)
    im_name = im_file.split('/')[-1].split('.')[0]
    for item in text_proposals:
        img_ori = cv2.rectangle(img_ori, (item[0], item[1]), (item[2], item[3]), (0, 255, 255))
    img_ori = cv2.resize(img_ori, None, None, fx=1.0 / rw, fy=1.0 / rh, interpolation=cv2.INTER_LINEAR)
    for i, box in enumerate(boxes):
        cv2.polylines(img_ori, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0),
                      thickness =2)
    img_ori = cv2.resize(img_ori, None, None, fx=0.9, fy=0.9, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_path, im_name + '.jpg'), img_ori)


if __name__ == "__main__":


#     dir_path = '/src/notebooks/train_data/icdar2015/ch4_test_images'
    dir_path = '/home/aistudio/work/data/MTWI2018/image'

    
    model_file = './model_save/ctpn_35.pth'
    img_save_path = './result'
    txt_save_path = './pre_gt'
    detect_type = 'O' # 'O' or 'H'
    base_model = 'shufflenet_v2_x1_0'
    
    detect_obj = DetectImg()
    detect_obj.load_model(model_file,base_model,detect_type)
    
    files = os.listdir(dir_path)
    bar = tqdm(total=len(files))
    for file in files:
        bar.update(1)
        fid = open(os.path.join(txt_save_path, 'res_' + file.split('.')[0] + '.txt'), 'w+', encoding='utf-8')
        im_file = os.path.join(dir_path, file)
        boxes, text_proposals, rh, rw = detect_obj.detect(im_file)
        for i, box in enumerate(boxes):
            box = box[:8].reshape(4, 2)
            box[:, 0] = box[:, 0] / rw
            box[:, 1] = box[:, 1] / rh
            box = box.reshape(1, 8).astype(np.int32)
            box = [str(x) for x in box.reshape(-1).tolist()]
            fid.write(','.join(box) + '\n')
        fid.close()
        show_img(img_save_path, im_file, boxes, text_proposals)



