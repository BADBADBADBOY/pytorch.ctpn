#-*- coding:utf-8 _*-
"""
@author:fxw
@file: 20191223.py
@time: 2019/12/23
"""
import math
import numpy as np
import cv2
import os
from tqdm import tqdm
import random
from torchvision.transforms import transforms

def order_point(pts):
    rect = np.zeros((4, 2), dtype="float32")
    # the top-left point will have the smallest sum, whereas
    # the bottom-right point will have the largest sum
    s = np.sum(pts, axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    # the top-right point will have the smallest difference,
    # whereas the bottom-left will have the largest difference
    d = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(d)]
    rect[3] = pts[np.argmax(d)]
    return rect

def cal_dis(coord1,coord2):
    return math.sqrt((coord1[0]-coord2[0])**2+(coord1[1]-coord2[1]))

def cal_k_b(coord1,coord2):
    k = (coord2[1]-coord1[1])/(coord2[0]-coord1[0])
    b = coord1[1]-k*coord1[0]
    return k,b

def cal_affine_coord(ori_coord,M):
    x = ori_coord[0]
    y = ori_coord[1]
    _x = x * M[0, 0] + y * M[0, 1] + M[0, 2]
    _y = x * M[1, 0] + y * M[1, 1] + M[1, 2]
    return [int(_x),int(_y)]

def random_rotate(img,random_angle = 20):
    angle = random.random() * 2 * random_angle - random_angle
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
    return img_rotation,rotation_matrix

def get_ctpn_bbox(rects,img_size,re_size,step=16):
    bboxes = []
    for line in rects:
        line = line.reshape(4,2)
        line[:, 0] = line[:, 0] / img_size[1] * re_size[1]
        line[:, 1] = line[:, 1] / img_size[0] * re_size[0]
        if((line[0][0]-line[1][0])==0):
            continue
        if((line[2][0]-line[3][0])==0):
            continue
        k_top, b_top = cal_k_b(line[0], line[1])
        k_bottom, b_bottom = cal_k_b(line[2], line[3])
        x_start = min(line[0][0], line[3][0])
        x_end = max(line[1][0], line[2][0])
        end_num = int(((x_end - x_start) - (x_end - x_start) % step) // step)
        # angle = math.atan(-k_top) * (180 / math.pi)
        bbox = []
        # if (abs(angle) > min_angle):
        #     end_num = end_num + 1
        start_num = 0
        # if (angle > 20):
        #     start_num = start_num - 1
        for i in range(start_num, end_num):
            y_s = int(k_top * (x_start + (i) * step) + b_top)
            y_e = int(k_bottom * (x_start + (i + 1) * step) + b_bottom)
            bbox.append([int(x_start + i * step), int(y_s), int(x_start + (i + 1) * step)-1, int(y_e)])
        if(len(bbox)==0):
            y_s = int(k_top * (x_start + (0) * step) + b_top)
            y_e = int(k_bottom * (x_start + (0 + 1) * step) + b_bottom)
            bbox.append([int(x_start + 0 * step), int(y_s), int(x_start + (0 + 1) * step)-1, int(y_e)])
        if(bbox[-1][2]<x_end):
            y_s = int(k_top * (x_start + (end_num) * step) + b_top)
            y_e = int(k_bottom * (x_start + (end_num) * step) + b_bottom)
            bbox.append([int(x_end- step), int(y_s), int(x_end)-1, int(y_e)])
        bboxes.append(bbox)
    bboxes = check_bbox(bboxes)
    return bboxes

def check_bbox(bboxes):
    new_bboxs = []
    for line in bboxes:
        box = []
        for item in line:
            if(item[0]>0 and item[1]>0 and item[2]>0 and item[3]>0):
                if(item[2]>item[0] and item[3]>item[1]):
                    box.append(item)
        new_bboxs.append(box)
    return new_bboxs

def resize_image(img,step=16,max_size=1200,min_size=600):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(min_size) / float(im_size_min)
    if np.round(im_scale * im_size_max) > max_size:
        im_scale = float(max_size) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // step == 0 else (new_h // step + 1) * step
    new_w = new_w if new_w // step == 0 else (new_w // step + 1) * step

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    re_size = re_im.shape
    return re_im,img_size,re_size

def flip_image(img,all_rects):
    img = cv2.flip(img, 1)
    h,w,_ = img.shape
    new_all_rects = []
    for item in all_rects:
        item = item.reshape(4,2)
        item[:,0]= w-item[:,0]
        item = order_point(item)
        new_all_rects.append(item.reshape(8))
    return img,new_all_rects

def get_rotate_img_boxes(img_file,txt_file,random_angle = 20):
    img = cv2.imread(img_file)
    fid = open(txt_file, 'r', encoding='utf-8')
    bboxes = []
    for line in fid.readlines():
        line = line.strip().replace('\ufeff', '').split(',')
        line = line[:8]
        line = [int(x) for x in line]
        line = np.array(line)
        line = line.reshape(4, 2)
        line = cv2.minAreaRect(line)
        line = cv2.boxPoints(line).astype(np.int)
        line = order_point(line)
        bboxes.append(line)
    img1, M = random_rotate(img,random_angle)
    new_all_rects = []
    for item in bboxes:
        rect = []
        for coord in item:
            rotate_coord = cal_affine_coord(coord,M)
            rect.append(rotate_coord)
        new_all_rects.append(np.array(rect).reshape(8))
    return img1,new_all_rects

def get_ctpn_bboxs(file_img,file_txt,step=16,random_angle=20):
    img, new_all_rects = get_rotate_img_boxes(file_img, file_txt,random_angle)
#     is_flip = np.random.choice([True,False],1)[0]
#     print(is_flip)
#     if(is_flip is True):
#         img,new_all_rects = flip_image(img, new_all_rects)
    re_im, img_size, re_size = resize_image(img, step)
    h, w, c = re_im.shape
    im_info = np.array([h, w, c]).reshape([1, 3])
    bboxs = get_ctpn_bbox(new_all_rects, img_size, re_size, step)
    return bboxs,re_im,im_info

def test_more(path,save_path):
    step = 16
    if not os.path.exists(os.path.join(save_path, "image")):
        os.makedirs(os.path.join(save_path, "image"))
    if not os.path.exists(os.path.join(save_path, "label")):
        os.makedirs(os.path.join(save_path, "label"))
    if not os.path.exists(os.path.join(save_path, "show_image")):
        os.makedirs(os.path.join(save_path, "show_image"))
    imageFiles = os.listdir(os.path.join(path,'image'))
    bar = tqdm(total=len(imageFiles))
    for file in imageFiles:
        bar.update(1)
        file_img = os.path.join(path,'image',file)
        file_txt = os.path.join(path,'label',file.split('.')[0]+'.txt')
        img,new_all_rects = get_rotate_img_boxes(file_img, file_txt)
        #######
        re_im, img_size, re_size = resize_image(img,step)
        ######
        img_ori = re_im.copy()
        bboxs = get_ctpn_bbox(new_all_rects, img_size, re_size, step)
        with open(os.path.join(save_path,'label',file.split('.')[0]+'.txt'),'w+',encoding='utf-8') as fid:
            for line_box in bboxs:
                for item in line_box:
                    item = [str(x) for x in item]
                    item = ",".join(item)
                    fid.write(item+"\n")
        cv2.imwrite(os.path.join(save_path,'image',file),img_ori)
        for line_box in bboxs:
            for item in line_box:
                re_im = cv2.rectangle(re_im, (item[0], item[1]), (item[2], item[3]), (0, 0, 255))
        cv2.imwrite(os.path.join(save_path,'show_image',file), re_im)
    bar.close()