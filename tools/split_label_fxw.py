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

def get_ctpn_bbox(txt_file,img_size,re_size,step=8,min_angle=10):
    fid = open(txt_file, 'r', encoding='utf-8')
    bboxes = []
    for line in fid.readlines():
        line = line.strip().split(',')
        line = line[:8]
        line = [int(x) for x in line]
        line = np.array(line)
        line = line.reshape(4, 2)
        line = cv2.minAreaRect(line)
        line = cv2.boxPoints(line).astype(np.int)
        line = order_point(line)
        line[:, 0] = line[:, 0] / img_size[1] * re_size[1]
        line[:, 1] = line[:, 1] / img_size[0] * re_size[0]
        k_top, b_top = cal_k_b(line[0], line[1])
        k_bottom, b_bottom = cal_k_b(line[2], line[3])
        x_start = min(line[0][0], line[3][0])
        x_end = max(line[1][0], line[2][0])
        end_num = int(((x_end - x_start) - (x_end - x_start) % step) // step)
        mod = (x_end - x_start) % step
        angle = math.atan(-k_top) * (180 / math.pi)
        bbox = []
        if (mod > step // 2 or abs(angle) > min_angle):
            end_num = end_num + 1
        start_num = 0
        if (angle > 10):
            start_num = start_num - 1
        for i in range(start_num, end_num):
            y_s = int(k_top * (x_start + (i) * step) + b_top)
            y_e = int(k_bottom * (x_start + (i + 1) * step) + b_bottom)
            bbox.append([int(x_start + i * step), int(y_s), int(x_start + (i + 1) * step)-1, int(y_e)])
        if(len(bbox)==0):
            y_s = int(k_top * (x_start + (0) * step) + b_top)
            y_e = int(k_bottom * (x_start + (0 + 1) * step) + b_bottom)
            bbox.append([int(x_start + 0 * step), int(y_s), int(x_start + (0 + 1) * step)-1, int(y_e)])
        if(bbox[-1][2]<x_end-step/2):
            y_s = int(k_top * (x_start + (end_num) * step) + b_top)
            y_e = int(k_bottom * (x_start + (end_num + 1) * step) + b_bottom)
            bbox.append([int(x_start + (end_num) * step), int(y_s), int(x_start + (end_num + 1) * step)-1, int(y_e)])
        bboxes.append(bbox)
    fid.close()
    return bboxes

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

def test_single():
    file = r'C:\Users\fangxuwei\Desktop\show2\100.jpg'
    file_txt = r'C:\Users\fangxuwei\Desktop\show2\100.txt'
    step = 8
    min_angle = 10
    img = cv2.imread(file)
    #######
    re_im, img_size, re_size = resize_image(img,step)
    ######
    bboxs = get_ctpn_bbox(file_txt, img_size, re_size, step, min_angle)
    for line_box in bboxs:
        for item in line_box:
            re_im = cv2.rectangle(re_im, (item[0], item[1]), (item[2], item[3]), (0, 0, 255))
    cv2.imwrite('result.jpg', re_im)
    # cv2.imshow('image',image)
    # cv2.waitKey()

def test_more(path,save_path):
    step = 16
    min_angle = 10
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
        img = cv2.imread(file_img)
        #######
        re_im, img_size, re_size = resize_image(img,step)
        ######
        img_ori = re_im.copy()
        bboxs = get_ctpn_bbox(file_txt, img_size, re_size, step, min_angle)
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

path = r'../data'
save_path =r'../data/split'
test_more(path,save_path)