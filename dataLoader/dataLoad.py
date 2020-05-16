"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: dataLoad.py
@time: 2020/4/5 10:34

"""
from PIL import Image
from torch.utils import data
from dataLoader.shrinkbox import *
import random
import torchvision.transforms as transforms
import torch
import os

# ic15_root_dir = '/home/aistudio/work/data/icdar/'
# ic15_train_data_dir = ic15_root_dir + 'train_img/'
# ic15_train_gt_dir = ic15_root_dir + 'train_gt/'


ic15_root_dir = '/home/aistudio/work/data/MTWI2018/'
ic15_train_data_dir = ic15_root_dir + 'image/'
ic15_train_gt_dir = ic15_root_dir + 'label/'



random.seed(123456)


def get_images(img_path,max_size):
    img = Image.open(img_path).convert('RGB')
    img = np.array(img)
    re_im, im_scale = resize_image(img, max_size)
    return re_im,im_scale,img.shape

def get_auchor_bbox(gt_path,im_scale,im_shape,re_im_show):
    polys = []
    with open(gt_path, 'r',encoding='utf-8') as f:
        lines = f.readlines()
    tag = []
    for line in lines:
        splitted_line = line.strip().lower().replace('\ufeff','').split(',')
        if ('#' in splitted_line[-1]):
            tag.append(-1)
        else:
            tag.append(1)
        x1, y1, x2, y2, x3, y3, x4, y4 = map(float, splitted_line[:8])
        poly = np.array([x1, y1, x2, y2, x3, y3, x4, y4]).reshape([4, 2])
        poly[:, 0] = poly[:, 0]/im_shape[1] * im_scale[1]
        poly[:, 1] = poly[:, 1]/im_shape[0] * im_scale[0]
        try:
            poly = orderConvex(poly)
        except:
            poly = np.array([[0,0],[1,1],[2,2],[1,0]]).astype(np.float)
        polys.append(poly)

    res_polys = []
    for ii in range(len(polys)):
        poly = polys[ii]
        # delete polys with width less than 10 pixel
        if np.linalg.norm(poly[0] - poly[1]) < 5 or np.linalg.norm(poly[3] - poly[0]) < 5:
            continue

        tag_t = tag[ii]
        res = shrink_poly(poly)
        for p in res:
            re_im_show = cv.polylines(re_im_show, [p.astype(np.int32).reshape((-1, 1, 2))], True, color=(0, 255, 0), thickness=1)

        res = res.reshape([-1, 4, 2])
        for r in res:
            x_min = np.min(r[:, 0])
            y_min = np.min(r[:, 1])
            x_max = np.max(r[:, 0])
            y_max = np.max(r[:, 1])
            res_polys.append([x_min, y_min, x_max, y_max,1, tag_t])

    return res_polys,re_im_show

class IC15Loader(data.Dataset):
    def __init__(self,size_list):
        self.size_list = size_list
        self.train_size = size_list[-1]

        data_dirs = [ic15_train_data_dir]
        gt_dirs = [ic15_train_gt_dir]

        self.img_paths = []
        self.gt_paths = []

        for data_dir, gt_dir in zip(data_dirs, gt_dirs):
            img_names =os.listdir(data_dir)
            img_paths = []
            gt_paths = []
            for idx, img_name in enumerate(img_names):
                if(not img_name.endswith('.jpg')):
                    continue
                img_path = data_dir + img_name
                img_paths.append(img_path)
                

                # gt_name = 'gt_'+img_name.split('.')[0] + '.txt'
                gt_name = img_name.split('.')[0] + '.txt'
                gt_path = gt_dir + gt_name
                gt_paths.append(gt_path)

            self.img_paths.extend(img_paths)
            self.gt_paths.extend(gt_paths)

    def __len__(self):
        return len(self.img_paths)

    def get_random_train_size(self):

        self.train_size = np.random.choice(self.size_list,1)[0]

    def __getitem__(self, index):
        img_path = self.img_paths[index]
        img, img_scale,img_shape = get_images(img_path,self.train_size)
        h, w, c = img.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        im_info = torch.from_numpy(im_info)
        gt_path_index = torch.from_numpy(np.array([index]))
        img_scale =  torch.from_numpy(np.array([img_scale]))
        img_shape =  torch.from_numpy(np.array([img_shape]))
        img = transforms.ToTensor()(img)
#         img = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(img)
        return img, img_scale,img_shape,gt_path_index,im_info

def get_bboxes(img,gt_files,gt_path_index,img_scales,img_shapes):
    """
    :param img: 图片
    :param gt_files: 训练集gt的txt文件
    :param gt_path_index: batch图片的gt的index
    :param img_scales: 缩放的尺度
    :return batch_res_polys: 一个batch的宽16的gt框
    
    """
    batch_res_polys = []
    for i in range(img.shape[0]):
        img_show = img[i].cpu().numpy() * 255
        img_show = img_show.transpose((1, 2, 0)).copy()
        res_polys, img_show = get_auchor_bbox(gt_files[gt_path_index[i]], img_scales[i].numpy()[0],img_shapes[i].numpy()[0], img_show)
        batch_res_polys.append(res_polys)

    return  batch_res_polys

# if __name__ =="__main__":
#     data_loader = IC15Loader(is_transform=True)
#     gt_files = data_loader.gt_paths
#     print(gt_files)
#     train_loader = torch.utils.data.DataLoader(
#             data_loader,
#             batch_size=4,
#             shuffle=True,
#             num_workers=0,
#             drop_last=True,
#             pin_memory=True)
#     from PIL import Image
#     for batch_idx, (img, img_scales,gt_path_index,im_info) in enumerate(train_loader):
#         data_loader.get_train_size()
#         print(img.shape)
#         print(img_scales)
#         print(gt_path_index)
#         print(im_info.shape)
#         print(gt_files[gt_path_index[0]])
#         batch_res_polys = []
#         for i in range(img.shape[0]):
#             img_show  = img[i].numpy()*255
#             img_show = img_show.transpose((1,2,0)).copy()
#             res_polys,img_show= get_auchor_bbox(gt_files[gt_path_index[i]],img_scales[i].numpy()[0],img_show)
#             cv.imwrite('../result_'+str(i)+'.jpg',img_show)
#             print(res_polys)
#         break
