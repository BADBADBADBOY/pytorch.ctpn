"""
#!-*- coding=utf-8 -*-
@author: BADBADBADBADBOY
@contact: 2441124901@qq.com
@software: PyCharm Community Edition
@file: shrinkbox.py
@time: 2020/4/5 10:48

"""
import numpy as np
from shapely.geometry import Polygon
import cv2 as cv

def Add_Padding(image,top, bottom, left, right, color):
    padded_image = cv.copyMakeBorder(image, top, bottom,
                                      left, right, cv.BORDER_CONSTANT, value=color)
    return padded_image

def resize_image(img,max_size,color=(0,0,0)):
    img_size = img.shape
    im_size_max = np.max(img_size[0:2])
    im_scale = float(max_size) / float(im_size_max)
    new_h = np.round(img_size[0] * im_scale).astype(np.int)
    new_w = np.round(img_size[1] * im_scale).astype(np.int)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv.resize(img, (new_w, new_h), interpolation=cv.INTER_LINEAR)
    im_scale = re_im.shape
    max_size = max(new_w, new_h)
    
    if(new_h!=max_size and new_w!=max_size):
        re_im = Add_Padding(re_im,0,0,0,max_size-new_w,color)
        re_im = Add_Padding(re_im, 0, max_size-new_h, 0,0, color)
    else:
        if(new_h==max_size):
            re_im = Add_Padding(re_im,0,0,0,max_size-new_w,color)
        else:
            re_im = Add_Padding(re_im, 0, max_size-new_h, 0,0, color)
            
    return re_im, im_scale

def pickTopLeft(poly):

    idx = np.argsort(poly[:, 0])
    if poly[idx[0], 1] < poly[idx[1], 1]:
        s = idx[0]
    else:
        s = idx[1]

    return poly[(s, (s + 1) % 4, (s + 2) % 4, (s + 3) % 4), :]


def orderConvex(p):

    points = Polygon(p).convex_hull

    points = np.array(points.exterior.coords)[:4]

    points = points[::-1]
    points = pickTopLeft(points)
    points = np.array(points).reshape([4, 2])
    return points


def shrink_poly(poly, r=16):
    # y = kx + b
    """
    :param poly: 标注框
    :param r: 下采样倍数
    :return res: 得到的anchor
    """
    sort_coord = sorted(poly[:, 0].reshape(-1).tolist())
    x_min = int((sort_coord[1]+sort_coord[0])/2)
    x_max = int((sort_coord[2]+sort_coord[3])/2)

    #     x_min = int(np.min(poly[:, 0]))
    #     x_max = int(np.max(poly[:, 0]))
    
    if((poly[1][0] - poly[0][0])!=0):
        k1_1 = (poly[1][0] - poly[0][0])
    else:
        k1_1 = 1e-10

    k1 = (poly[1][1] - poly[0][1]) / k1_1
    b1 = poly[0][1] - k1 * poly[0][0]
    
    if((poly[2][0] - poly[3][0])!=0):
        k2_1 = (poly[2][0] - poly[3][0])
    else:
        k2_1 = 1e-10

    k2 = (poly[2][1] - poly[3][1]) / k2_1
    b2 = poly[3][1] - k2 * poly[3][0]

    res = []

    start = int((x_min // r + 1) * r)
    end = int((x_max // r) * r)

    p = x_min
    if (start - p > r//4):

        res.append([start - r, int(k1 * p + b1),
                    start - 1, int(k1 * (p + r-1) + b1),
                    start - 1, int(k2 * (p + r-1) + b2),
                    start - r, int(k2 * p + b2)])

    for p in range(start, end + 1, r):
        res.append([p, int(k1 * p + b1),
                    (p + r-1), int(k1 * (p + r-1) + b1),
                    (p + r-1), int(k2 * (p + r-1) + b2),
                    p, int(k2 * p + b2)])
    return np.array(res, dtype=np.int).reshape([-1, 8])
