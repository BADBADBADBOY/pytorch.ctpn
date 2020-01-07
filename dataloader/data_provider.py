# encoding:utf-8
import os
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
from dataloader.data_util import GeneratorEnqueuer
from dataloader.create_random_ctpn_bbox import get_ctpn_bboxs

DATA_FOLDER = "/src/notebooks/train_data/cwtdata/" #train data root path 

stride_step = 16 #  pooling down
random_angle = 5 # img will transform random from -5 to 5 when training


def get_training_data():
    img_files = []
    exts = ['jpg', 'png', 'jpeg', 'JPG']
    for parent, dirnames, filenames in os.walk(os.path.join(DATA_FOLDER, "image")):
        for filename in filenames:
            for ext in exts:
                if filename.endswith(ext):
                    img_files.append(os.path.join(parent, filename))
                    break
    print('Find {} images'.format(len(img_files)))
    return img_files


def load_annoataion(bboxs):
    bbox = []
    for line_box in bboxs:
        for item in line_box:
            x_min, y_min, x_max, y_max = map(int, item)
            bbox.append([x_min, y_min, x_max, y_max, 1])
    return bbox


def generator(vis=False):
    image_list = np.array(get_training_data())
    print('{} training images in {}'.format(image_list.shape[0], DATA_FOLDER))
    index = np.arange(0, image_list.shape[0])
    while True:
        np.random.shuffle(index)
        for i in index:
            try:
                im_fn = image_list[i]
                _, fn = os.path.split(im_fn)
                fn, _ = os.path.splitext(fn)
                txt_fn = os.path.join(DATA_FOLDER, "label", fn + '.txt')
                if not os.path.exists(txt_fn):
                    print("Ground truth for image {} not exist!".format(im_fn))
                    continue
                bboxs,im,im_info = get_ctpn_bboxs(im_fn,txt_fn,step=stride_step,random_angle=random_angle)
                bbox = load_annoataion(bboxs)
                if len(bbox) == 0:
                    print("Ground truth for image {} empty!".format(im_fn))
                    continue
                yield [im], bbox, im_info

            except Exception as e:
                print(e)
                continue


def get_batch(num_workers, **kwargs):
    try:
        enqueuer = GeneratorEnqueuer(generator(**kwargs), use_multiprocessing=True)
        enqueuer.start(max_queue_size=24, workers=num_workers)
        generator_output = None
        while True:
            while enqueuer.is_running():
                if not enqueuer.queue.empty():
                    generator_output = enqueuer.queue.get()
                    break
                else:
                    time.sleep(0.01)
            yield generator_output
            generator_output = None
    finally:
        if enqueuer is not None:
            enqueuer.stop()


