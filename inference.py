import os
os.environ['CUDA_VISIBLE_DEVICES'] ='3'
import cv2
import numpy as np
import torch
import torch.nn.functional as F
from utils.rpn_msr.proposal_layer import proposal_layer
from utils.text_connector.detectors import TextDetector
from torchvision.transforms import transforms
from models.ctpn import *
import time

def rotate(img,angle):
    w, h = img.shape[:2]
    rotation_matrix = cv2.getRotationMatrix2D((h / 2, w / 2), angle, 1)
    img_rotation = cv2.warpAffine(img, rotation_matrix, (h, w))
    return img_rotation

def resize_image(img):
    img_size = img.shape
    im_size_min = np.min(img_size[0:2])
    im_size_max = np.max(img_size[0:2])

    im_scale = float(600) / float(im_size_min)
    if np.round(im_scale * im_size_max) > 1200:
        im_scale = float(1200) / float(im_size_max)
    new_h = int(img_size[0] * im_scale)
    new_w = int(img_size[1] * im_scale)

    new_h = new_h if new_h // 16 == 0 else (new_h // 16 + 1) * 16
    new_w = new_w if new_w // 16 == 0 else (new_w // 16 + 1) * 16

    re_im = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
    return re_im, (new_h / img_size[0], new_w / img_size[1])

def toTensorImage(image,is_cuda=True):
    image = transforms.ToTensor()(image)
    image = transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])(image).unsqueeze(0)
    if(is_cuda is True):
        image = image.cuda()
    return image


class DetectImg():
    def load_model(self,model_path):
        model_dict = torch.load(model_file)
        model = CTPN_Model().cuda()
        model.load_state_dict(model_dict)
        self.model = model
    def detect(self,img_file):
        start_time = time.time()
        img = cv2.imread(im_file)
        img_ori,(rh, rw) = resize_image(img)
        h, w, c = img_ori.shape
        im_info = np.array([h, w, c]).reshape([1, 3])
        img = toTensorImage(img_ori)

        pre_score,pre_reg = self.model(img)
        score = pre_score.reshape((pre_score.shape[0], 10, 2, pre_score.shape[2], pre_score.shape[3])).squeeze(0).permute(0,2,3,1).reshape((-1, 2))
        score = F.softmax(score, dim=1)
        score = score.reshape((10, pre_reg.shape[2], -1, 2))

        pre_score =score.permute(1,2,0,3).reshape(pre_reg.shape[2],pre_reg.shape[3],-1).unsqueeze(0).cpu().detach().numpy()
        pre_reg =pre_reg.permute(0,2,3,1).cpu().detach().numpy()

        textsegs, _ = proposal_layer(pre_score, pre_reg, im_info)
        scores = textsegs[:, 0]
        textsegs = textsegs[:, 1:5]

        textdetector = TextDetector(DETECT_MODE='O')
        boxes ,text_proposals= textdetector.detect(textsegs, scores[:, np.newaxis], img_ori.shape[:2])
        boxes = np.array(boxes, dtype=np.int)
        text_proposals = text_proposals.astype(np.int)
        print('cost_time:'+str(time.time()-start_time)+'s')
        return boxes,text_proposals
        
def show_img(save_path,im_file,boxes,text_proposals):
    img_ori = cv2.imread(im_file)
    img_ori,(rh, rw) = resize_image(img_ori)
    im_name = im_file.split('/')[-1].split('.')[0]
    for item in text_proposals:
        img_ori = cv2.rectangle(img_ori,(item[0],item[1]),(item[2],item[3]),(0,255,255))
    for i, box in enumerate(boxes):
        cv2.polylines(img_ori, [box[:8].astype(np.int32).reshape((-1, 1, 2))], True, color=(255, 0, 0),
                  thickness=2)
    img_ori = cv2.resize(img_ori, None, None, fx=1.0 / rh, fy=1.0 / rw, interpolation=cv2.INTER_LINEAR)
    cv2.imwrite(os.path.join(save_path,im_name+'_result.jpg'), img_ori[:, :, ::-1])

if __name__=="__main__":    
    import os
    dir_path = './test'
	save_path = './result'
    files = os.listdir(dir_path)
    model_path  = './model_save/ctpn_50.pth'
    detect_obj = DetectImg()
    detect_obj.load_model(model_path )
    for file in files:
        im_file = os.path.join(dir_path,file)
        boxes,text_proposals = detect_obj.detect(im_file)
        show_img(im_file,boxes,text_proposals)



