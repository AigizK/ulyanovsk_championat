from utils.torch_utils import select_device
from models.experimental import attempt_load
import torch
from utils.general import check_img_size,non_max_suppression,scale_coords
import cv2 as cv
import numpy as np
from torch import Tensor
from classes import Plate,Char
from sort import sortCharacters,sortPlatePossibilities
from itertools import product
import argparse

def letterbox(img, new_shape=(640, 640), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True, stride=32):
    shape = img.shape[:2]  # current shape [height, width]
    if isinstance(new_shape, int):
        new_shape = (new_shape, new_shape)

    # Scale ratio (new / old)
    r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
    if not scaleup:  # only scale down, do not scale up (for better test mAP)
        r = min(r, 1.0)

    # Compute padding
    ratio = r, r  # width, height ratios
    new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
    dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
    if auto:  # minimum rectangle
        dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
    elif scaleFill:  # stretch
        dw, dh = 0.0, 0.0
        new_unpad = (new_shape[1], new_shape[0])
        ratio = new_shape[1] / shape[1], new_shape[0] / shape[0]  # width, height ratios

    dw /= 2  # divide padding into 2 sides
    dh /= 2

    if shape[::-1] != new_unpad:  # resize
        img = cv.resize(img, new_unpad, interpolation=cv.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv.copyMakeBorder(img, top, bottom, left, right, cv.BORDER_CONSTANT, value=color)  # add border
    return img, ratio, (dw, dh)


def detechPlate(source_img):
    plates=[]
    org_img = source_img
    img = letterbox(org_img, imgsize_plate)[0]
    img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
    img = np.ascontiguousarray(img)
    img = torch.from_numpy(img).to(device)
    img = img.half() if half else img.float()  # uint8 to fp16/32
    img /= 255.0  # 0 - 255 to 0.0 - 1.0
    if img.ndimension() == 3:
        img = img.unsqueeze(0)
    # Inference
    pred = model_plate(img, augment=False)[0]
    pred = non_max_suppression(pred, 0.4, 0.5, classes=0, agnostic=False)
    for i, det in enumerate(pred):
        gn = torch.tensor(img.shape)[[1, 0, 1, 0]]  # normalization gain whwh
        if det is not None and len(det):
            det[:, :4] = scale_coords(img.shape[2:], det[:, :4], org_img.shape).round()
            
            for *xyxy, conf, cls in reversed(det):
                if(conf.item()>0.59):
                    x1, y1,x2,y2 = int(xyxy[0]), int(xyxy[1]), int(xyxy[2]), int(xyxy[3])
                    plates.append((x1, y1,x2,y2))
    return plates 

import cv2 as cv2

def rotate_image(image, angle):
    image_center = tuple(np.array(image.shape[1::-1]) / 2)
    rot_mat = cv2.getRotationMatrix2D(image_center, angle, 1.0)
    result = cv2.warpAffine(image, rot_mat, image.shape[1::-1], flags=cv2.INTER_LINEAR)
    return result

if __name__ == '__main__':
    
    parser = argparse.ArgumentParser()
    parser.add_argument('--source', type=str, default='/root/stepik/hacaton_ulyanovsk/dataset/train', help='source image ')
    opt = parser.parse_args()

    device = "cpu"
    half = False

    plate_weights = 'detect_plate_weight.pt'
    model_plate = attempt_load(plate_weights, map_location=device)
    print("Loading plate detection weights:   success")
    
    stride = int(model_plate.stride.max())  # model_char stride
    imgsize_plate=640
    imgsize_plate = check_img_size(imgsize_plate, s=stride)  # check img_size

    from os import listdir
    from os.path import isfile, join
    mypath=opt.source
    onlyfiles = [f for f in listdir(mypath) if isfile(join(mypath, f))]
    
    result=[]
    
    import pyheif 
    from PIL import Image
    
    #filt_list=['img_2716','img_1890','img_2760']
    
    for source_img_name in onlyfiles:
        #if source_img_name.split('.')[0] not in filt_list:
        #    continue
        print(source_img_name)
    
        source_img_path=f'{mypath}/{source_img_name}'
        item={'path':source_img_path,'found_plate':False,'plate':[]}
        
        if 'heic' in source_img_name:
            heif_file = pyheif.read(source_img_path)
            source_img = Image.frombytes(heif_file.mode, heif_file.size, heif_file.data, "raw", heif_file.mode, heif_file.stride)
            source_img = np.array(source_img)
        else:
            source_img = cv.imread(source_img_path)
            
        if source_img is None:
            result.append(item)
            print("can't read:", source_img_name)
            continue
            
        found=False
        for ang in [0,90]:
            zoom_val=1
            if ang==90:
                source_img=rotate_image(source_img,ang)
                
            item['shape']=source_img.shape
            item['angle']=ang
            orig_width = source_img.shape[1]
            orig_heig = source_img.shape[0]
            
            zoomed_width=None
            zoomed_heig=None
            
            while zoom_val<=6:
                print(f'zoom={zoom_val}, ang={ang}')
                zoomed_heig=int(orig_heig/zoom_val)
                zoomed_width=int(orig_width/zoom_val)

                dx=int((orig_width-zoomed_width)/2)
                dy=int((orig_heig-zoomed_heig)/2)

                zoomed = source_img[dy:dy+zoomed_heig, dx:dx+zoomed_width]
                    
                plateList = detechPlate(zoomed)
                
                min_x=zoomed.shape[1]*0.4
                max_x=zoomed.shape[1]*0.6
                if len(plateList)>0:
                    for x1,y1,x2,y2 in plateList:
                        if (x1>min_x and x2<max_x) or (x2>min_x and x2<max_x) or (x1<min_x and x2>max_x):
                            found=True
                            item['found_plate']=True
                            item['plate'].append([dx+x1,dy+y1,dx+x2,dy+y2])
                            #print(source_img_name)
                            #print(x1,y1,x2,y2)
#                             print(min_x,max_x)
#                             print('found')
#                             break
                if found:
                    break
                            
                zoom_val+=0.5
                continue
            if found:
                break
                
        result.append(item)
        
    import json
    
    with open(mypath.replace("/","_")+".json","w") as f:
        json.dump(result, f)

