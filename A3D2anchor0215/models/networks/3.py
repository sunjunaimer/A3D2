import os
import platform
import subprocess
from glob import glob
from pathlib import Path
import shutil
import threading
import time
import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torch.optim as optim

import numpy as np
import pandas as pd
from PIL import Image


import numpy as np
import torch


import sys
sys.path.append('/data3/sunjun/work/code/TBD/DA22/APViT')



def preprocess_image(image_path):
    # 加载图像
    img = Image.open(image_path)
    
    # 图像预处理
    preprocess = transforms.Compose([
        transforms.Resize((112, 112)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    img = preprocess(img)
    return img            


apvit_dir = '/data3/sunjun/work/code/TBD/DA22/APViTmodel.pth'
model = torch.load(apvit_dir).to('cuda')


# params = list(model.named_parameters())
# for name, param in params:
#     print(name, param.shape)    

# 获取文件夹中的所有图片文件
# img_dir = '/mnt/dolphinfs/hdd_pool/docker/user/hadoop-aipnlp/zhangxinxin33/DA/d2d_data/meld/506_2_train'
# image_files = sorted([os.path.join(img_dir, f) for f in os.listdir(img_dir) if f.endswith('.jpg')])

# features = []
# for image_file in image_files:
#     img = preprocess_image(image_file)  # 从这里开始预处理
#     img = img.to('cuda')
#     img = img.unsqueeze(0) 
#     with torch.no_grad():
#         feat = model.extract_feat(img)
#         output = feat[0]

#     features.append(output.flatten())  

# features = torch.stack(features, dim=0)
# print(features.shape)


class APViT():

    def __init__(self, device):
        self.device = device
        self.model = torch.load(apvit_dir).to(self.device)

    def __call__(self, imgs):
        return self.forward(imgs)

    def forward(self, imgs):
        features = []
        for img in imgs:
            img = img.to(self.device)
            img = img.unsqueeze(0) 
            
            feat = self.model.extract_feat(img)
            output = feat[0]

            features.append(output.flatten())  # 直接使用 PyTorch 的 flatten 方法

        features = torch.stack(features)  # 使用 torch.stack 将列表中的 tensor 堆叠为一个新的 tensor
        return features




class APViT_video(nn.Module):  # 一整个视频一起处理

    def __init__(self, device):
        super(APViT_video, self).__init__()
        
        self.device = device
        self.model = torch.load(apvit_dir).to(self.device)

        for name, param in self.model.named_parameters():
            # if "blocks" in name and not name.startswith(f"blocks.{len(self.model.vit.blocks)-1}") and "head" not in name:
            param.requires_grad = False

    def __call__(self, videos):
        return self.forward(videos)

    def forward(self, videos):
        features = []
        t1 = time.time()
        # for video in videos:
        videos = videos.to(self.device)
        # with torch.no_grad():     
        feat = self.model.extract_feat(videos)
        output = feat[0]
        video_pool = torch.mean(output, dim=0)

        features.append(output)  # 直接使用 PyTorch 的 flatten 方法


        features = torch.stack(features)  # 使用 torch.stack 将列表中的 tensor 堆叠为一个新的 tensor
        t2 = time.time()
        print(f'APViT一个batch的特征维度：{features.shape}, 用时{t2-t1}')
        return output




if __name__ == '__main__':
    model = APViT_video('cuda:0')
    # model = APViT('cuda:0')
    model.eval()
    params = list(model.named_parameters())
    image = '/data3/sunjun/work/code/DA/d2d_data/meld/0_1_test/0000.jpg' #ssh://mygpu/data3/sunjun/work/code/DA/d2d_data
    image2 = '/data3/sunjun/work/code/DA/d2d_data/meld/342_15_train/0001.jpg'
    x = preprocess_image(image)
    z = preprocess_image(image2)
    # x2 = x
    print(x.shape)
    

    y = [x, x, z]
    y = torch.stack(y, dim=0)
    print(y.shape)

    z = torch.unsqueeze(z, dim=0)

    model.eval()
    rz = model(z)
    r2 = model(y)

    print(r2.shape)
    print(r2[:, 0:10])
    print(rz[:, 0:10])
    # print(r2[-1])
    
    # print(rz-r2[-1])



