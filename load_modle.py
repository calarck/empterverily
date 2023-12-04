import torch
import cv2
import numpy as np
from PIL import Image
from torchvision import transforms
import requests
from io import BytesIO


def evalImage(imagePath):
    file =  requests.get(imagePath)
    img = cv2.imdecode(np.fromstring(file.content,np.uint8),1)
    # 读取本地图片
    # img = cv2.imread(imagePath)
    print(img.shape)
    img = torch.from_numpy(np.transpose(img,(2,0,1)))
    img = Image.open(BytesIO(file.content))
    img = transforms.Resize(224)(img)
    img = transforms.CenterCrop(224)(img)
    img = transforms.ToTensor()(img)
    img = transforms.Normalize([0.485,0.456,0.406],[0.229,0.224,0.225])(img)

    outputs = empterModel(img.unsqueeze(0))
    x,predes = torch.max(outputs,1)
    print(class_name[predes])


if __name__ == '__main__':
    empterModel = torch.load("empter_o.pt")
    class_name = ["empter","notempter"]
    evalImage("https://www.hwagain.cn/eagle_img_new/20230520/1659678390130057216_20230520_055214.jpg")