import os

import cv2
import torch
from PIL import Image
from scipy.spatial.distance import euclidean

import models
from torchvision import transforms
from util.FeatureExtractor import FeatureExtractor
import numpy as np

from sklearn.preprocessing import normalize
from util.utils import img_to_tensor, read_image, dtw


def pool2d(tensor, type='max'):
    sz = tensor.size()
    if type == 'max':
        x = torch.nn.functional.max_pool2d(tensor, kernel_size=(int(sz[2] / 8), int(sz[3])))
    if type == 'mean':
        x = torch.nn.functional.mean_pool2d(tensor, kernel_size=(sz[2] / 8, sz[3]))
    x = x[0].cpu().data.numpy()
    x = np.transpose(x, (2, 1, 0))[0]
    return x


class AlignedReIDModel:

    def __init__(self, ser_model_path, cuda_device=0):
        os.environ['CUDA_VISIBLE_DEVICES'] = str(cuda_device)
        self.use_gpu = torch.cuda.is_available()
        self.model = models.init_model(name='resnet50', num_classes=751, loss={'softmax', 'metric'},
                                       use_gpu=self.use_gpu, aligned=True)
        checkpoint = torch.load(ser_model_path)
        self.model.load_state_dict(checkpoint['state_dict'])

        if self.use_gpu:
            self.model = self.model.cuda()

        self.img_transform = transforms.Compose([
            transforms.Resize((256, 128)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        exact_list = ['7']
        self.myexactor = FeatureExtractor(self.model, exact_list)

    def get_embeddings(self, img, img_type="opencv"):

        if img_type == "opencv":
            img = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))

        img_tensor = img_to_tensor(img, self.img_transform)
        if self.use_gpu:
            img_tensor = img_tensor.cuda()
        self.model.eval()
        f1 = self.myexactor(img_tensor)
        a1 = normalize(pool2d(f1[0], type='max'))

        emb = a1.reshape(-1)

        return emb

    def get_distance_from_emb(self, emb1, emb2):
        a1 = emb1.reshape((8, 2048))
        a2 = emb2.reshape((8, 2048))

        return self.get_distance_from_emb_matrix(a1, a2)

    def get_distance_from_emb_matrix(self, a1, a2):
        dist = np.zeros((8, 8))
        for i in range(8):
            temp_feat1 = a1[i]
            for j in range(8):
                temp_feat2 = a2[j]
                dist[i][j] = euclidean(temp_feat1, temp_feat2)
        d, D, sp = dtw(dist)

        return d

    def get_distance_from_pil_imgs(self, img1, img2):
        # PIL images in RGB

        img1 = img_to_tensor(img1, self.img_transform)
        img2 = img_to_tensor(img2, self.img_transform)
        if self.use_gpu:
            model = self.model.cuda()
            img1 = img1.cuda()
            img2 = img2.cuda()
        else:
            model = self.model
        model.eval()
        f1 = self.myexactor(img1)
        f2 = self.myexactor(img2)
        a1 = normalize(pool2d(f1[0], type='max'))
        a2 = normalize(pool2d(f2[0], type='max'))
        dist = np.zeros((8, 8))
        for i in range(8):
            temp_feat1 = a1[i]
            for j in range(8):
                temp_feat2 = a2[j]
                dist[i][j] = euclidean(temp_feat1, temp_feat2)
        d, D, sp = dtw(dist)

        return d

    def read_img_from_path(self, img_pathh):
        return read_image(img_pathh)

    def get_distance_from_paths(self, img_path1, img_path2):
        img1 = read_image(img_path1)
        img2 = read_image(img_path2)

        return self.get_distance_from_pil_imgs(img1, img2)
