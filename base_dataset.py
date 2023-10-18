import os.path as osp
import os
import numpy as np
import cv2
import torch
from torch.utils.data import Dataset
import torchvision
import utils.transforms as tf
from .registry import DATASETS
from fuzzy_algocompare import *
from fuzzylab import *

@DATASETS.register_module
class BaseDataset(Dataset):
    def __init__(self, img_path, data_list, list_path='list', cfg=None):
        self.cfg = cfg
        self.img_path = img_path
        self.list_path = osp.join(img_path, list_path)
        self.data_list = data_list
        self.is_training = ('train' in data_list)

        self.img_name_list = []
        self.full_img_path_list = []
        self.label_list = []
        self.exist_list = []
        self.cannythreshold = 1

        self.transform = self.transform_train() if self.is_training else self.transform_val()

        self.init()

    def transform_train(self):
        raise NotImplementedError()

    def transform_val(self):
        val_transform = torchvision.transforms.Compose([
            tf.SampleResize((self.cfg.img_width, self.cfg.img_height)),
            tf.GroupNormalize(mean=(self.cfg.img_norm['mean'], (0, )), std=(
                self.cfg.img_norm['std'], (1, ))),
        ])
        return val_transform

    def view(self, img, coords, file_path=None):
        for coord in coords:
            for x, y in coord:
                if x <= 0 or y <= 0:
                    continue
                x, y = int(x), int(y)
                cv2.circle(img, (x, y), 4, (255, 0, 0), 2)

        if file_path is not None:
            if not os.path.exists(osp.dirname(file_path)):
                os.makedirs(osp.dirname(file_path))
            cv2.imwrite(file_path, img)


    def init(self):
        raise NotImplementedError()

    def process_images(self,folder):
        cannycontrol = 0
        # threshold = self.cannythreshold
        threshold = 1
        for index in range(20):
            filename = folder + str(index+1) + '.jpg'
            img = cv2.imread(filename)
            out = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            height, width = out.shape
            y_intercept = int(1 / 4 * height)
            x_intercept = int(width / 2)
            filtered = cv2.bilateralFilter(out, 7, 25, 50)
            # print(items)
            # print('threshold before add = ', threshold)
            threshold = threshold + cannycontrol
            # print('threshold after add = ',threshold)
            if threshold < 0:
                threshold = 1
            high = threshold
            low = high / 3
            edge = cv2.Canny(filtered, low, high, None, 3)
            edge = np.uint8(edge)

            myROI = np.array([[(x_intercept, y_intercept), (0, height - 1), (width - 1, height - 1)]],
                             dtype=np.int32)  # 30->10
            mask = np.zeros_like(edge)
            region = cv2.fillPoly(mask, myROI, 255)
            roi = cv2.bitwise_and(edge, region)
            lines = cv2.HoughLines(roi, 1, np.pi / 180, 3, None, 0, 0)
            if lines is not None:
                rhoall = lines[:, :, 0]
                thetaall = lines[:, :, 1]
                totalinesall = len(rhoall)
            else:
                totalinesall = 0
            if totalinesall > 58000:
                totalinesall = 58000
            cannycontrol = fuzzy_canny(totalinesall)
            # print('canny control = ',cannycontrol)
        self.cannythreshold = threshold
        #img[:, :, 0] = edge
        #img[:, :, 1] = edge
        #img[:, :, 2] = edge
        min_val = np.min(edge)
        max_val = np.max(edge)
        normalized_edge = (edge - min_val) / (max_val - min_val)
        scaled_edge = (normalized_edge * 255).astype(np.uint8)
        # expand_edge = np.expand_dims(scaled_edge, axis=2)
        # output = np.concatenate((img, expand_edge), axis=2)
        img[:, :, 0] = scaled_edge
        img[:, :, 2] = scaled_edge

        output = img

        return output

    def __len__(self):
        return len(self.full_img_path_list)

    def __getitem__(self, idx):
        image_name = self.full_img_path_list[idx]
        #print(image_name)
        folder = image_name[:-6]
        img = self.process_images(folder)
        #img = cv2.imread(image_name).astype(np.float32)
        img = img[self.cfg.cut_height:, :, :]

        if self.is_training:
            label = cv2.imread(self.label_list[idx], cv2.IMREAD_UNCHANGED)
            if len(label.shape) > 2:
                label = label[:, :, 0]
            label = label.squeeze()
            label = label[self.cfg.cut_height:, :]
            exist = self.exist_list[idx]
            if self.transform:
                img, label = self.transform((img, label))
            label = torch.from_numpy(label).contiguous().long()
        else:
            img, = self.transform((img,))

        img = torch.from_numpy(img).permute(2, 0, 1).contiguous().float()
        meta = {'full_img_path': self.full_img_path_list[idx],
                'img_name': self.img_name_list[idx]}

        data = {'img': img, 'meta': meta}
        if self.is_training:
            data.update({'label': label, 'exist': exist})
        return data
