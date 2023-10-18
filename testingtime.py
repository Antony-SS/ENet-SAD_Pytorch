import json
import os
import cv2
import numpy as np
import torch
from torch.utils.data import Dataset
from fuzzylab import *
from fuzzy_algocompare import *
import time

def process_images(folder = "./TUSimple/train_set/clips/0601/1494453427635226083/"):
        cannycontrol = 0
        # threshold = self.cannythreshold
        threshold = 1
        start_time = time.time()
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
        # self.cannythreshold = threshold
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
        end = time.time()

        print("execution time: ", end - start_time)
        return output
    
process_images()