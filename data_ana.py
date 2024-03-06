# -*- coding: utf-8 -*-
"""
Created on Sun Jan 10 21:48:48 2021

@author: YaoYee
"""

import os
import xml.etree.cElementTree as et
import numpy as np
import matplotlib.pyplot as plt
from tqdm import tqdm



def area_ratio():
    path = "/home/leon/opt-exprements/expments/data_test/template_match/rubby-test/xml_mixed-up"
    files = os.listdir(path)
    area_list = []
    ratio_list = []

    def file_extension(path):
        return os.path.splitext(path)[1]

    for xmlFile in tqdm(files, desc='Processing'):
        if not os.path.isdir(xmlFile):
            if file_extension(xmlFile) == '.xml':
                tree = et.parse(os.path.join(path, xmlFile))
                root = tree.getroot()
                filename = root.find('filename').text
                # print("--Filename is", xmlFile)

                for Object in root.findall('object'):
                    bndbox = Object.find('bndbox')
                    xmin = bndbox.find('xmin').text
                    ymin = bndbox.find('ymin').text
                    xmax = bndbox.find('xmax').text
                    ymax = bndbox.find('ymax').text

                    area = (int(ymax) - int(ymin)) * (int(xmax) - int(xmin))
                    area_list.append(area)
                    if area> 80*80:
                        print("file is:{} object is {}".format(filename, Object.findall('name')[0].text))

                    ratio = (int(ymax) - int(ymin)) / (int(xmax) - int(xmin))
                    ratio_list.append(ratio)
                    # print("Ratio is", round(ratio,2))

    square_array = np.array(area_list)
    square_max = np.max(square_array)
    square_min = np.min(square_array)
    square_mean = np.mean(square_array)
    square_var = np.var(square_array)
    # square_array = np.square(square_array)
    plt.figure(1)
    # plt.xticks(range(0, square_max+10, 6))
    plt.hist(square_array, 20)#bins=range(square_min, square_max+10, 10))# 20
    # plt.xticks(range(0, square_max, int((square_max - square_min) / 10)))
    plt.xlabel('Area in pixel')
    plt.ylabel('Frequency of area')
    plt.title('Area\n' \
              + 'max=' + str(square_max) + ', min=' + str(square_min) + '\n' \
              + 'mean=' + str(int(square_mean)) + ', var=' + str(int(square_var))
              )

    ratio_array = np.array(ratio_list)
    ratio_max = np.max(ratio_array)
    ratio_min = np.min(ratio_array)
    ratio_mean = np.mean(ratio_array)
    ratio_var = np.var(ratio_array)
    plt.figure(2)
    plt.hist(ratio_array, 20)
    plt.xlabel('Ratio of width / height')
    plt.ylabel('Frequency of ratio')
    plt.title('Ratio\n' \
              + 'max=' + str(round(ratio_max, 2)) + ', min=' + str(round(ratio_min, 2)) + '\n' \
              + 'mean=' + str(round(ratio_mean, 2)) + ', var=' + str(round(ratio_var, 2))
              )
    plt.show()

def main():
    area_ratio()

if __name__ == "__main__":
    main()