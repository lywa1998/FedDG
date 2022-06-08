import os
import SimpleITK as sitk
# import nibabel as nib
import os
import numpy as np
from glob import glob
import time
import shutil
import matplotlib.pyplot as plt
from PIL import Image

def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    
    fft = np.fft.fft2( img_np, axes=(-2, -1) )
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np

client_data_list = ['./Domain1']
client_number = 4

for client_idx in range(client_number):
    for data_idx, each_data_path in enumerate(client_data_list[client_idx]):
        img = Image.open(each_data_path)
        img = img.resize( (384,384), Image.BICUBIC )
        img_np = np.asarray(img)

        amp = extract_amp_spectrum(img_np)
        np.save(f'./client{client_idx+1}/freq_amp_npy/amp_sample{data_idx}', amp)
