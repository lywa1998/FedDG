from PIL import Image
import cv2
import sys
from unittest import main
import numpy as np
from pathlib import Path


dic_name = ['Domain1', 'Domain2', 'Domain3', 'Domain4']
img_name = ['train/ROIs/image', 'test/ROIs/image']
mask_name = ['train/ROIs/mask', 'test/ROIs/mask']

def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    
    fft = np.fft.fft2( img_np, axes=(-2, -1) )
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np


if __name__ == "__main__":
    path = Path('D:/Project/FedDG-ELCFS/dataset')
    for ii, domain in enumerate(dic_name):
        dic_path = path / domain
        output_path = path / f"client{ii}"
        if not output_path.exists():
            output_path.mkdir()
        
        jj = 0
        for img_file, mask_file in zip(img_name, mask_name):
            img_path = dic_path / img_file
            mask_path = dic_path / mask_file

            for img_file_path, mask_file_path in zip(img_path.iterdir(), mask_path.iterdir()):
                print(f"current file {img_file_path.name}, {mask_file_path.name}")
                if img_file_path.stem != mask_file_path.stem:
                    sys.exit(-1)
                img = cv2.imread(str(img_file_path))
                mask = cv2.imread(str(mask_file_path))
                print(img_file_path, mask_file_path)

                img = np.asarray(img, np.float32)
                mask = np.asarray(mask, np.float32)
                if mask.ndim == 3:  # expanda dimension for concatenate
                    mask = np.mean(mask, axis=2)
                
                disc = mask.copy()  # Disc
                disc[mask == 255.] = 0
                disc[mask == 128.] = 1
                disc[mask == 0] = 0

                cup = mask.copy()  # Cup
                cup[mask == 255.] = 0
                cup[mask == 128.] = 0
                cup[mask == 0] = 1

                img = cv2.resize(img, (384, 384))
                disc = cv2.resize(disc, (384, 384))
                cup = cv2.resize(cup, (384, 384))
                disc = disc[..., np.newaxis]
                cup = cup[..., np.newaxis]
                
                print(img.shape, disc.shape, cup.shape)
                data = np.concatenate( (img, disc, cup), axis=2)

                # data_npy
                data_npy_path = output_path / 'data_npy'
                if not data_npy_path.exists():
                    data_npy_path.mkdir()
                data_npy_name = data_npy_path / f"sample{jj}"
                np.save(data_npy_name, data)

                # freq_amp_npy
                img = img.transpose((2, 0, 1))

                data = extract_amp_spectrum(img)

                data_npy_path = output_path / 'freq_amp_npy'
                if not data_npy_path.exists():
                    data_npy_path.mkdir()
                data_npy_name = data_npy_path / f"sample{jj}"
                np.save(data_npy_name, data)
                jj += 1
    
