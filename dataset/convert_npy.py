from PIL import Image
import numpy as np

from pathlib import Path

def extract_amp_spectrum(img_np):
    # trg_img is of dimention CxHxW (C = 3 for RGB image and 1 for slice)
    
    fft = np.fft.fft2( img_np, axes=(-2, -1) )
    amp_np, pha_np = np.abs(fft), np.angle(fft)

    return amp_np

client_data_list = [r'D:\Project\FedDG-ELCFS\dataset\Domain1', r'D:\Project\FedDG-ELCFS\dataset\Domain1', \
     r'D:\Project\FedDG-ELCFS\dataset\Domain1', r'D:\Project\FedDG-ELCFS\dataset\Domain1Domian4']
client_number = 4

for client_idx in range(client_number):
    root_path = Path(client_data_list[client_idx])
    if not root_path.exists():
        print('path error')
        break
    data_path = root_path / 'train' / 'image'
    mask_path = root_path / 'train' / 'mask'
    for data_idx, each_data_path in enumerate(data_path.iterdir()):
        img = Image.open(each_data_path)
        img = img.resize( (384,384), Image.BICUBIC )
        img_np = np.asarray(img)

        # mask 
        mask_img_path = mask_path / each_data_path.name
        mask_img = Image.open(mask_img_path)
        mask_img = mask_img.resize( (384, 384), Image.BICUBIC)
        mask_img_np = np.asarray(mask_img)

        out_img = np.concatenate([img_np, mask_img_np], axis=2)

        # amp = extract_amp_spectrum(img_np)
        np.save(f'./client{client_idx+1}/data_npy/sample{data_idx}.npy', out_img)