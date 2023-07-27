import numpy as np
from scipy import ndimage
from PIL import Image
import matplotlib.pyplot as plt
import cv2

def _get_coutour_sample(y_true):
    disc_mask = np.expand_dims(y_true[..., 0], axis=2)
    disc_erosion = ndimage.binary_erosion(disc_mask[..., 0], iterations=1).astype(disc_mask.dtype)
    disc_dilation = ndimage.binary_dilation(disc_mask[..., 0], iterations=5).astype(disc_mask.dtype)
    disc_contour = np.expand_dims(disc_mask[..., 0] - disc_erosion, axis = 2)
    disc_bg = np.expand_dims(disc_dilation - disc_mask[..., 0], axis = 2)
    
    cup_mask = np.expand_dims(y_true[..., 1], axis=2)
    cup_erosion = ndimage.binary_erosion(cup_mask[..., 0], iterations=1).astype(cup_mask.dtype)
    cup_dilation = ndimage.binary_dilation(cup_mask[..., 0], iterations=5).astype(cup_mask.dtype)
    cup_contour = np.expand_dims(cup_mask[..., 0] - cup_erosion, axis = 2)
    cup_bg = np.expand_dims(cup_dilation - cup_mask[..., 0], axis = 2)

    return [disc_contour, disc_bg, cup_contour, cup_bg]

file_path = "./gdrishtiGS_001.png"

img = cv2.imread(file_path)
print(type(img), img.dtype)
# img = img.resize( (384, 384), Image.BICUBIC)
img = np.asarray(img, np.float32)
print(img.ndim)
img = np.mean(img, axis=2, keepdims=True)
print(np.unique(img))
disc = img.copy()
disc[img == 255.] = 0

disc[img == 0] = 0
disc[disc != 0.] = 1
cup = img.copy()
cup[img != 0] = 0
cup[img == 0] = 1
data = np.concatenate( (disc, cup), axis=2)
print(data.shape)
s = cv2.resize(data, (384, 384))
print(s.shape)

disc_contour, disc_bg, cup_contour, cup_bg = _get_coutour_sample(data)

plt.figure()
plt.subplot(131)
plt.imshow(img, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("Mask", fontsize=20)
plt.subplot(132)
plt.imshow(disc, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("Optic Disc", fontsize=20)
plt.subplot(133)
plt.imshow(cup, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("Optic Cup", fontsize=20)
plt.savefig("disc_cup.png")

plt.figure()
plt.subplot(221)
plt.imshow(disc_contour, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("Disc Contour")
plt.subplot(222)
plt.imshow(disc_bg, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("Disc Background")
plt.subplot(223)
plt.imshow(cup_contour, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("Cup Contour")
plt.subplot(224)
plt.imshow(cup_bg, cmap="gray")
plt.xticks([]), plt.yticks([])
plt.title("Cup Background")
plt.savefig('process.png')

plt.show()
