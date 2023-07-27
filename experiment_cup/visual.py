import argparse
from pathlib import Path
import numpy as np
import cv2
import matplotlib.pyplot as plt

parser = argparse.ArgumentParser()
parser.add_argument('--file_path', type=str, default="../output/xxxx/prediction")
parser.add_argument('--sample', type=int, default=0)
args = parser.parse_args()
args = vars( args )


file = Path(args['file_path'])
plt.figure()
# origin image
nn = f"3_sample{args['sample']}.npy_img.npy"
fn = file / nn
print(fn)
img = np.load(fn)
img = img.transpose((1, 2, 0))
img = img.astype(np.uint8)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
plt.subplot(131)
plt.imshow(img)
plt.xticks([]), plt.yticks([])
plt.title("origin")

# prediction
nn = f"3_sample{args['sample']}.npy_pred.npy"
fn = file / nn
print(fn)
data = np.load(fn)
data = data.transpose((1, 2, 0))
mask = np.squeeze(data[..., 0])
mask = mask.astype(np.uint8)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # extract contours
img_pred = img.copy()
cv2.drawContours(img_pred, contours, -1, (0, 255, 0), 3)
plt.subplot(132)
plt.imshow(img_pred)
plt.xticks([]), plt.yticks([])
plt.title("prediction")

# truth
nn = f"3_sample{args['sample']}.npy_gth.npy"
fn = file / nn
print(fn)
data = np.load(fn)
data = data.transpose((1, 2, 0))
mask = np.squeeze(data[..., 0])
mask = mask.astype(np.uint8)
contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)  # extract contours
img_gth = img.copy()
cv2.drawContours(img_gth, contours, -1, (0, 255, 0), 3)
plt.subplot(133)
plt.imshow(img_gth)
plt.xticks([]), plt.yticks([])
plt.title("truth")

plt.savefig(f"test-{args['sample']}.png")
