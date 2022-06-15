import cv2
import matplotlib.pyplot as plt

img = cv2.imread('dataset/Domain1/train/image/gdrishtiGS_002.png')
imgray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
ret, thresh = cv2.threshold(imgray, 50, 255, 0)
contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
cv2.drawContours(img, contours, -1, (0,255,0), 3)
plt.imshow(img)
plt.savefig('cnt.png')
pass