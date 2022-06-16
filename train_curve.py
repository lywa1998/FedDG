import  numpy as np
import matplotlib.pyplot as plt

def smooth(arr: np.ndarray, factor: float) -> np.ndarray:
    """
    Tensorboard 平滑算法
    https://blog.csdn.net/qq_40939814/article/details/106149166
    """
    weight = factor
    for ii in range(1, len(arr)):
        arr[ii] = (arr[ii - 1] * weight + arr[ii]) / (weight + 1)
        weight = (weight + 1) * factor
    return arr

OD = []
OC = []
with open("evaluation_result.txt", "r") as f:
    data = f.readlines()
for ll in data:
    ll = ll.strip()
    ll = ll.split(",")[0]
    if ll.startswith("OD"):
        OD.append(eval(ll.split(":")[1]))
    else:
        OC.append(eval(ll.split(":")[1]))

OD = np.asarray(OD)
OC = np.asarray(OC)
OD = smooth(OD, 0.6)
OC = smooth(OC, 0.6)

plt.figure()
plt.plot(OD, label='Optic Disc')
plt.plot(OC, label='Optic Cup')
plt.xlabel("Epoch", fontsize=20)
plt.ylabel("Ratio", fontsize=20)
plt.grid()
plt.title("Dice Coefficient", fontsize=20)
plt.legend(fontsize=20, loc='lower right')
plt.show()