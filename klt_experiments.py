import cv2
from matplotlib import pyplot as plt
from matplotlib.patches import Polygon
import numpy as np
import time

from initialise_vo import DataLoader, Bootstrap
from cont_vo import VO
from utils import find_epipole

# 412
# 120
dl = DataLoader("parking")
b = Bootstrap(dl, init_frames=None)
v = VO(b)
poses = []
fig, ax = plt.subplots()
plt.ion()

# Image centre:
w = dl[0].shape[1]
h = dl[0].shape[0]
c = np.array([w/2, h/2])

w_factor = 10
h_factor = 1
min_w = 0.6
min_h = 0.3

for image in dl[b.init_frames[1]+1:]:
    pnew, _, _, _ = v.process_frame(image, [])
    poses.append(pnew)
    
    if len(poses) >= 2:
        cn_b = find_epipole(poses[-2], poses[-1], b.K)
        
        # find image coord epipole
        uv = dl.K @ cn_b
        epipole = uv[:2]
        
        # The sampling square around the epipole: all in normalised coord
        left_extreme = w_factor * cn_b[0] if w_factor * cn_b[0] < -min_w/2 else -min_w/2
        right_extreme = w_factor * cn_b[0] if w_factor * cn_b[0] > min_w else min_w/2
        lower_extreme = h_factor * cn_b[1] if h_factor * cn_b[1] < -min_h/2 else -min_h/2
        upper_extreme = h_factor * cn_b[1] if h_factor * cn_b[1] > min_h/2 else min_h/2

        ul_bar = np.c_[left_extreme, upper_extreme].T
        ur_bar = np.c_[right_extreme, upper_extreme].T
        lr_bar = np.c_[right_extreme, lower_extreme].T
        ll_bar = np.c_[left_extreme, lower_extreme].T

        ul = (dl.K @ np.vstack((ul_bar, np.c_[1])))[:2].flatten()
        ur = (dl.K @ np.vstack((ur_bar, np.c_[1])))[:2].flatten()
        lr = (dl.K @ np.vstack((lr_bar, np.c_[1])))[:2].flatten()
        ll = (dl.K @ np.vstack((ll_bar, np.c_[1])))[:2].flatten()

        corners = np.array([ll, lr, ur, ul])

    """
    if len(poses) == 2:
        epipole = find_epipole(poses[-2], poses[-1], b.K)
        print(epipole)
        break
    """
    
    if len(poses) == 2:
        p = ax.plot(epipole[0], epipole[1], "go")[0]
        ax.plot(c[0], c[1], "ro")[0]
        ax.imshow(image, cmap="grey")
        rect = Polygon(corners, closed=True, edgecolor="g", facecolor="none",
                       linewidth=3)
        ax.add_patch(rect)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)
    if len(poses) >= 2:
        p.set_data([epipole[0]], [epipole[1]])
        rect.set_xy(corners)
        ax.imshow(image, cmap="grey")
        fig.canvas.draw()
        fig.canvas.flush_events()
    

