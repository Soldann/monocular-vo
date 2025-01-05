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
dl = DataLoader("kitti")
b = Bootstrap(dl, init_frames=(410, 412))
v = VO(b)
poses = []
baseline_vecs = []
fig, ax = plt.subplots()
plt.ion()
w = dl[0].shape[1]
h = dl[0].shape[0]
c = np.array([w/2, h/2])
"""
for image in dl[b.init_frames[1]+1:]:
    pnew, _, _, _, extremes = v.process_frame(image, [])
    poses.append(pnew)

    corners = np.array([[extremes[0, 0], extremes[0, 1], extremes[0, 1], extremes[0, 0]],
                        [extremes[1, 1], extremes[1, 1], extremes[1, 0], extremes[1, 0]]]).T
    if len(poses) == 2:
        ax.plot(c[0], c[1], "ro")[0]
        ax.imshow(image, cmap="grey")
        rect = Polygon(corners, closed=True, edgecolor="g", facecolor="none",
                       linewidth=3)
        ax.add_patch(rect)

        fig.canvas.draw()
        fig.canvas.flush_events()
        plt.show(block=False)
    if len(poses) >= 2:
        rect.set_xy(corners)
        ax.imshow(image, cmap="grey")
        fig.canvas.draw()
        fig.canvas.flush_events()
"""
# Image centre:
w = dl[0].shape[1]
h = dl[0].shape[0]
c = np.array([w/2, h/2])


w_factor = 5
h_factor = 1

w_pfactor1 = 100
w_pfactor2 = 0.005
v_pfactor1 = 1
min_w = 0.6
min_h = 0.3

min_pix_w = 0.28 * w
min_pix_h = 0.28 * h

for image in dl[b.init_frames[1]+1:]:
    pnew, _, _, _, _ = v.process_frame(image, [])
    poses.append(pnew)
    
    if len(poses) >= 2:
        cn_b, rot_vec = find_epipole(poses[-2], poses[-1], b.K)
        baseline_vecs.append(rot_vec)
        turning_mode = None

        if len(baseline_vecs) >= 3: 
            
            # cn_b = np.mean(np.vstack([b for b in baseline_vecs[-2:]]), axis=0)
            if all([baseline_vecs[-1][1] >= 0.028,
                   baseline_vecs[-2][1] >= 0.028,
                   baseline_vecs[-3][1] >= 0.028]):
                turning_mode = "left"
            elif all([baseline_vecs[-1][1] <= -0.028,
                     baseline_vecs[-2][1] <= -0.028,
                     baseline_vecs[-3][1] <= -0.028]):
                turning_mode = "right"
            else:
                turning_mode = None
        
        # find image coord epipole
        uv = dl.K @ cn_b
        epipole = uv[:2]

        # Polynomial for dampening
        def poly_w(dev: float):
            """
            Deviation dev from the image centre in normalised coordinates
            """
            global w_pfactor1, w_pfactor2
            return w_pfactor1 * dev**2 + w_pfactor2
        
        def poly_v(dev: float):
            """
            Deviation dev from the image centre in normalised coordinates
            """
            global v_pfactor1 
            return v_pfactor1 * dev**2
        
        """
        # The sampling square around the epipole: all in normalised coord
        left_extreme = poly_w(cn_b[0]) if poly_w(cn_b[0]) < -min_w/2 else -min_w/2
        right_extreme = poly_w(cn_b[0]) if poly_w(cn_b[0]) > min_w else min_w/2
        lower_extreme = poly_v(cn_b[1]) if poly_v(cn_b[1]) < -min_h/2 else -min_h/2
        upper_extreme = poly_v(cn_b[1]) if poly_v(cn_b[1]) > min_h/2 else min_h/2
        
        ul_bar = np.c_[left_extreme, upper_extreme].T
        ur_bar = np.c_[right_extreme, upper_extreme].T
        lr_bar = np.c_[right_extreme, lower_extreme].T
        ll_bar = np.c_[left_extreme, lower_extreme].T

        ul = (dl.K @ np.vstack((ul_bar, np.c_[1])))[:2].flatten()
        ur = (dl.K @ np.vstack((ur_bar, np.c_[1])))[:2].flatten()
        lr = (dl.K @ np.vstack((lr_bar, np.c_[1])))[:2].flatten()
        ll = (dl.K @ np.vstack((ll_bar, np.c_[1])))[:2].flatten()
        
        """

        # Sampling square based on turning mode
        if turning_mode is None:
            left_extreme = c[0] - min_pix_w
            right_extreme = c[0] + min_pix_w
            lower_extreme = c[1] - min_pix_h
            upper_extreme = c[1] + min_pix_h

        elif turning_mode == "left":
            left_extreme = 0
            right_extreme = c[0] + min_pix_w /2
            lower_extreme = c[1] - min_pix_h
            upper_extreme = c[1] + min_pix_h

        elif turning_mode == "right":
            left_extreme = c[0] - min_pix_w / 2
            right_extreme = w
            lower_extreme = c[1] - min_pix_h
            upper_extreme = c[1] + min_pix_h

        ul = np.r_[left_extreme, upper_extreme]
        lr = np.r_[right_extreme, lower_extreme]
        ur = np.r_[right_extreme, upper_extreme]
        ll = np.r_[left_extreme, lower_extreme]

        corners = np.array([ll, lr, ur, ul])

    
    #if len(poses) == 2:
    #    epipole = find_epipole(poses[-2], poses[-1], b.K)
    #    print(epipole)
    #    break
    
    
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
    