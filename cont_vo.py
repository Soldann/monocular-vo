import cv2
import numpy as np
import matplotlib.pyplot as plt

class VO:

    def __init__(self, Pi, Xi, Ci, Fi, Ti):
        # The current state of the pipeline
        #   P_i    In state i: the set of all current 2D keypoints
        #   X_i    In state i: the set of all current 3D landmarks
        #   C_i    In state i: the set of candidate 2D keypoints currently
        #          being tracked
        #   F_i    In state i: for each candidate keypoint in C_i, its
        #          position in the first frame it was tracked in
        #   T_i    In state i: the camera pose at the first observation of
        #          each keypoint in C_i
        self.Pi_1 = Pi
        self.Xi_1 = Xi
        self.Ci_1 = Ci
        self.Fi_1 = Fi
        self.Ti_1 = Ti

        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def process_frame(self, img_i_1, img_i):
        # run KLT 
        p_i, tracked, err = cv2.calcOpticalFlowPyrLK(img_i_1, img_i, self.Pi_1.astype(np.float32), None, **self.lk_params)
        tracked = tracked.astype(np.bool_).flatten()
        # Select good points
        if p_i is not None:
            good_new = p_i[tracked]
            good_old = self.Pi_1[tracked]
        print(tracked)
        # new_candidates = (good_old[:, None] == good_new).all(axis=-1).any(axis=1)
        # print(new_candidates)
        fig, ax = plt.subplots()
        ax.imshow(img_i)
        ax.scatter(good_new[:, 0], good_new[:, 1], s=2, c="red", alpha=0.5)
        plt.show()
        
        # update our state vectors
