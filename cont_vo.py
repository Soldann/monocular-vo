import cv2
import numpy as np
import matplotlib.pyplot as plt

class VO:

    def __init__(self, bootstrap_obj):
        """
        Visual Odometry pipeline.
        
        SLAM continues from the second frame used for bootstrapping.

        ### Parameters
        1. bootstra_obj : Bootstrap
            - Instance of bootstrap giving the initial point cloud of land-
              marks and the positions of the corresponding keypoints
        """

        # The current state of the pipeline
        #   P_i    In state i: the set of all current 2D keypoints
        #   X_i    In state i: the set of all current 3D landmarks
        #   C_i    In state i: the set of candidate 2D keypoints currently
        #          being tracked
        #   F_i    In state i: for each candidate keypoint in C_i, its
        #          position in the first frame it was tracked in
        #   T_i    In state i: the camera pose at the first observation of
        #          each keypoint in C_i

        # The previous state i-1 of the pipeline:
        self.Pi_1 = None
        self.Xi_1 = None
        self.Ci_1 = None
        self.Fi_1 = None
        self.Ti_1 = None

        self.img_i_1 = None
        self.i = 0              # state number

        # The current (new) state i of the pipeline
        self.Pi = None
        self.Xi = None

        self.img_i = None

        # Setting information from bootstrapping
        self.dl = bootstrap_obj.data_loader                              # data loader
        self.Pi_1, self.Xi_1 = bootstrap_obj.get_points()       # landmarks, keypoints
        self.i = bootstrap_obj.init_frames[-1]                  # state counter
        last_bootstrap_img_path = self.dl.all_im_paths[self.i]  # setting last image
        self.img_i = cv2.imread(last_bootstrap_img_path.as_posix(), 
                                cv2.IMREAD_GRAYSCALE)

        # Parameters Lucas Kanade
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


    def next_image(self):
        """
        Load the next image making it self.img_i, make the last image self.img_i_1,
        discard the image from before that, advance state counter
        """

        self.i += 1                                         # advance state counter
        next_image_path = self.dl.all_im_paths[self.i]
        self.img_i_1 = self.img_i
        self.img_i = cv2.imread(next_image_path.as_posix(), 
                                cv2.IMREAD_GRAYSCALE)

    def track_keypoints(self):
        """
        Call after next_image(). Modifies self.Pi and self.Pi_1
        """

        # run KLT 
        p_i, tracked, err = cv2.calcOpticalFlowPyrLK(self.img_i_1, self.img_i, 
                                                     self.Pi_1.astype(np.float32), 
                                                     None, **self.lk_params)
        tracked = tracked.astype(np.bool_).flatten()
        # Select good points
        if p_i is not None:
            good_new = p_i[tracked]
            good_old = self.Pi_1[tracked]
        # print(tracked)
        # new_candidates = (good_old[:, None] == good_new).all(axis=-1).any(axis=1)
        # print(new_candidates)

        self.Pi = good_new
        self.Pi_1 = good_old

        return good_new

        fig, ax = plt.subplots()
        ax.imshow(img_i)
        ax.scatter(good_new[:, 0], good_new[:, 1], s=2, c="red", alpha=0.5)
        plt.show()
        
        # update our state vectors

    def draw_keypoint_tracking(self):
        """
        Call after track_features. Visualises the features tracks over the
        last image pair
        """
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.imshow(self.img_i, cmap="gray")
        u_coord = np.column_stack((self.Pi_1[:, 0], self.Pi[:, 0])) 
        v_coord = np.column_stack((self.Pi_1[:, 1], self.Pi[:, 1]))
        ax.scatter(self.Pi_1[:, 0], self.Pi_1[:, 1], marker="o", s=3, alpha=0.6)
        ax.scatter(self.Pi[:, 0], self.Pi[:, 1], marker="o", s=3, alpha=0.6)
        ax.plot(u_coord.T, v_coord.T, "-", linewidth=2, alpha=0.9, c="r")
        ax.set_xlim((0, self.img_i.shape[-1]))
        ax.set_ylim((self.img_i.shape[0], 0))
        plt.show(block=True)
