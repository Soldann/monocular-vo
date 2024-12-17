import cv2
import numpy as np
import matplotlib.pyplot as plt

class VO:

    def __init__(self, K, Pi, Xi, Ci, Pose, max_keypoints=100):
        # The current state of the pipeline
        #   K      Intrinsic matrix of camera
        #
        #   P_i    In state i: the set of all current 2D keypoints
        #   X_i    In state i: the set of all current 3D landmarks
        #   C_i    In state i: the set of candidate 2D keypoints currently
        #          being tracked
        #   F_i    In state i: for each candidate keypoint in C_i, its
        #          position in the first frame it was tracked in
        #   T_i    In state i: the camera pose at the first observation of
        #          each keypoint in C_i
        self.K = K
        self.distortion_coefficients = None

        self.Pi_1 = Pi
        self.Xi_1 = Xi
        self.Ci_1 = Ci
        self.Fi_1 = Ci
        self.Ti_1 = None

        self.max_keypoints = max_keypoints

        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))

    def run_KLT(self, img_i_1, img_i, points_to_track, name_of_feature="features", debug=False):
        """
            Wrapper function for CV2 KLT
        """
        tracked_points, tracked, err = cv2.calcOpticalFlowPyrLK(img_i_1, img_i, points_to_track.astype(np.float32), None, **self.lk_params)
        tracked = tracked.astype(np.bool_).flatten()
        # Select good points
        if tracked_points is not None:
            good_tracked_points = tracked_points[tracked]
            good_previous_points = points_to_track[tracked]
        
        if debug:
            fig, (ax1, ax2) = plt.subplots(1,2)
            ax1.imshow(img_i_1, cmap="grey")
            ax2.imshow(img_i, cmap="grey")

            for i, (new, old) in enumerate(zip(good_tracked_points, good_previous_points)):
                a, b = new.ravel()
                c, d = old.ravel()
                # cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                ax1.plot((a, c), (b, d), '-', linewidth=2, c="red")
                ax2.plot((a, c), (b, d), '-', linewidth=2, c="green")
                # cv2.circle(img_i, (int(a), int(b)), 5, (0, 255, 0), -1)
            
            ax1.scatter(good_previous_points[:, 0], good_previous_points[:, 1], s=5, c="green", alpha=0.5)
            ax2.scatter(good_tracked_points[:, 0], good_tracked_points[:, 1], s=5, c="red", alpha=0.5)
            ax1.set_title(f"KLT on {name_of_feature} in Old Image")
            ax2.set_title(f"KLT on {name_of_feature} in New Image")
            plt.show()
        
        return good_tracked_points, tracked


    def process_frame(self, img_i_1, img_i, debug=False):
        # Step 1: run KLT  on the points P
        P_i, P_i_tracked = self.run_KLT(img_i_1, img_i, self.Pi_1, "P", debug)

        # Step 2: Run PnP to get pose for the new frame
        
        success, rvec, tvec = cv2.solvePnP(self.Xi_1[P_i_tracked], self.Pi_1, self.K, self.distortion_coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
        # TODO: tvec is given as the position of the previous origin in the new camera frame
        rotation_matrix, _ = cv2.Rodrigues(rvec)
        transformation_i_1_to_i = np.column_stack((rotation_matrix, tvec))
        transformation_world_to_i = 

        # Step 3: Run KLT on candidate keypoints
        C_i, C_i_tracked = self.run_KLT(img_i_1, img_i, self.Ci_1, "C", debug)

        # Step 4: Calculate angles between each tracked C_i

        # TODO: Is there some way to do this without triangulation? 
        triangulated_points = 
        # projectionMat1 = self.K @ np.column_stack((np.identity(3), np.zeros(3)))
        # projectionMat2 = self.K @ np.column_stack((self.R, self.t))

        # self.triangulated_points = cv2.triangulatePoints(projectionMat1, projectionMat2, 
        #                                                  points1[ransac_inliers].T, points2[ransac_inliers].T)

        
        
        # update our state vectors
