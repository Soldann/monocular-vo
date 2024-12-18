import cv2
import numpy as np
from initialise_vo import Bootstrap
import matplotlib.pyplot as plt

class VO:

    def __init__(self, bootstrap_obj: Bootstrap, max_keypoints=100):
        """
        Visual Odometry pipeline.
        
        Initialize VO using a bootstrap object. 
        SLAM continues from the second frame used for bootstrapping.

        ### Parameters
        1. bootstra_obj : Bootstrap
            - Instance of bootstrap giving the initial point cloud of land-
              marks and the positions of the corresponding keypoints
        """

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

        self.K = bootstrap_obj.data_loader.K
        self.distortion_coefficients = None

        self.Pi_1, self.Xi_1, self.Ci_1, initial_pose = bootstrap_obj.get_points()       # landmarks, keypoints
        self.F_i = self.Ci_1.copy()

        self.Ti_1 = np.tile(initial_pose.reshape(-1), (len(self.Ci_1), 1))

        self.max_keypoints = max_keypoints

        # Setting information from bootstrapping
        self.dl = bootstrap_obj.data_loader                              # data loader
        self.img_i = self.dl[bootstrap_obj.init_frames[-1]]

        # Parameters Lucas Kanade
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
        
        success, r_cw, c_t_cw = cv2.solvePnP(self.Xi_1[P_i_tracked], self.Pi_1, self.K, self.distortion_coefficients, flags=cv2.SOLVEPNP_ITERATIVE)
        # TODO: c_t_cw is the vector from camera frame to world frame, in the camera coordinates
        R_cw, _ = cv2.Rodrigues(r_cw) # rotation vector world to camera frame
        transformation_i_1_to_i = np.column_stack((R_cw, c_t_cw))

        # Step 3: Run KLT on candidate keypoints
        C_i, C_i_tracked = self.run_KLT(img_i_1, img_i, self.Ci_1, "C", debug)

        # Step 4: Calculate angles between each tracked C_i
        for candidate_i, candidate_i_1 in zip(C_i, self.Ci_1[C_i_tracked]):
            vector_to_candidate_i = np.linalg.inv(self.K) @ np.append(candidate_i, 1)[:, None]
            angle_between_baseline_and_point_i = np.arccos(
                                                            np.dot(c_t_cw.reshape(-1), vector_to_candidate_i.reshape(-1)) / 
                                                            (np.linalg.norm(c_t_cw) * np.linalg.norm(vector_to_candidate_i))
                                                            )
            vector_to_candidate_i_1 = np.linalg.inv(self.K) @ np.append(candidate_i_1, 1)[:, None]
            w_t_wc = - np.linalg.inv(R_cw) @ c_t_cw
            angle_between_baseline_and_point_i_1 = np.arccos(
                                                            np.dot(w_t_wc.reshape(-1), vector_to_candidate_i_1.reshape(-1)) / 
                                                            (np.linalg.norm(w_t_wc) * np.linalg.norm(vector_to_candidate_i_1))
                                                            )
            
            angle_between_points = np.pi - angle_between_baseline_and_point_i_1 - angle_between_baseline_and_point_i
            print(angle_between_points)
            if debug:
                fig, (ax1, ax2) = plt.subplots(1,2)
                ax1.imshow(img_i_1, cmap="grey")
                ax2.imshow(img_i, cmap="grey")

                a, b = candidate_i.ravel()
                c, d = candidate_i_1.ravel()
                # cv2.line(mask, (int(a), int(b)), (int(c), int(d)), (0, 255, 0), 2)
                ax1.plot((a, c), (b, d), '-', linewidth=2, c="red")
                ax2.plot((a, c), (b, d), '-', linewidth=2, c="green")
                # cv2.circle(img_i, (int(a), int(b)), 5, (0, 255, 0), -1)
                
                ax1.scatter(c, d, s=5, c="green", alpha=0.5)
                ax2.scatter(a, b, s=5, c="red", alpha=0.5)
                ax1.set_title(f"C_i in Old Image")
                ax2.set_title(f"C_i in New Image")
                plt.show()

                """
                # Here is some code that does the same as above but using triangulation for the algorithm
                # You can use it to verify if the above works

                projectionMat1 = self.K @ np.column_stack((np.identity(3), np.zeros(3)))
                projectionMat2 = self.K @ np.column_stack((R_cw, c_t_cw))

                triangulated_point = cv2.triangulatePoints(projectionMat1, projectionMat2, candidate_i_1, candidate_i)
                triangulated_point /= triangulated_point[3]
                triangulated_point = triangulated_point[:3]

                triangulated_point_in_i = R_cw @ triangulated_point + c_t_cw
                ci_1_in_i = R_cw @ vector_to_candidate_i_1 + c_t_cw
                
                triangulated_point_to_ci_1 = ci_1_in_i - triangulated_point_in_i
                triangulated_point_to_ci = vector_to_candidate_i - triangulated_point_in_i

                angle_between_points_with_triangulation = np.arccos(
                        np.dot(triangulated_point_to_ci.reshape(-1), triangulated_point_to_ci_1.reshape(-1)) / 
                        (np.linalg.norm(triangulated_point_to_ci) * np.linalg.norm(triangulated_point_to_ci_1))
                )
                """

        # Step 5: Add candidates that match thresholds to sets
            
        # Step 6: 
            
            
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
