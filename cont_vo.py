import cv2
import numpy as np
from initialise_vo import Bootstrap
import matplotlib.pyplot as plt
from utils import inverse_transformation, multiply_transformation

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

        self.Pi_1, self.Xi_1, self.Ci_1, self.T_Ci_1__w = bootstrap_obj.get_points() # landmarks, keypoints, candidate points, transform world to camera position i-1
        self.Fi_1 = self.Ci_1.copy()

        self.Ti_1 = np.tile(self.T_Ci_1__w.reshape(-1), (len(self.Ci_1), 1))
        
        self.max_keypoints = max_keypoints

        # Setting information from bootstrapping
        self.dl = bootstrap_obj.data_loader                              # data loader
        self.img_i_1 = self.dl[bootstrap_obj.init_frames[-1]]

        # Parameters Lucas Kanade
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 0,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # angle threshold
        self.angle_threshold = 0.09991679144388552 # Assuming baseline is 10% of the depth


    def run_KLT(self, img_i_1, img_i, points_to_track, name_of_feature="features", debug=False):
        """
            Wrapper function for CV2 KLT
        """
        tracked_points, tracked, err = cv2.calcOpticalFlowPyrLK(img_i_1, img_i, points_to_track.astype(np.float32), None, **self.lk_params)
        tracked = tracked.astype(np.bool_).flatten()
        
        if debug:
            # Select good points
            if tracked_points is not None:
                good_tracked_points = tracked_points[tracked]
                good_previous_points = points_to_track[tracked]

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
        
        return tracked_points, tracked


    def process_frame(self, img_i, debug=False):
        # Step 1: run KLT  on the points P
        P_i, P_i_tracked = self.run_KLT(self.img_i_1, img_i, self.Pi_1, "P")
        
        self.Pi_1 = P_i[P_i_tracked] # Update Pi with the points we successfully tracked
        self.Xi_1 = self.Xi_1[P_i_tracked] # Update Xi with the ones we successfully tracked

        # Step 2: Run PnP to get pose for the new frame
        
        success, r_cw, c_t_cw, ransac_inliers = cv2.solvePnPRansac(self.Xi_1, self.Pi_1, self.K, self.distortion_coefficients, flags=cv2.SOLVEPNP_EPNP) # Note that Pi_1 and Xi_1 are actually for Pi and Xi, since we updated them above
        ransac_inliers = ransac_inliers.flatten()
        self.Pi_1 = self.Pi_1[ransac_inliers] # Update Pi with ransac
        self.Xi_1 = self.Xi_1[ransac_inliers] # Update Xi with ransac

        # TODO: c_t_cw is the vector from camera frame to world frame, in the camera coordinates
        R_cw, _ = cv2.Rodrigues(r_cw) # rotation vector world to camera frame
        self.T_Ci_1__w = np.column_stack((R_cw, c_t_cw))
        
        R_w_Ci = self.T_Ci_1__w[:3,:3].T # rotation vector from Ci to world is inverse or rotation vector world to Ci
        w_t_w_Ci = - R_w_Ci @ self.T_Ci_1__w[:3,3, None] # tranformation vector from Ci to world in world frame, vertical vector

        # Step 3: Run KLT on candidate keypoints
        C_i, C_i_tracked = self.run_KLT(self.img_i_1, img_i, self.Ci_1, "C")

        # Step 4: Calculate angles between each tracked C_i
        # TODO: Can we do this without a for loop?
        for point_index in np.where(C_i_tracked)[0]: # loop through all tracked candidate points
            candidate_i = C_i[point_index]
            candidate_f = self.Fi_1[point_index]
            transformation_fw = self.Ti_1[point_index]

            T_if = multiply_transformation(self.T_Ci_1__w, inverse_transformation(transformation_fw.reshape(3,4)))
            R_cf = T_if.reshape(3,4)[:,:3]
            c_t_cf = T_if.reshape(3,4)[:,3][:,None] # column vector

            vector_to_candidate_i = np.linalg.inv(self.K) @ np.append(candidate_i, 1)[:, None]
            angle_between_baseline_and_point_i = np.arccos(
                                                            np.dot(c_t_cf.reshape(-1), vector_to_candidate_i.reshape(-1)) / 
                                                            (np.linalg.norm(c_t_cf) * np.linalg.norm(vector_to_candidate_i))
                                                            )
            vector_to_candidate_f = np.linalg.inv(self.K) @ np.append(candidate_f, 1)[:, None]
            f_t_fc = - R_cf.T @ c_t_cf # Take inverse of R_cf using the transpose since orthonormal
            angle_between_baseline_and_point_f = np.arccos(
                                                            np.dot(f_t_fc.reshape(-1), vector_to_candidate_f.reshape(-1)) / 
                                                            (np.linalg.norm(f_t_fc) * np.linalg.norm(vector_to_candidate_f))
                                                            )
            
            angle_between_points = np.pi - angle_between_baseline_and_point_f - angle_between_baseline_and_point_i

            """
            # Here is some code that does the same as above but using triangulation for the algorithm
            # You can use it to verify if the above works
            """

            projectionMat1 = self.K @ np.column_stack((np.identity(3), np.zeros(3)))
            projectionMat2 = self.K @ np.column_stack((R_cf, c_t_cf))

            triangulated_point = cv2.triangulatePoints(projectionMat1, projectionMat2, candidate_f, candidate_i)
            triangulated_point /= triangulated_point[3]
            triangulated_point = triangulated_point[:3]

            triangulated_point_in_i = R_cf @ triangulated_point + c_t_cf
            f_in_i = R_cf @ vector_to_candidate_f + c_t_cf
            
            triangulated_point_to_f = f_in_i - triangulated_point_in_i
            triangulated_point_to_ci = vector_to_candidate_i - triangulated_point_in_i

            angle_between_points_with_triangulation = np.arccos(
                    np.dot(triangulated_point_to_ci.reshape(-1), triangulated_point_to_f.reshape(-1)) / 
                    (np.linalg.norm(triangulated_point_to_ci) * np.linalg.norm(triangulated_point_to_f))
            )
            
            if debug and point_index ==  np.where(C_i_tracked)[0][0]:
                print(angle_between_points)
                print(angle_between_points_with_triangulation)

                fig = plt.figure(figsize=(14, 5))
                gs = fig.add_gridspec(2, 2)
                ax1 = fig.add_subplot(gs[0, 0])
                ax2 = fig.add_subplot(gs[0, 1])

                # Plot image and KLT results for candidate keypoint tracking
                ax1.imshow(self.img_i_1, cmap="grey")
                ax2.imshow(img_i, cmap="grey")

                a, b = candidate_i.ravel()
                c, d = candidate_f.ravel()
                ax1.plot((a, c), (b, d), '-', linewidth=2, c="red")
                ax2.plot((a, c), (b, d), '-', linewidth=2, c="green")
                
                ax1.scatter(c, d, s=5, c="green", alpha=0.5)
                ax2.scatter(a, b, s=5, c="red", alpha=0.5)
                ax1.set_title(f"C_i in Old Image")
                ax2.set_title(f"C_i in New Image")

                # Plot triangulation results
                ax3 = fig.add_subplot(gs[1:, :], projection='3d')
                ax3.view_init(elev=-90, azim=0, roll=-90)
                ax3.set_box_aspect((20, 10, 15)) # aspect_x, aspect_y, aspect_z
                ax3.set_xlabel("X")
                ax3.set_ylabel("Y")
                ax3.set_zlabel("Z")
                points_to_plot = [
                    (triangulated_point_in_i, "red"),
                    (np.array([0,0,0]), "black"), # camera i center
                    (c_t_cf, "cyan"), # camera f center
                    (f_in_i, "green"),
                    (vector_to_candidate_i, "magenta"),
                ]
                for point, colour in points_to_plot:
                    ax3.scatter(*point, marker='o', s=5, c=colour, alpha=0.5)

                plt.show()
                
        # Step 5: Add candidates that match thresholds to sets
            if angle_between_points_with_triangulation >= self.angle_threshold:
                # TODO: Don't use append
                self.Pi_1 = np.append(self.Pi_1, candidate_i[None,:], axis=0)
                self.Xi_1 = np.append(self.Xi_1, (R_w_Ci @ triangulated_point + w_t_w_Ci).T, axis=0)
                C_i_tracked[point_index] = False # remove this point from tracking

        # Step 6: Run SIFT if C is too small to add new candidates
        # TODO: Implement this step

        # Step 7: Update state vectors
        self.Ci_1 = C_i[C_i_tracked]
        self.Fi_1 = self.Fi_1[C_i_tracked]
        self.Ti_1 = self.Ti_1[C_i_tracked]
        self.img_i_1 = img_i
            
        # Step 8: Return pose
        return self.T_Ci_1__w


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
