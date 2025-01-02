import cv2
import numpy as np
from initialise_vo import Bootstrap
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Optional
from utils import *
import time

class VO:

    class Debug(Enum):
        KLT = 0
        TRIANGULATION = 1
        SIFT = 2

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

        self.frames_since_last_sift = 0

        # Setting information from bootstrapping
        self.dl = bootstrap_obj.data_loader                              # data loader
        self.img_i_1 = self.dl[bootstrap_obj.init_frames[-1]]

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

        # landmarks, keypoints, candidate points, transform world to camera position i-1
        self.Pi_1, self.Xi_1, self.Ci_1, self.T_Ci_1__w = bootstrap_obj.get_points() 
        self.Fi_1 = self.Ci_1.copy()

        self.Ti_1 = np.tile(self.T_Ci_1__w.reshape(-1), (len(self.Ci_1), 1))
        self.debug_ids = [self.dl.init_frames[-1]] * len(self.Ci_1)
        self.debug_counter = self.dl.init_frames[-1]
        
        self.max_keypoints = max_keypoints

        # Parameters Lucas Kanade
        self.lk_params = dict( winSize  = (15, 15),
                  maxLevel = 3,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))
        
        # angle threshold
        self.angle_threshold = 0.09991679144388552 # Assuming baseline is 10% of the depth
        self.angle_threshold = 0.01

        self.sift = cv2.SIFT_create(sigma=2, nOctaveLayers=3, edgeThreshold=10, nfeatures=50)

        self.sift_keypoint_similarity_threshold = 10

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


    def process_frame(self, img_i: np.array, debug: Optional[List[Debug]] = None):
        """
            Runs the continuous pipeline on the image provided, updating internal state wherever necessary

            ### Parameters
            1. img_i : np.array
                - numpy image to use as the next frame input

            2. debug : Optional[List[Debug]]
                - Provide a list of elements from the Enum VO.Debug to trigger additional prints
                  and visualizations useful for debugging.

            ### Returns
            - pose : np.array
                - A (3 x 4) np.array indicating the most recent camera pose. Columns 1 to 3 give
                R_cw, column 4 gives c_t_cw
            - X : np.array
                - Np.array containing the landmarks currently used for camera
                localisation on its rows. (n x 3). The landmarks are given in
                the world coordinate system.
            - P : np.array
                - The current keypoints in set P belonging to the landmarks
                X. Shape (n x 2)
        """
        self.frames_since_last_sift += 1
        # Step 1: run KLT  on the points P
        P_i, P_i_tracked = self.run_KLT(self.img_i_1, img_i, self.Pi_1, "P", self.Debug.KLT in debug if debug else False)
        
        self.Pi_1 = P_i[P_i_tracked] # Update Pi with the points we successfully tracked
        self.Xi_1 = self.Xi_1[P_i_tracked] # Update Xi with the ones we successfully tracked

        # Step 2: Run PnP to get pose for the new frame
        
        success, r_cw, c_t_cw, ransac_inliers = cv2.solvePnPRansac(self.Xi_1, self.Pi_1, self.K, self.distortion_coefficients, reprojectionError=8, iterationsCount=100) # Note that Pi_1 and Xi_1 are actually for Pi and Xi, since we updated them above
        if ransac_inliers is None or len(ransac_inliers) < len(self.Pi_1)/2:
            success, r_cw, c_t_cw, ransac_inliers = cv2.solvePnPRansac(self.Xi_1, self.Pi_1, self.K, self.distortion_coefficients, reprojectionError=9, iterationsCount=100) # Note that Pi_1 and Xi_1 are actually for Pi and Xi, since we updated them above

        ransac_inliers = ransac_inliers.flatten()
        self.Pi_1 = self.Pi_1[ransac_inliers] # Update Pi with ransac
        self.Xi_1 = self.Xi_1[ransac_inliers] # Update Xi with ransac

        # TODO: c_t_cw is the vector from camera frame to world frame, in the camera coordinates
        R_cw, _ = cv2.Rodrigues(r_cw)# rotation vector world to camera frame
        self.T_Ci_1__w = np.column_stack((R_cw, c_t_cw))
        
        R_w_Ci = self.T_Ci_1__w[:3,:3].T # rotation vector from Ci to world is inverse or rotation vector world to Ci
        w_t_w_Ci = - R_w_Ci @ self.T_Ci_1__w[:3,3, None] # tranformation vector from Ci to world in world frame, vertical vector
        
        # Step 3: Run KLT on candidate keypoints
        C_i, C_i_tracked = self.run_KLT(self.img_i_1, img_i, self.Ci_1, "C", self.Debug.KLT in debug if debug else False)

        # Step 4: Calculate angles between each tracked C_i
        # TODO: Can we do this without a for loop?
        if debug and self.Debug.TRIANGULATION in debug:
            self.debug_counter += 1

        # get the points
        point_indices = np.where(C_i_tracked)[0]

        candidate_is = C_i[point_indices]
        candidate_fs = self.Fi_1[point_indices]

        # to homogeneous to apply K
        candidate_is_homogeneous = np.hstack([candidate_is, np.ones((candidate_is.shape[0], 1))])  # (n, 3)
        candidate_fs_homogeneous = np.hstack([candidate_fs, np.ones((candidate_fs.shape[0], 1))])  # (n, 3)   
        
        # apply k to get directions
        inv_K = np.linalg.inv(self.K)
        vectors_to_candidate_is = candidate_is_homogeneous @ inv_K.T
        vectors_to_candidate_fs = candidate_fs_homogeneous @ inv_K.T

        # rotation matrix for each f point
        transformation_fw = self.Ti_1[point_indices]
        transformation_fw = transformation_fw[:, :12].reshape(-1, 3, 4)
        transformation_fw_r_tracked = transformation_fw[:, :, :3]  # (M, 3, 3)
        transformation_fw_r_tracked = transformation_fw_r_tracked.transpose(0, 2, 1)  # (M, 3, 3)

        T_Ci_1__w_r = self.T_Ci_1__w[:, :3]  # (3, 3)
        R_cfs = np.einsum('ij,mjk->mik', T_Ci_1__w_r, transformation_fw_r_tracked)  # (M, 3, 3)

        # normalized directions
        dirs_i = vectors_to_candidate_is / np.linalg.norm(vectors_to_candidate_is, axis=1, keepdims=True)  # (M, 3)
        dirs_f = vectors_to_candidate_fs / np.linalg.norm(vectors_to_candidate_fs, axis=1, keepdims=True)  # (M, 3)

        # rotate each dir_f by the rotation matrix
        dirs_f = np.einsum('nij,nj->ni', R_cfs, dirs_f) # (M, 3)

        # calculate the angle between each dir_i and dir_f
        angles_between_points = np.arccos(np.einsum('ij,ij->i', dirs_i, dirs_f))  # (M,)

        # get corresponding triangulations only of the matching ones
        angle_mask = angles_between_points >= self.angle_threshold
        print("Angle mask: ", str(angle_mask.sum()), "of", len(angle_mask))

        masked_candidate_is = candidate_is[angle_mask]  # (M', 3)
        masked_candidate_fs = candidate_fs[angle_mask]  # (M', 3)  
        masked_transformation_fw = transformation_fw[angle_mask]

        # Get projection matrices for triangulation
        masked_projection_fs = np.einsum('ij,kjl->kil', self.K, masked_transformation_fw)  # (M, 3, 3)
        projection_Ci = self.K @ self.T_Ci_1__w
        
        # zero list for the points
        triangulate_points_w = np.zeros((len(masked_candidate_is), 3))  # (M', 4)

        if debug and self.Debug.TRIANGULATION in debug:
            point_index = np.where(C_i_tracked)[0][0]
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

            """
            This solution triangulates the points directly in the world frame
            """

            projection_f = self.K @ transformation_fw.reshape(3,4)
            projection_Ci = self.K @ self.T_Ci_1__w

            triangulated_point_w = cv2.triangulatePoints(projection_f, projection_Ci, candidate_f, candidate_i)
            triangulated_point_w /= triangulated_point_w[3]
            triangulated_point_w = triangulated_point_w[:3]

            transformation_wf = inverse_transformation(transformation_fw.reshape(3,4))
            f_in_w = transformation_wf[:,:3] @ vector_to_candidate_f + transformation_wf[:,3][:,None]
            c_in_w = R_w_Ci @ vector_to_candidate_i + w_t_w_Ci

            triangulated_point_w_to_f = f_in_w - triangulated_point_w
            triangulated_point_w_to_ci = c_in_w - triangulated_point_w

            angle_between_points_with_triangulation_in_world = np.arccos(
                    np.dot(triangulated_point_w_to_f.reshape(-1), triangulated_point_w_to_ci.reshape(-1)) / 
                    (np.linalg.norm(triangulated_point_w_to_f) * np.linalg.norm(triangulated_point_w_to_ci))
            )
            
            print("Triangle method: ", angle_between_points)
            print("Triangulation method: ", angle_between_points_with_triangulation)
            print("Triangulation method in world frame: ", angle_between_points_with_triangulation_in_world)
            print("Triangulation method Luca: ", angles_between_points[0])

            # fig = plt.figure(figsize=(14, 5))
            # gs = fig.add_gridspec(2, 2)
            # ax1 = fig.add_subplot(gs[0, 0])
            # ax2 = fig.add_subplot(gs[0, 1])
            # ax3 = fig.add_subplot(gs[1, 0])

            # # Plot image and KLT results for candidate keypoint tracking
            # ax1.imshow(self.img_i_1, cmap="grey")
            # ax2.imshow(img_i, cmap="grey")
            # ax3.imshow(self.dl[self.debug_ids[point_index]], cmap="grey")

            # a, b = candidate_i.ravel()
            # c, d = self.Ci_1[point_index].ravel()
            # e, f = candidate_f.ravel()
            # ax1.plot((a, c), (b, d), '-', linewidth=2, c="red")
            # ax2.plot((a, c), (b, d), '-', linewidth=2, c="green")
            # ax3.plot((a, c), (b, d), '-', linewidth=2, c="pink")

            # ax1.scatter(c, d, s=5, c="green", alpha=0.5)
            # ax2.scatter(a, b, s=5, c="red", alpha=0.5)
            # ax3.scatter(e, f, s=5, c="green", alpha=0.5)
            # ax1.set_title(f"C_i in Old Image")
            # ax2.set_title(f"C_i in New Image")
            # ax3.set_title(F"C_i in Original Image")

            # # Plot triangulation results
            # ax4 = fig.add_subplot(gs[1, 1], projection='3d')
            # ax4.view_init(elev=-90, azim=0, roll=-90)
            # ax4.set_box_aspect((20, 10, 15)) # aspect_x, aspect_y, aspect_z
            # ax4.set_xlabel("X")
            # ax4.set_ylabel("Y")
            # ax4.set_zlabel("Z")
            # points_to_plot = [
            #     (triangulated_point_in_i, "red"),
            #     (np.array([0,0,0]), "black"), # camera i center
            #     (c_t_cf, "cyan"), # camera f center
            #     (f_in_i, "green"),
            #     (vector_to_candidate_i, "magenta"),
            # ]
            # for point, colour in points_to_plot:
            #     ax4.scatter(*point, marker='o', s=5, c=colour, alpha=0.5)

            # plt.show()

        # nooooo, a for loop
        for i in range(len(masked_candidate_is)):
            projection_f = masked_projection_fs[i]  # (3, 3)
            candidate_i = masked_candidate_is[i]  # (3,) 
            candidate_f = masked_candidate_fs[i]  # (3,)
            a, _ = linear_LS_triangulation(candidate_f.reshape(1,2), projection_f, candidate_i.reshape(1,2), projection_Ci)
            triangulated_point_w = np.append(a, 1)  # add a 1 to the end of the vector
            triangulate_points_w[i] = triangulated_point_w[:3]
            basis_change_matrix = np.vstack([self.T_Ci_1__w, np.array([0, 0, 0, 1])])
            p_in_camera = basis_change_matrix @ triangulated_point_w
            p_in_camera = p_in_camera[:3] / p_in_camera[3]  # Normalize the point

            if p_in_camera[2] < 0:
                pass
                #triangulated_point_w = triangulated_point_w * -1
                triangulated_point_w = triangulated_point_w * 0
                #print("behind???")

            triangulated_point_w = triangulated_point_w[:3]
        
            triangulate_points_w[i] = triangulated_point_w.reshape(-1)

        # remove close to 0
        # Step 4: Filter out close to zero values
        mask = np.all(np.abs(triangulate_points_w) > 0.1, axis=1)
        triangulate_points_w = triangulate_points_w[mask]
        masked_candidate_is = masked_candidate_is[mask]


        # Step 5: Add candidates that match thresholds to sets
        self.Pi_1 = np.vstack([self.Pi_1, masked_candidate_is])  # Stack candidates that match threshold
        self.Xi_1 = np.vstack([self.Xi_1, triangulate_points_w])  # Stack triangulated points

        # back to unfiltered array to remove added points from tracking
        filtered_indices = np.where(C_i_tracked)[0]  # Get indices of tracked candidate points
        new_filter_unfiltered = np.zeros_like(C_i_tracked, dtype=bool)
        new_filter_unfiltered[filtered_indices] = angle_mask
        C_i_tracked[new_filter_unfiltered] = False  # Update filtered array with new mask

        # Step 7: Update state vectors
        # how many true values in C_i_tracked?
        num_true = np.sum(C_i_tracked)
        print(f"Number of true values in C_i_tracked: {num_true}")
        print(f"Number of true values added to tracking data: {len(masked_candidate_is)}")
        print(f"total tracking data: {len(self.Pi_1)}")


        self.Ci_1 = C_i[C_i_tracked]
        self.Fi_1 = self.Fi_1[C_i_tracked]
        self.Ti_1 = self.Ti_1[C_i_tracked]
        self.img_i_1 = img_i

        # Step 6: Run SIFT if C is too small to add new candidates
        # TODO: Implement this step
        h, w = img_i.shape
        total = len(self.Ci_1)
        left_of_screen = np.sum(self.Ci_1[:, 0] < w*3/5)
        right_of_screen = np.sum(self.Ci_1[:, 0] > w*2/5)

        if (total <= self.max_keypoints or self.frames_since_last_sift > 500 or (left_of_screen < total/3 and left_of_screen < 100) or (right_of_screen < total/3 and right_of_screen < 100)) and True:
            
            old_features = np.row_stack((self.Pi_1, self.Ci_1))

            self.frames_since_last_sift = 0
            #feature_add_count = int(max(len(old_features)*2, 100))
            feature_add_count = 500 * 1000//len(self.Pi_1)
            self.sift = cv2.SIFT_create(nfeatures=feature_add_count, sigma=2.0, edgeThreshold=10)

            sift_img = img_i.copy()  # Copy the original image to work on it
            
            split_count_w = 5
            split_count_h = 3
            border_to_remove_w = w % split_count_w
            border_to_remove_h = h % split_count_h

            sift_w = w -border_to_remove_w
            sift_h = h - border_to_remove_h
            sift_img = sift_img[:sift_h, :sift_w]

            mask = np.ones(sift_img.shape, dtype=np.uint8) * 255  # Start with full mask
            for kp in old_features:
                cv2.circle(mask, (int(kp[0]), int(kp[1])), radius=int((10 * w/1000) * (len(P_i) / 300)), color=0, thickness=-1)
            
            blocks = sift_img.reshape(split_count_h, h // split_count_h, split_count_w, w // split_count_w).transpose(0, 2, 1, 3).reshape(split_count_h * split_count_w, h // split_count_h, w // split_count_w)
            mask_blocks = mask.reshape(split_count_h, h // split_count_h, split_count_w, w // split_count_w).transpose(0, 2, 1, 3).reshape(split_count_h * split_count_w, h // split_count_h, w // split_count_w)
            
            new_candidates = []

            t = time.time()

            for idx in range(blocks.shape[0]):
                block = blocks[idx]  # Get the current block
                mask_block = mask_blocks[idx]  # Get the corresponding mask block

                if right_of_screen < total/3.0 and idx % split_count_w <= split_count_w*2 // 3:
                    continue
                elif left_of_screen < total/3.0 and idx % split_count_w >= split_count_w // 3:
                    continue
                
                row = idx // split_count_w
                col = idx % split_count_w
                offset_x = col * (sift_w // split_count_w)
                offset_y = row * (sift_h // split_count_h)
                block_offset = np.array([offset_x, offset_y])
                keypoints = self.sift.detect(block, mask_block)
            
                supposed_len = 10 * 1000//len(self.Pi_1)
                for keypoint in sorted(keypoints, key=lambda k: -k.response)[:supposed_len]:
                    keypoint.pt += block_offset
                    new_candidates.append(keypoint)
                
                if(len(keypoints) < supposed_len):
                    # do Shi-Tomasi
                    corners = cv2.goodFeaturesToTrack(block, maxCorners=50, qualityLevel=0.1, minDistance=60)
                    if corners is not None:
                        for corner in corners:
                            new_candidates.append(cv2.KeyPoint(corner[0][0] + offset_x, corner[0][1] + offset_y, 1))

            print(f"Time taken for {blocks.shape[0]} blocks: {time.time() - t} seconds")

            # if total <= self.max_keypoints:
            #     new_candidates = self.sift.detect(sift_img, mask)
            # elif left_of_screen < total/3:
            #     left_half = sift_img[:, :w // 3]  # Left half
            #     left_mask = mask[:, :w // 3]
            #     new_candidates = self.sift.detect(left_half, left_mask)

            # elif right_of_screen < total/3:
            #     right_half = sift_img[:, w*2 // 3:]  # Right half
            #     right_mask = mask[:, w*2 // 3:]
            #     new_candidates = self.sift.detect(right_half, right_mask)
            #     # add w//2 to every x coorndiate
            #     for c in new_candidates:
            #         c.pt = (c.pt[0] + w*2 // 3, c.pt[1])


            #new_candidates = [kp for kp in new_candidates if kp.response > 0.0001]
            candidates_to_add = []
            for new_point in new_candidates:
                    candidates_to_add.append(new_point.pt)
            
            poses_to_add = np.tile(self.T_Ci_1__w.flatten(), (len(candidates_to_add), 1))
            
            print(f"Added keypoint count: {len(candidates_to_add)}")
            if debug and self.Debug.SIFT in debug:
                print(len(candidates_to_add))

            if len(candidates_to_add) > 0:
                self.Ci_1 = np.row_stack((self.Ci_1, candidates_to_add))
                self.Fi_1 = np.row_stack((self.Fi_1, candidates_to_add))
                self.Ti_1 = np.row_stack((self.Ti_1, poses_to_add))

            if self.Debug.TRIANGULATION:
                self.debug_ids.extend([self.debug_counter] * len(poses_to_add))

        # Step 8: Return pose, P, and X. Returning the i-1 version since the
        # sets were updated already
        return self.T_Ci_1__w, self.Pi_1, self.Xi_1, self.Ci_1


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
