import cv2
import numpy as np
from initialise_vo import Bootstrap
import matplotlib.pyplot as plt
from enum import Enum
from typing import List, Optional
from utils import *
import time
from pose_graph_optimizer import PoseGraphOptimizer

class VO:

    class Debug(Enum):
        KLT = 0
        TRIANGULATION = 1
        SIFT = 2

    def __init__(self, bootstrap_obj: Bootstrap, hyperparams):
        """
        Visual Odometry pipeline.
        
        Initialize VO using a bootstrap object. 
        SLAM continues from the second frame used for bootstrapping.

        ### Parameters
        1. bootstra_obj : Bootstrap
            - Instance of bootstrap giving the initial point cloud of land-
              marks and the positions of the corresponding keypoints
        """

        self.hyperparams = hyperparams

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
        self.Pi_1, self.Pi_old, self.Xi_1, self.Ci_1, self.T_Ci_1__w = bootstrap_obj.get_points() 
        self.Fi_1 = self.Ci_1.copy()

        self.Ti_1 = np.tile(self.T_Ci_1__w.reshape(-1), (len(self.Ci_1), 1))
        self.debug_ids = [self.dl.init_frames[-1]] * len(self.Ci_1)
        self.debug_counter = self.dl.init_frames[-1]
        
        self.max_keypoints = self.hyperparams.max_kp

        # Parameters Lucas Kanade
        self.lk_params = dict( winSize  = (31, 31),
                  maxLevel = 5)
        
        # angle threshold
        self.angle_threshold = 0.09991679144388552 # Assuming baseline is 10% of the depth
        self.angle_threshold = self.hyperparams.angle_threshold

        self.pose_optimiser = PoseGraphOptimizer(self.K, self.T_Ci_1__w.copy() )
        self.pose_optimiser.add_image(self.img_i_1.copy(), self.T_Ci_1__w.copy()) # add if not repeating the second bootstrap image

    def run_KLT(self, img_i_1, img_i, points_to_track, name_of_feature="features", debug=False):
        """
            Wrapper function for CV2 KLT
        """
        tracked_points, tracked, err = cv2.calcOpticalFlowPyrLK(img_i_1, img_i, points_to_track.astype(np.float32), None, **self.lk_params)
        if tracked is None:
            return None, None
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


    def reproject_to_unit_sphere_opencv(self, points_2d, K, dist_coeffs=None):
        """
        Reprojects 2D image coordinates to 3D points on a unit sphere using OpenCV.
        
        Args:
            points_2d: Array of shape (n, 2) containing 2D image coordinates (u, v).
            K: Camera matrix of shape (3, 3).
            dist_coeffs: Distortion coefficients (optional, set to None if no distortion).
            
        Returns:
            np.ndarray: Array of shape (n, 3) containing 3D points on the unit sphere.
        """
        # Ensure input points are in the correct shape for cv2.undistortPoints
        points_2d = np.asarray(points_2d, dtype=np.float32).reshape(-1, 1, 2)
        
        # Undistort points (convert to normalized camera coordinates)
        undistorted_points = cv2.undistortPoints(points_2d, cameraMatrix=K, distCoeffs=dist_coeffs)
        
        # Add z' = 1 to the normalized points (to form 3D direction vectors)
        normalized_points = np.hstack([undistorted_points.squeeze(axis=1), np.ones((len(points_2d), 1))])
        
        # Normalize to unit sphere
        points_on_sphere = normalized_points / np.linalg.norm(normalized_points, axis=1, keepdims=True)
        
        return points_on_sphere

    def compute_theta(self, old_P_1, P_1):
        """
        Compute theta for each pair of points in old_P_1 and P_1.
        
        Args:
            old_P_1 (numpy.ndarray): Array of shape (n, 3), old point correspondences (x, y, z).
            P_1 (numpy.ndarray): Array of shape (n, 3), new point correspondences (x, y, z).
        
        Returns:
            numpy.ndarray: Array of shape (n,) containing theta for each point pair.
        """
        
        # Unpack old and new points
        old_x, old_y, old_z = old_P_1.T
        x, y, z = P_1.T

        # Compute numerator and denominator for the arctan function
        # Note the FLIPPED x and y components due to us moving into the y direction.
        numerator = old_x * z - old_z * x 
        denominator = old_y * z + old_z * y

        # Compute theta for each point pair
        theta = -2 * np.arctan(numerator/denominator)
        
        return theta

    def find_indices_in_max_bin(self, thetas, num_bins=30):
        """
        Bins the given array of thetas into the specified number of bins,
        finds the bin with the maximum count, and returns the indices of thetas in that bin.
        
        Args:
            thetas: Array of shape (n, 1) or (n,) containing angle values (in radians).
            num_bins: Number of bins to divide the range of thetas into (default: 180).
        
        Returns:
            indices: Indices of thetas that fall into the bin with the maximum count.
        """
        # Flatten the input to ensure it's 1D
        thetas = thetas.flatten()
        
        # Bin thetas into the specified number of bins
        counts, bin_edges = np.histogram(thetas, bins=num_bins, range=(-np.pi, np.pi))
        
        # Find the bin with the maximum count
        max_bin_idx = np.argmax(counts)
        print(f"max angle: {bin_edges[max_bin_idx]}")
        
        # Determine the range of the max bin
        bin_min = bin_edges[max_bin_idx -1]
        bin_max = bin_edges[max_bin_idx + 2]
        
        # Find indices of thetas within this bin
        indices = np.where((thetas >= bin_min) & (thetas < bin_max))[0]
        
        return indices


    def pnp_RANSAC(self):
        ransac_3d_points = self.Xi_1
        ransac_2d_points = self.Pi_1
        mask = np.ones(len(ransac_2d_points), dtype=bool)

        if self.hyperparams.do_ransac1:
            unit_p1 = self.reproject_to_unit_sphere_opencv(self.Pi_1, self.K)
            unit_p2 = self.reproject_to_unit_sphere_opencv(self.Pi_old, self.K)
            thetas = self.compute_theta(unit_p1, unit_p2)
            indices_valid = self.find_indices_in_max_bin(thetas, self.hyperparams.ransac1_bincount)

            mask = np.zeros(len(thetas), dtype=bool)
            print(f"1-P RANSAC inlier count: {len(indices_valid)}")
            if len(indices_valid) < len(thetas)*1/5 or len(indices_valid) < 30: #more than 1/3 of the points are outliers
                print("Too many outliers. Giving up with constraint.")
                # give up with the constraint. We tried, we failed.
                mask = np.ones(len(thetas), dtype=bool)

            mask[indices_valid] = True

            ransac_3d_points = self.Xi_1[mask]
            ransac_2d_points = self.Pi_1[mask]
        
        success, r_cw, c_t_cw, ransac_inliers = cv2.solvePnPRansac(ransac_3d_points, ransac_2d_points, self.K, self.distortion_coefficients, reprojectionError=self.hyperparams.ransac_acc, iterationsCount=500) # Note that Pi_1 and Xi_1 are actually for Pi and Xi, since we updated them above
        if ransac_inliers is None or len(ransac_inliers) < len(ransac_2d_points)/3:
            success, r_cw, c_t_cw, ransac_inliers = cv2.solvePnPRansac(ransac_3d_points, ransac_2d_points, self.K, self.distortion_coefficients, reprojectionError=9, iterationsCount=300) # Note that Pi_1 and Xi_1 are actually for Pi and Xi, since we updated them above
        #success, r_cw, c_t_cw = cv2.solvePnP(ransac_3d_points, ransac_2d_points, self.K, self.distortion_coefficients, flags=cv2.SOLVEPNP_ITERATIVE) # Note that Pi_1 and Xi_1 are actually for Pi and Xi, since we updated them above

        ransac_inliers = ransac_inliers.flatten()
        #ransac_inliers = np.ones(len(ransac_3d_points), dtype=bool)
        self.Pi_1 = self.Pi_1[mask][ransac_inliers] # Update Pi with ransac
        self.Xi_1 = self.Xi_1[mask][ransac_inliers] # Update Xi with ransac

        # TODO: c_t_cw is the vector from camera frame to world frame, in the camera coordinates
        R_cw, _ = cv2.Rodrigues(r_cw)# rotation vector world to camera frame
        self.T_Ci_1__w = np.column_stack((R_cw, c_t_cw))
        
        # R_w_Ci = self.T_Ci_1__w[:3,:3].T # rotation vector from Ci to world is inverse or rotation vector world to Ci
        # w_t_w_Ci = - R_w_Ci @ self.T_Ci_1__w[:3,3, None] # tranformation vector from Ci to world in world frame, vertical vector

        # return R_w_Ci, w_t_w_Ci

    def get_tracked_ci_data(self, C_i, C_i_tracked):
        # get the points
        point_indices = np.where(C_i_tracked)[0]

        candidate_is = C_i[point_indices]
        candidate_fs = self.Fi_1[point_indices]

        # rotation matrix for each f point
        transformation_fw = self.Ti_1[point_indices]
        transformation_fw = transformation_fw[:, :12].reshape(-1, 3, 4)

        return candidate_is, candidate_fs, transformation_fw

    def get_current_angle_mask(self, candidate_is, candidate_fs, transformation_fw):
        # to homogeneous to apply K
        candidate_is_homogeneous = np.hstack([candidate_is, np.ones((candidate_is.shape[0], 1))])  # (n, 3)
        candidate_fs_homogeneous = np.hstack([candidate_fs, np.ones((candidate_fs.shape[0], 1))])  # (n, 3)   
        
        # apply k to get directions
        inv_K = np.linalg.inv(self.K)
        vectors_to_candidate_is = candidate_is_homogeneous @ inv_K.T
        vectors_to_candidate_fs = candidate_fs_homogeneous @ inv_K.T

        # rotation matrix for each f point
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
        return angle_mask, angles_between_points

    def triangulate_points_from_cis(self, candidate_is, candidate_fs, transformation_fw, angle_mask, angles_between_points):

        angles_between_points = angles_between_points[angle_mask]  # (M',)
        masked_candidate_is = candidate_is[angle_mask]  # (M', 3)
        masked_candidate_fs = candidate_fs[angle_mask]  # (M', 3)  
        masked_transformation_fw = transformation_fw[angle_mask]

        # Get projection matrices for triangulation
        masked_projection_fs = np.einsum('ij,kjl->kil', self.K, masked_transformation_fw)  # (M, 3, 3)
        projection_Ci = self.K @ self.T_Ci_1__w
        
        # zero list for the points
        triangulate_points_w = np.zeros((len(masked_candidate_is), 3))  # (M', 4)

        # sort masked_candidate_is by angles_between_points

        # masked_candidate_is = masked_candidate_is[np.argsort(-angles_between_points)]
        # masked_candidate_fs = masked_candidate_fs[np.argsort(-angles_between_points)]
        # masked_transformation_fw = masked_transformation_fw[np.argsort(-angles_between_points)]

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
                triangulated_point_w = triangulated_point_w * 0
                #print("behind???")

            triangulated_point_w = triangulated_point_w[:3]
        
            triangulate_points_w[i] = triangulated_point_w.reshape(-1)

        # remove close to 0
        # Step 4: Filter out close to zero values
        mask = np.all(np.abs(triangulate_points_w) > 0.1, axis=1)
        triangulate_points_w = triangulate_points_w[mask]
        masked_candidate_is = masked_candidate_is[mask]

        return triangulate_points_w, masked_candidate_is

    def extract_new_features(self, split_count_w, split_count_h, total, right_too_small, left_too_small, total_too_small):
        h, w = self.img_i_1.shape

        old_features = np.row_stack((self.Pi_1, self.Ci_1))



        sift_img = self.img_i_1.copy()  # Copy the original image to work on it
        
        border_to_remove_w = w % split_count_w
        border_to_remove_h = h % split_count_h

        sift_w = w -border_to_remove_w
        sift_h = h - border_to_remove_h
        sift_img = sift_img[:sift_h, :sift_w]

        mask = np.ones(sift_img.shape, dtype=np.uint8) * 255  # Start with full mask
        for kp in old_features:
            cv2.circle(mask, (int(kp[0]), int(kp[1])), radius=self.hyperparams.block_radius, color=0, thickness=-1)
        
        blocks = sift_img.reshape(split_count_h, h // split_count_h, split_count_w, w // split_count_w).transpose(0, 2, 1, 3).reshape(split_count_h * split_count_w, h // split_count_h, w // split_count_w)
        mask_blocks = mask.reshape(split_count_h, h // split_count_h, split_count_w, w // split_count_w).transpose(0, 2, 1, 3).reshape(split_count_h * split_count_w, h // split_count_h, w // split_count_w)
        
        if self.hyperparams.do_sift:
            feature_add_count = 500 * 1000//len(self.Pi_1)
            self.sift = cv2.SIFT_create(nfeatures=feature_add_count, sigma=2.0, edgeThreshold=10)

        new_candidates = []

        t = time.time()

        for idx in range(blocks.shape[0]):
            block = blocks[idx]  # Get the current block
            mask_block = mask_blocks[idx]  # Get the corresponding mask block
            # if right_too_small and not total_too_small and idx % split_count_w <= split_count_w*2 / 3:
            #     continue
            # elif left_too_small and not total_too_small and idx % split_count_w >= split_count_w/ 3:
            #     continue
            
            row = idx // split_count_w
            col = idx % split_count_w
            offset_x = col * (sift_w // split_count_w)
            offset_y = row * (sift_h // split_count_h)
            block_offset = np.array([offset_x, offset_y])
            
            keypoints = []
            if self.hyperparams.do_sift:
                keypoints = self.sift.detect(block, mask_block)
                for keypoint in sorted(keypoints, key=lambda k: -k.response):
                    keypoint.pt += block_offset
                    new_candidates.append(keypoint.pt)

            supposed_len = 30 * 1000//len(self.Pi_1)
            if(len(keypoints) < supposed_len):
                # do Shi-Tomasi
                corners = cv2.goodFeaturesToTrack(block, maxCorners=self.hyperparams.harris_count, qualityLevel=self.hyperparams.harris_threshold, minDistance=self.hyperparams.block_radius)
                if corners is not None:
                    for corner in corners:
                        new_candidates.append(np.array([corner[0][0] + offset_x, corner[0][1] + offset_y]))

        print(f"Time taken for {blocks.shape[0]} blocks: {time.time() - t} seconds")

        #new_candidates = [kp for kp in new_candidates if kp.response > 0.0001]
        
        poses_to_add = np.tile(self.T_Ci_1__w.flatten(), (len(new_candidates), 1))
        
        print(f"Added keypoint count: {len(new_candidates)}")

        if len(new_candidates) > 0:
            self.Ci_1 = np.row_stack((self.Ci_1, new_candidates))
            self.Fi_1 = np.row_stack((self.Fi_1, new_candidates))
            self.Ti_1 = np.row_stack((self.Ti_1, poses_to_add))

        if self.Debug.TRIANGULATION:
            self.debug_ids.extend([self.debug_counter] * len(poses_to_add))

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
        start_t = time.time()

        # Step 1: run KLT  on the points P
        P_i, P_i_tracked = self.run_KLT(self.img_i_1, img_i, self.Pi_1, "P", self.Debug.KLT in debug if debug else False)
        
        self.Pi_old = self.Pi_1[P_i_tracked]
        self.Pi_1 = P_i[P_i_tracked] # Update Pi with the points we successfully tracked
        self.Xi_1 = self.Xi_1[P_i_tracked] # Update Xi with the ones we successfully tracked

        KLT_1_time = time.time() - start_t

        # Step 2: Run PnP to get pose for the new frame
        self.pnp_RANSAC()

        RANSAC_t = time.time() - start_t - KLT_1_time
        
        # Step 3: Run KLT on candidate keypoints
        C_i, C_i_tracked = self.run_KLT(self.img_i_1, img_i, self.Ci_1, "C", self.Debug.KLT in debug if debug else False)

        KLT_2_t = time.time() - start_t - KLT_1_time - RANSAC_t

        # Step 4: Calculate angles between each tracked C_i
        if debug and self.Debug.TRIANGULATION in debug:
            self.debug_counter += 1

        if C_i_tracked is not None:
            candidate_is, candidate_fs, transformation_fw = self.get_tracked_ci_data(C_i, C_i_tracked)
            angle_mask, angles_between_points = self.get_current_angle_mask(candidate_is, candidate_fs, transformation_fw)
        
            triangulate_points_w, masked_candidate_is = self.triangulate_points_from_cis(candidate_is, candidate_fs, transformation_fw, angle_mask, angles_between_points)

            Angle_t = time.time() - start_t - KLT_1_time - RANSAC_t - KLT_2_t

            # 20% original data!
            triangulate_points_w = triangulate_points_w[:4*len(self.Pi_1), :]
            masked_candidate_is = masked_candidate_is[:4*len(self.Pi_1), :]

            # Step 5: Add candidates that match thresholds to sets
            self.Pi_1 = np.vstack([self.Pi_1, masked_candidate_is])  # Stack candidates that match threshold
            self.Xi_1 = np.vstack([self.Xi_1, triangulate_points_w])  # Stack triangulated points

            # back to unfiltered array to remove added points from tracking
            filtered_indices = np.where(C_i_tracked)[0]  # Get indices of tracked candidate points
            new_filter_unfiltered = np.zeros_like(C_i_tracked, dtype=bool)
            new_filter_unfiltered[filtered_indices] = angle_mask
            C_i_tracked[new_filter_unfiltered] = False  # Update filtered array with new mask

            # Step 6: Update state vectors
            # how many true values in C_i_tracked?
            num_true = np.sum(C_i_tracked)
            print(f"Number of true values in C_i_tracked: {num_true}")
            print(f"Number of true values added to tracking data: {len(masked_candidate_is)}")
            print(f"total tracking data: {len(self.Pi_1)}")


            self.Ci_1 = C_i[C_i_tracked]
            self.Fi_1 = self.Fi_1[C_i_tracked]
            self.Ti_1 = self.Ti_1[C_i_tracked]
            self.img_i_1 = img_i

        Triangulation_t = time.time() - start_t - KLT_1_time - RANSAC_t - KLT_2_t - Angle_t


        # Step 7: Run feature detection if C is too small to add new candidates. May be SIFT or Shi-Tomasi
        h, w = img_i.shape
        total = len(self.Ci_1)
        left_of_screen = np.sum(self.Ci_1[:, 0] < w*3/5)
        right_of_screen = np.sum(self.Ci_1[:, 0] > w*2/5)

        total_too_small = total <= self.max_keypoints
        left_too_small = left_of_screen < total/3 and left_of_screen < self.max_keypoints
        right_too_small = right_of_screen < total/3 and right_of_screen < self.max_keypoints

        if total_too_small or left_too_small or right_too_small or True:
            self.extract_new_features(self.hyperparams.w_split, self.hyperparams.h_split, total, right_too_small, left_too_small, total_too_small)

        SIFT_t = time.time() - start_t - KLT_1_time - RANSAC_t - KLT_2_t - Angle_t - Triangulation_t


        # Step 8 Pose graph optimisation
        Optimizer_keypoints_t = time.time() - start_t - KLT_1_time - RANSAC_t - KLT_2_t - Angle_t - Triangulation_t - SIFT_t
        if self.hyperparams.do_optimize:
            self.pose_optimiser.add_image(img_i,self.T_Ci_1__w.copy())
            optimised_poses = self.pose_optimiser.optimize()
            optimised_poses = [twist2HomogMatrix(pose)[:3,:] for pose in optimised_poses]
            self.T_Ci_1__w = optimised_poses[-1]
        else:
            optimised_poses = [self.T_Ci_1__w]

        Optimizer_t = time.time() - start_t - KLT_1_time - RANSAC_t - KLT_2_t - Angle_t - Triangulation_t - SIFT_t - Optimizer_keypoints_t
        total_time =  time.time() - start_t

        # print all times
        print("KLT_1_time:", KLT_1_time)
        print("RANSAC_t:", RANSAC_t)
        print("KLT_2_t:", KLT_2_t)
        print("Angle_t:", Angle_t)
        print("Triangulation_t:", Triangulation_t)
        print("SIFT_t:", SIFT_t)
        print("Optimizer_keypoints_t:", Optimizer_keypoints_t)
        print("Optimizer_t:", Optimizer_t)
        print("total_time:", total_time)

        # Step 9: Return pose, P, and X. Returning the i-1 version since the
        # sets were updated already
        return optimised_poses, self.Pi_1, self.Xi_1, self.Ci_1

