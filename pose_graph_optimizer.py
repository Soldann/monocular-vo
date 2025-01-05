import numpy as np
from collections import deque

import scipy.linalg
import scipy.optimize
from utils import inverse_transformation, multiply_transformation, twist2HomogMatrix, HomogMatrix2twist
from initialise_vo import Bootstrap
import cv2
import scipy
import matplotlib.pyplot as plt
    
class PoseGraphOptimizer:

    class OptimizerType(Enum):
        POSE_GRAPH = 0
        BA = 1
        POSE_GRAPH_AND_BA = 2

    def __init__(self, K, initial_transform, sliding_window_size = 3, optimizer_type=OptimizerType.POSE_GRAPH):
        self.K = K
        self.sliding_window_size = sliding_window_size
        self.images = deque(maxlen=sliding_window_size)
        self.relative_transforms = deque(maxlen=self.sliding_window_size) # X by Y by 12, where relative_transforms[X,Y,:] is T_XY (transform from Y to X)
        # Note: Relative transforms could be size self.sliding_window_size - 1, as we don't need relative transforms to the last tracked frame. But for ease of indexing we store it anyways
        self.transform_to_world = deque(maxlen=self.sliding_window_size) # N x 12
        self.sift_features = deque(maxlen=self.sliding_window_size)

        self.optimizer_type = optimizer_type # what type of optimisation to run


    def add_image(self, new_image, transform_estimate):
        """
        Compute set of relative transforms from the images we currently have to the new image we get
         ### Parameters
            - new_image np.array (image size)
                - A new image frame
            - transform_estimate np.array shape (3, 4)
                - the current estimate T_cw for this new image frame
        """
        self.images.append(new_image)
        self.transform_to_world.append(HomogMatrix2twist(np.row_stack((transform_estimate, [0,0,0,1]))))

        if self.optimizer_type == self.OptimizerType.BA:
            return # skip all the SIFT stuff since we don't need it

        transforms = []
        sift = cv2.SIFT_create()
        kp2, des2 = sift.detectAndCompute(new_image, None)
        self.sift_features.append((kp2, des2))

        for i in range(len(self.images) - 1):
            image = self.images[i]
            ## Compute transform using DLT from all images in window to new image (reuse sift code lol)

            # Feature extraction
            kp1, des1 = self.sift_features[i]

            bf = cv2.BFMatcher()
            matches = bf.knnMatch(des1, des2, k=2)

            # Apply ratio test
            good = []
            for m,n in matches:
                # if m.distance < 0.8*n.distance or n.distance < 0.8*m.distance:
                if m.distance < 0.8*n.distance:
                    good.append([m])

            # Essential matrix by 8p algorithm
            points1 = np.array([kp1[match[0].queryIdx].pt for match in good])
            points2 = np.array([kp2[match[0].trainIdx].pt for match in good])
            E, ransac_inliers = cv2.findEssentialMat(points1, points2, self.K, method=cv2.FM_RANSAC, prob=0.999, threshold=1)
            ransac_inliers = ransac_inliers.astype(np.bool_).reshape(-1)

            _, R, t, _ = cv2.recoverPose(E, points1[ransac_inliers], points2[ransac_inliers], self.K)
            transforms.append(np.row_stack((np.column_stack((R, t)), [0,0,0,1])))

        self.relative_transforms.append(transforms)

    def optimize(self, landmarks, observations, with_pattern=True):
        pattern = None

        num_frames = len(self.transform_to_world)

        if self.optimizer_type == self.OptimizerType.POSE_GRAPH:
            values_to_optimize = np.array(self.transform_to_world).flatten()
        elif self.optimizer_type == self.OptimizerType.BA:
            values_to_optimize = np.concatenate((
                np.array(self.transform_to_world).flatten(),
                landmarks,
            ))
        else:
            raise RuntimeError("Unexpected Optimizer Type")

        if with_pattern:
            if self.optimizer_type == self.OptimizerType.POSE_GRAPH:
                num_error_terms = int(((len(self.relative_transforms) - 1) * len(self.relative_transforms))/2) # equals sum of range 1 to (len(self.relative_transforms) - 1)

                pattern = scipy.sparse.lil_matrix((num_error_terms, values_to_optimize.shape[0]), dtype=np.int8)
                for i in range(1, len(self.transform_to_world)):
                    pattern[:i*6] = 1 # each error is affected by i poses, which are 6 values each

                pattern = scipy.sparse.csr_matrix(pattern)
            elif self.optimizer_type == self.OptimizerType.BA:
                num_observations = (observations.shape[0]) / 3

                # Factor 2, one error for each x and y direction.
                num_error_terms = int(2 * num_observations)
                # Each error term will depend on one pose (6 entries) and one landmark position (3 entries),
                # so 9 nonzero entries per error term:
                pattern = scipy.sparse.lil_matrix((num_error_terms, values_to_optimize.shape[0]), dtype=np.int8)
                
                # Fill pattern for each frame individually:
                observation_i = 2  # iterator into serialized observations
                error_i = 0  # iterating frames, need another iterator for the error

                for frame_i in range(num_frames):
                    num_keypoints_in_frame = int(observations[observation_i])
                    # All errors of a frame are affected by its pose.
                    pattern[error_i:error_i + 2 * num_keypoints_in_frame, frame_i*6:(frame_i + 1)*6] = 1

                    # Each error is then also affected by the corresponding landmark.
                    landmark_indices = observations[observation_i + 2 * num_keypoints_in_frame + 1:
                                                    observation_i + 3 * num_keypoints_in_frame + 1]

                    for kp_i in range(landmark_indices.shape[0]):
                        pattern[error_i + kp_i * 2:error_i + (kp_i+1) * 2,
                                num_frames * 6 + int(landmark_indices[kp_i] - 1) * 3:num_frames * 6 + int(landmark_indices[kp_i]) * 3] = 1


                    observation_i = observation_i + 1 + 3 * num_keypoints_in_frame
                    error_i = error_i + 2 * num_keypoints_in_frame

                pattern = scipy.sparse.csr_matrix(pattern)


        def pose_residuals(world_to_cameras, poses):
            # world_to_cameras: the parameters to optimize
            # poses: the set of relative camera poses we use to inform the optimisation

            world_to_cameras = world_to_cameras.reshape((-1,6)) # reshape into array of transformation matrices
            residuals = []
            for i in range(1, len(poses)): # loop over the transforms from frame i to j. Start at 1 because there is no residual for the first image
                
                T_iw = twist2HomogMatrix(world_to_cameras[i]) # transform from w to i
                
                for j in range(i): # there should be i relative transformation matrices for the i-th pose we are optimising
                    T_jw = twist2HomogMatrix(world_to_cameras[j])
                    T_ij = poses[i][j] # relative_transform from pose j to pose i

                    residual = scipy.linalg.norm((T_iw - T_jw @ T_ij)[:3,:]) # compute difference between current estimate for world position of camera, and what we get from relative transforms
                    # ^ WTF it should be T_iw - T_ij @ T_jw but it only actually works when i do it this way what did i do wrong???
                    residuals.append(residual)
            return np.array(residuals).flatten()

        def baError(hidden_state):
            plot_debug = False
            num_frames = int(observations[0])
            T_W_C = hidden_state[:num_frames * 6].reshape([-1, 6]).T
            p_W_landmarks = hidden_state[num_frames * 6:].reshape([-1, 3]).T

            error_terms = []
            
            # Iterator into the observations that are encoded as explained in the problem statement.
            observation_i = 1

            for i in range(num_frames):
                single_T_W_C = twist2HomogMatrix(T_W_C[:, i])
                num_frame_observations = int(observations[observation_i + 1])
                keypoints = np.flipud(observations[observation_i + 2:observation_i + 2 + num_frame_observations*2].reshape([-1, 2]).T)

                landmark_indices = observations[observation_i + 2 + num_frame_observations*2:observation_i + 2 + num_frame_observations * 3]
                
                # Landmarks observed in this specific frame.
                p_W_L = p_W_landmarks[:, landmark_indices.astype(np.int) - 1]
                
                # Transforming the observed landmarks into the camera frame for projection.
                T_C_W = np.linalg.inv(single_T_W_C)
                p_C_L = np.matmul(T_C_W[:3, :3], p_W_L.transpose(1, 0)[:, :, None]).squeeze(-1) + T_C_W[:3, -1]

                # From exercise 1.
                projections = cv2.projectPoints(p_C_L, self.K)
                
                # Can be used to verify that the projections are reasonable.
                if plot_debug:
                    plt.clf()
                    plt.close()
                    plt.plot(projections[:, 0], projections[:, 1], 'o')
                    plt.plot(keypoints[0, :], keypoints[1, :], 'x')
                    plt.axis('equal')
                    plt.show()

                error_terms.append(keypoints.transpose(1, 0) - projections)
                observation_i = observation_i + num_frame_observations * 3 + 1

            return np.concatenate(error_terms).flatten()

        if self.optimizer_type == self.OptimizerType.POSE_GRAPH:
            result = scipy.optimize.least_squares(pose_residuals, values_to_optimize, args=(self.relative_transforms,), max_nfev=20, jac_sparsity=pattern)
            new_poses = (result.x).reshape(-1,6)
        elif self.optimizer_type == self.OptimizerType.BA:
            result = scipy.optimize.least_squares(baError, values_to_optimize, max_nfev=20, jac_sparsity=pattern)
            new_poses = (result.x)[:num_frames * 6].reshape([-1, 6])
            landmarks = (result.x)[num_frames * 6:].reshape([-1, 3]).T

        # print("Pose Graph Optimised: ")
        # for value in (result.x).reshape(-1,6):
        #     print(twist2HomogMatrix(value)[:3,:])
        # print("Regular output: ") # Separator for when the non-optimised value gets printed by the rest of the code
        return new_poses, landmarks
