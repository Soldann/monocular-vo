"""
Module for trajectory evaluation

Note: 
- The self-produced datasets do not have a ground truth.
- The Malaga dataset does not provide orientation ground truths. The relative
  error requires the orientation to align the subtrajectories at their starting
  states. Hence, the relative error is not implemented for Malaga.
"""

from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import numpy as np
from cv2 import Rodrigues
from initialise_vo import DataLoader

from utils import inverse_transformation, drawCamera


class TrajectoryEval:

    def __init__(self, dataset_name: str, first_frame: int):
        """
        
        ### Parameter
        1. dataset_name : str
            - Name of a dataset. Looks for the corresponding .pkl file of the folder 
            "solution_trajectories". Should be one of ("parking", "malaga", "kitti",
            "own_1", "own_2")
        2. first_frame : int
            - Where SLAM started. Specifically: the index of the second bootstrapping
            frame. It is assumed that the poses list contain the T matrix for the
            second bootstrapping frame as their first element, followed by one T for
            every consecutive frame
        """

        # Naming conventions:
        #   T_cw_list       a list of (3 x 4) np.arrays. T from w to current c frame
        #   T_cw_array      an np.array of (3n x 4) where n is the number of poses. Also 
        #                   contains T_cw
        #   T_wc_list       As above but containing the inverse of each matrix: T_wc
        #   T_wc_array      See above.
        #
        #   gt_T_wc_list    The ground truth version of T_wc_list
        #   gt_T_wc_array   The ground truth version of T_wc_array

        # ---------- LOAD EST. TRAJECTORY FROM FILE ---------- #

        self.dataset_name = dataset_name

        # Load T_cw_list from .pkl file in "solution_trajectories"
        trajec_dir = Path().cwd().joinpath("solution_trajectories")
        poses_path = trajec_dir.joinpath(self.dataset_name + ".pkl")
        with open(poses_path, "rb") as f:
            self.T_cw_list = pickle.load(f)
        
        # Find the inverse transforms T_wc
        self.T_wc_list = [inverse_transformation(T_wc) for T_wc in self.T_cw_list]
        self.T_wc_array = np.vstack(self.T_wc_list)

        # ---------- LOAD GT TRAJECTORY FROM FILE ---------- #
        
        # Depending on the passed datset_name, load the ground truth trajectory
        if self.dataset_name == "parking":
            
            # Load ground truth poses
            gt_poses_path = Path.cwd().joinpath("datasets", "parking", "poses.txt")
            gt_poses_array = np.loadtxt(gt_poses_path.as_posix())

            # Find the window of ground truth poses that are needed & reshape to list
            self.n_poses = min(gt_poses_array.shape[0], len(self.T_wc_list))
            gt_poses_array = gt_poses_array[first_frame:(first_frame + self.n_poses)]
            self.gt_T_wc_array = gt_poses_array.reshape(-1, 4)
            self.gt_T_wc_list = [self.gt_T_wc_array[(3 * i):(3 * i + 3)] 
                                 for i in range(self.n_poses)]

        elif self.dataset_name == "kitti":

            # Load ground truth poses
            gt_poses_path = Path.cwd().joinpath("datasets", "kitti", "poses", "05.txt")
            gt_poses_array = np.loadtxt(gt_poses_path.as_posix())

            # Find the window of ground truth poses that are needed & reshape to list
            self.n_poses = min(gt_poses_array.shape[0], len(self.T_wc_list))
            gt_poses_array = gt_poses_array[first_frame:(first_frame + self.n_poses)]
            self.gt_T_wc_array = gt_poses_array.reshape(-1, 4)
            self.gt_T_wc_list = [self.gt_T_wc_array[(3 * i):(3 * i + 3)] 
                                 for i in range(self.n_poses)]
        
        elif self.dataset_name == "malaga":

            # All variables starting with m_ relate to Malaga

            # Path to the txt files
            m_path = Path.cwd().joinpath("datasets", "malaga-urban-dataset-extract-07")
            
            # ----- ALIGN VIDEO AND GPS ----- #

            m_idx_to_time_path = m_path.joinpath("malaga-urban-dataset-extract-07_all"
                                                 + "-sensors_IMAGES.txt")
            m_gps_path = m_path.joinpath("malaga-urban-dataset-extract-07_all-sensors"
                                         +"_GPS.txt")

            # Loading array of when each image was taken (m_idx_to_time)
            def conv_gps(img_name):
                return np.float64(img_name[12:29])

            m_idx_to_time = np.loadtxt(m_idx_to_time_path.as_posix(), 
                                       dtype=np.float64, converters=conv_gps)
            m_idx_to_time = m_idx_to_time[::2]      # stereo: want only every second

            # Load GPS data
            m_gps = np.loadtxt(m_gps_path.as_posix(), skiprows=1, usecols=(0, 8, 9, 10))

            # Find the closest time among the images for each GPS time
            m_diff_gps = np.abs(m_idx_to_time[:, np.newaxis] - m_gps[:, 0])
            m_match_idx_gps = np.argmin(m_diff_gps, axis=0)

            # The images where comparison can actually happen
            m_max_index = min(first_frame + len(self.T_cw_list), m_match_idx_gps[-1])
            m_gps_mask = (m_match_idx_gps <= m_max_index) & (m_match_idx_gps >= first_frame)
            m_match_idx_gps = m_match_idx_gps[m_gps_mask]
            self.n_poses = m_match_idx_gps.shape[0]

            # Format to index an array of stacked T-matrices (T_wc)
            m_match_idx_Twc = np.c_[3 * m_match_idx_gps, 3 * m_match_idx_gps + 1,
                                     3 * m_match_idx_gps + 2].flatten()

            # Can only compare at matched locations: restrict T_wc
            self.T_wc_array = self.T_wc_array[m_match_idx_Twc]
            self.T_cw_list = [self.T_wc_array[(3 * i):(3 * i + 3)] 
                              for i in range(self.n_poses)]
            
            # The corresponding GPS positions (orientation not given)
            m_gps_pos = m_gps[m_gps_mask, 1:]

            # ---------- POSE ALIGNMENT USING TWO GT POSITIONS --------- #

            # Malaga does not provide ground truth orientations. The IMU data has
            # not been found to provide very reliable GT estimates.
            # Since the camera is forward-facing in malaga and there is no rolling
            # the following approximation of the orientation can be made based on
            # pairs of points from the ground truth positions

            w_y_c = np.array([0, 1, 0])
            R_wc = []

            for i in range(m_gps_pos.shape[0] - 1):

                current_t = m_gps_pos[i]
                next_t = m_gps_pos[i + 1]

                # The difference vector as the z-direction of the camera frame
                delta = next_t - current_t

                w_z_c = delta / np.sqrt(np.sum(delta**2))
                w_x_c = np.cross(w_y_c, w_z_c)
                w_x_c /= np.sqrt(np.sum(w_x_c**2))

                R_wc.append(np.c_[w_x_c, w_y_c, w_z_c])

            R_wc.append(R_wc[-1])

            self.gt_T_wc_array = np.c_[np.vstack(R_wc), m_gps_pos.flatten()]
            self.gt_T_wc_list = [self.gt_T_wc_array[(3 * i):(3 * i + 3)] 
                                 for i in range(self.n_poses)]
        
        else:

            raise(NotImplementedError)  
                    
    def draw_trajectory(self, gt=False, add_cam_frame=None):
        """
        Draw the VO trajectory; plots the current self.T_wc

        ### Parameters
        1. gt : bool (default: False)
            - Inidicate whether to add the ground truth trajectory to the plot.
            Call self.similarity_transform_3d() before setting this to true
        2. add_cam_frame : int (default: None)
            - Add the camera frame (3 arrows) at the corresponding frame index
            of the VO trajectory. If None: no camera frame is added.
            The colour coding of the arrows:
                - c_X : Red
                - c_Y : Green
                - c_Z : Blue
        """

        w_t_wc__x = self.T_wc_array[0::3, 3]
        w_t_wc__y = self.T_wc_array[1::3, 3]
        w_t_wc__z = self.T_wc_array[2::3, 3]

        fig = plt.figure()
        ax = fig.add_subplot(projection="3d")
        ax.plot(w_t_wc__x, w_t_wc__y, w_t_wc__z, color="b")
        ax.set_xlabel("$X_w$")
        ax.set_ylabel("$Y_w$")
        ax.set_zlabel("$Z_w$")
        ax.set_title("Absolute Trajectory Error")

        if gt:  # If the ground truth should be plotted as well
            gt_w_t_wc__x = self.gt_T_wc_array[0::3, 3]
            gt_w_t_wc__y = self.gt_T_wc_array[1::3, 3]
            gt_w_t_wc__z = self.gt_T_wc_array[2::3, 3]
            ax.plot(gt_w_t_wc__x, gt_w_t_wc__y, gt_w_t_wc__z, color="r")
            ax.legend(["VO estimate", "Ground truth"])

            # Compute the limits for the plot; same scale for both axes
            xmin = np.min(np.r_[w_t_wc__x, gt_w_t_wc__x]) - 1
            xmax = np.max(np.r_[w_t_wc__x, gt_w_t_wc__x]) + 1
            ymin = np.min(np.r_[w_t_wc__y, gt_w_t_wc__y]) - 1
            ymax = np.max(np.r_[w_t_wc__y, gt_w_t_wc__y]) + 1
            zmin = np.min(np.r_[w_t_wc__z, gt_w_t_wc__z]) - 1
            zmax = np.max(np.r_[w_t_wc__z, gt_w_t_wc__z]) + 1
        
        else:
            # Compute the limits for the plot; same scale for both axes
            xmin, xmax = np.min(w_t_wc__x) - 1, np.max(w_t_wc__x) + 1
            ymin, ymax = np.min(w_t_wc__y) - 1, np.max(w_t_wc__y) + 1
            zmin, zmax = np.min(w_t_wc__z) - 1, np.max(w_t_wc__z) + 1

        max_range = max(xmax - xmin, zmax - zmin, ymax - ymin)
        mid_x = 0.5 * (xmin + xmax)
        mid_z = 0.5 * (zmin + zmax)
        mid_y = 0.5 * (ymin + ymax)

        # Set the computed limits
        ax.set_xlim(mid_x - 0.5 * max_range, mid_x + 0.5 * max_range)
        ax.set_ylim(mid_y - 0.5 * max_range, mid_y + 0.5 * max_range)
        ax.set_zlim(mid_z - 0.5 * max_range, mid_z + 0.5 * max_range)

        # Have the camera view be the orientation of the world coord sys.
        ax.view_init(elev=-80, azim=-90, roll=0)

        # Add a camera frame at some position if desired:
        if add_cam_frame is not None:
            drawCamera(ax, self.T_wc_list[add_cam_frame][:, 3],
                    self.T_wc_list[add_cam_frame][:, :3], 
                    set_ax_limits=False)
        
        plt.show()


    def similarity_transform_3d(self):
        """
        Apply a 3d similarity transform to the trajectory (Umeyama method).
        Alignes the trajectory to the ground truth in terms of:
        - Absolute scale
        - Absolute rotation
        - Absolute translation
        """

        # ---------- FIND R, t, s BY UMEYAMA ---------- #

        # The point coordinates in a (n x 3) array (...in world coordinates)
        gt_w_t_wc = self.gt_T_wc_array[:, 3].reshape(-1, 3)
        w_t_wc = self.T_wc_array[:, 3].reshape(-1, 3)

        # The mean point for the ground truth and estimated trajectors
        mu_p_hat = np.mean(w_t_wc, axis=0)          # mean trajectory point
        mu_p = np.mean(gt_w_t_wc, axis=0)           # mean g.t. point

        # The mean squared deviation of points
        sigma_p_hat_squared = np.sum((w_t_wc - mu_p_hat)**2) / self.n_poses
        sigma_p_squared = np.sum((gt_w_t_wc - mu_p)**2) / self.n_poses

        # Sigma matrix: sum of outer products (covariance)
        Sigma = (gt_w_t_wc - mu_p).T @ (w_t_wc - mu_p_hat) / self.n_poses

        # Singular value decomposition
        svd = np.linalg.svd(Sigma)

        # Find W based on the SVD
        if np.linalg.det(svd.U) * np.linalg.det(svd.Vh) < 0:
            W = np.diag((1, 1, -1))
        else:
            W = np.eye(3)
        
        # Find the solution rotation matrix R: essentially R_gt_w
        # from the VO world system to the ground truth system
        R = svd.U @ W @ svd.Vh

        # Find the solution scale factor s
        s = np.trace(np.diag(svd.S) @ W) / sigma_p_hat_squared

        # Find the solution translation vector t
        t = mu_p - s * R @ mu_p_hat

        # ---------- APPLY SIMILARITY TRANSFORM TO TRAJECTORY ---------- #

        # Rotate the camera frame orientations by R (not rescaling by s):
        R_wc_dash = self.T_wc_array[:, :3] @ R

        # Apply the similarity transform to the points w_t_wc
        t_dash = s * w_t_wc @ R.T + t

        # Update self.T_wc_array and self.T_wc_list
        self.T_wc_array = np.hstack((R_wc_dash, t_dash.reshape(-1, 1)))
        self.T_wc_list = [self.T_wc_array[(3 * i):(3 * i + 3)] 
                          for i in range(self.n_poses)]

    
    def absolue_trajectory_error(self):
        """
        Compute the ATE. The trajectories should already be aligned by use
        of self.similarity_transform_3d() at this point
        """

        # Compute the RMSE as shown on the lecture slides

        RMSE = np.sqrt(np.sum((self.gt_T_wc_array[:, 3] 
                               - self.T_wc_array[:, 3])**2) / self.n_poses)
        
        return RMSE
    
    def _get_relative_error(self, trajectory_length: float):
        """
        The relative error is composed of two separable errors: roation and position.
        Scale drift can also be visualised by the change of alignment scales across
        the trajectory

        ### Parameters
        1. trajectory_length : float
            - Trajectory length in meters

        ### Returns
        1. position error
        2. rotation error
        3. alignment scales
        4. split and aligned trajectories
        """

        split_pos = self.re_get_split_pos(trajectory_length)
        split_trajec = np.zeros([split_pos[-1], 3])
        rot_err = np.zeros([len(split_pos) - 1, 3])
        scales = np.zeros(len(split_pos) - 1)
        pos_err = np.zeros(len(split_pos) - 1)

        for i in range(len(split_pos) - 1):

            pos = split_pos[i]
            next_pos = split_pos[i + 1]

            # ---------- COMPUTE ROTATION ERROR ---------- #

            # Difference of rotation between the poses
            R_align = self.gt_T_wc_list[pos][:, :3] @ self.T_wc_list[pos][:, :3].T

            # Compute the rotation error at the last frame of the subtrajectory
            # First apply alignment to the rotation estimation of the last frame
            R_last_aligned = R_align @ self.T_wc_list[next_pos][:, :3] 
            R_err = R_last_aligned @ self.gt_T_wc_list[next_pos][:, :3].T

            # Convert the rotation error to vector representation and save
            radian_error = Rodrigues(R_err)[0].flatten()
            rot_err[i] = radian_error * 180 / np.pi

            # ---------- ALIGN THE SUBTRAJECTORIES ---------- #

            # Estimated translations of the subtrajectory as a (3 x n)
            t_est = self.T_wc_array[(3 * pos):(3 * next_pos), 3]
            t_est = t_est.reshape(3, -1, order="F")

            # Subtracting the first element: rotation around the first position
            t_est_rel = t_est - np.c_[t_est[:, 0]]
            t_est_rot = R_align @ t_est_rel

            # Adapting the scale: Minimise the distance between the aligned vector
            # and the ground truth
            gt_end_vec = self.gt_T_wc_list[next_pos][:, 3] - self.gt_T_wc_list[pos][:, 3]
            t_end_vec = t_est_rot[:, -1]
            scale = np.dot(t_end_vec, gt_end_vec) / np.sum(t_end_vec**2)
            t_est_rot *= scale
            scales[i] = scale

            # Adding the translation vec of the GT's first pose:
            t_aligned = t_est_rot + np.c_[self.gt_T_wc_list[pos][:, 3]]

            # Compute the position error after rescaling
            pos_err[i] = np.sqrt(np.sum((t_aligned[:, -1]
                                         - self.gt_T_wc_list[next_pos][:, 3])**2))

            # Save the split trajectories for visualisation
            split_trajec[pos:next_pos] = t_aligned.T

        return pos_err, rot_err, scales, split_trajec, split_pos


    def relative_error(self, trajec_lenghts=(7, 23, 31, 39, 100)):
        """
        Compute and visualise the relative error measures for several subtrajectory 
        lengths.
        Note: since Malaga does not provide ground-truth orientations, the relative
        error is not impoemented for this dataset.

        ### Parameters
        1. trajec_lengths: tuple
            - The subtrajectory lengths in meters for which the relative error and its
            statistics should be computed. The last of these can be visualised in a 3D
            plot. The first of these provides the information for the scale-drift plot
        """

        n_lengths = len(trajec_lenghts)

        pos_errs = []
        rot_errs = []
        abs_rot_errs = []
        scale_drifts = []
        split_trajecs = []
        split_pos_s = []

        for i in range(n_lengths):

            for_length_i = self._get_relative_error(trajec_lenghts[i])
            pos_err, rot_err, scale_drift, split_trajec, split_pos = for_length_i
            pos_errs.append(pos_err)
            rot_errs.append(rot_err)
            scale_drifts.append(scale_drift)
            split_trajecs.append(split_trajec)
            split_pos_s.append(split_pos)

            abs_rot_err = np.sqrt(np.sum(rot_err**2, axis=1))
            abs_rot_errs.append(abs_rot_err)

        # ---------- STATISTICS PLOT ---------- #

        mosaic_struc = np.array([["box_pos"],
                                 ["box_rot"],
                                 ["scale"]])

        fig1, axs = plt.subplot_mosaic(mosaic_struc, layout="constrained",
                                       figsize=(6, 7))
        
        axs["box_pos"].boxplot(pos_errs)
        axs["box_pos"].set_title("Translation error")
        axs["box_pos"].set_xlabel("Subtrajectory length [m]")
        axs["box_pos"].set_ylabel("Translation error [m]")
        axs["box_pos"].set_xticklabels([str(i) for i in trajec_lenghts])

        axs["box_rot"].boxplot(abs_rot_errs)
        axs["box_rot"].set_title("Rotation error")
        axs["box_rot"].set_xlabel("Subtrajectory length [m]")
        axs["box_rot"].set_ylabel("Rotation error [deg]")
        axs["box_rot"].set_xticklabels([str(i) for i in trajec_lenghts])

        dist_travelled = [trajec_lenghts[0] * i for i in range(len(scale_drifts[0]))]

        axs["scale"].plot(dist_travelled, scale_drifts[0])
        axs["scale"].set_title("Scale Drift")
        axs["scale"].set_xlabel("Lenght along the total trajectory [m]")
        axs["scale"].set_ylabel("Scaling factor wrt. ground truth")
        axs["scale"].grid(True)


        # ---------- PLOT 3D ---------- #

        w_t_wc__x = split_trajecs[-1][:, 0]
        w_t_wc__y = split_trajecs[-1][:, 1]
        w_t_wc__z = split_trajecs[-1][:, 2]

        gt_w_t_wc__x = self.gt_T_wc_array[0::3, 3]
        gt_w_t_wc__y = self.gt_T_wc_array[1::3, 3]
        gt_w_t_wc__z = self.gt_T_wc_array[2::3, 3]

        fig1 = plt.figure()
        ax = fig1.add_subplot(projection="3d")
        ax.plot(gt_w_t_wc__x, gt_w_t_wc__y, gt_w_t_wc__z, color="r")
        for i in range(len(split_pos) - 1):
            ax.plot(w_t_wc__x[split_pos_s[-1][i]:split_pos_s[-1][i + 1]], 
                    w_t_wc__y[split_pos_s[-1][i]:split_pos_s[-1][i + 1]], 
                    w_t_wc__z[split_pos_s[-1][i]:split_pos_s[-1][i + 1]], color="b")
            ax.plot(w_t_wc__x[split_pos_s[-1][i]], 
                    w_t_wc__y[split_pos_s[-1][i]], 
                    w_t_wc__z[split_pos_s[-1][i]], "go", markersize=3)
        ax.set_xlabel("$X_w$")
        ax.set_ylabel("$Y_w$")
        ax.set_zlabel("$Z_w$")
        ax.set_title("Relative Trajectory Error")
        ax.legend(["Ground truth", "VO estimate", "Subtrajectory"])

        # Compute the limits for the plot; same scale for both axes
        xmin = np.min(np.r_[w_t_wc__x, gt_w_t_wc__x]) - 1
        xmax = np.max(np.r_[w_t_wc__x, gt_w_t_wc__x]) + 1
        ymin = np.min(np.r_[w_t_wc__y, gt_w_t_wc__y]) - 1
        ymax = np.max(np.r_[w_t_wc__y, gt_w_t_wc__y]) + 1
        zmin = np.min(np.r_[w_t_wc__z, gt_w_t_wc__z]) - 1
        zmax = np.max(np.r_[w_t_wc__z, gt_w_t_wc__z]) + 1

        max_range = max(xmax - xmin, zmax - zmin, ymax - ymin)
        mid_x = 0.5 * (xmin + xmax)
        mid_z = 0.5 * (zmin + zmax)
        mid_y = 0.5 * (ymin + ymax)

        # Set the computed limits
        ax.set_xlim(mid_x - 0.5 * max_range, mid_x + 0.5 * max_range)
        ax.set_ylim(mid_y - 0.5 * max_range, mid_y + 0.5 * max_range)
        ax.set_zlim(mid_z - 0.5 * max_range, mid_z + 0.5 * max_range)

        # Have the camera view be the orientation of the world coord sys.
        ax.view_init(elev=-80, azim=-90, roll=0)

        plt.show()

    def re_get_split_pos(self, trajectory_length: float):
        """
        Get the locations in the ground truth and the estimate trajectories at which 
        they should be split into sub-trajectories. A split is made where the trans-
        lation vectors of the ground truth add to trajectory_length. 

        ### Parameters
        1. trajectory_lenght : float
            - Minimum length of each sub-trajectory in meters
        """

        split_pos = [0]
        travelled = 0 
        previous_t = self.gt_T_wc_list[0][:, 3]

        for pos, gt_T_wc in enumerate(self.gt_T_wc_list):
            
            # The difference in translation compared to the previous pose
            travel_vec = gt_T_wc[:, 3] - previous_t

            # Increment the distance counter by the L2 norm of the translation
            travelled += np.sqrt(np.sum(travel_vec**2))

            if travelled >= trajectory_length:
                split_pos.append(pos)
                travelled = 0
            
            previous_t = gt_T_wc[:, 3]

        return split_pos


if __name__ == "__main__":
    
    # Set the name and first frame. Make sure the trajectory file is in the directory
    dl = DataLoader("kitti", preload=False)
    te = TrajectoryEval(dataset_name=dl.dataset_str, first_frame=dl.init_frames[1])

    # To show the relative error:

    # Some suggestions for subtrajectory lengths (in meters)
    trajec_length_kitti = (7, 23, 31, 100)
    trajec_length_parking = (1, 5, 8)
    trajec_length_malaga = (20, 30)
    te.relative_error(trajec_lenghts=trajec_length_kitti)

    # To compute the absolute error and show the resulting plot:
    te.similarity_transform_3d()
    te.draw_trajectory(gt=True)
    print(f"Root mean squared position ATE: {te.absolue_trajectory_error()}")