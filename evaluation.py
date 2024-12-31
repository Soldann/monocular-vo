"""
Module for trajectory evaluation
"""

from matplotlib import pyplot as plt
from pathlib import Path
import pickle
import numpy as np
from cv2 import Rodrigues

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

        # Load T_cw_list from .pkl file in "solution_trajectories"
        trajec_dir = Path().cwd().joinpath("solution_trajectories")
        poses_path = trajec_dir.joinpath(dataset_name + ".pkl")
        with open(poses_path, "rb") as f:
            self.T_cw_list = pickle.load(f)
        
        # Find the inverse transforms T_wc
        self.T_wc_list = [inverse_transformation(T_wc) for T_wc in self.T_cw_list]
        self.T_wc_array = np.vstack(self.T_wc_list)

        # ---------- LOAD GT TRAJECTORY FROM FILE ---------- #
        
        # Depending on the passed datset_name, load the ground truth trajectory
        if dataset_name == "parking":
            
            # Load ground truth poses
            gt_poses_path = Path.cwd().joinpath("datasets", "parking", "poses.txt")
            gt_poses_array = np.loadtxt(gt_poses_path.as_posix())

            # Find the window of ground truth poses that are needed & reshape to list
            self.n_poses = min(gt_poses_array.shape[0], len(self.T_wc_list))
            gt_poses_array = gt_poses_array[first_frame:(first_frame + self.n_poses)]
            self.gt_T_wc_array = gt_poses_array.reshape(-1, 4)
            self.gt_T_wc_list = [self.gt_T_wc_array[(3 * i):(3 * i + 3)] 
                                 for i in range(self.n_poses)]

        elif dataset_name == "kitti":

            # Load ground truth poses
            gt_poses_path = Path.cwd().joinpath("datasets", "kitti", "poses", "05.txt")
            gt_poses_array = np.loadtxt(gt_poses_path.as_posix())

            # Find the window of ground truth poses that are needed & reshape to list
            self.n_poses = min(gt_poses_array.shape[0], len(self.T_wc_list))
            gt_poses_array = gt_poses_array[first_frame:(first_frame + self.n_poses)]
            self.gt_T_wc_array = gt_poses_array.reshape(-1, 4)
            self.gt_T_wc_list = [self.gt_T_wc_array[(3 * i):(3 * i + 3)] 
                                 for i in range(self.n_poses)]
        
        elif dataset_name == "malaga":
            
            # Load ground truth poses
            gt_poses_path = Path.cwd().joinpath("datasets", "malaga-urban-dataset-extract-07",
                                                "malaga-urban-dataset-extract-07_all-sensors_GPS.txt")
            gt_poses_array = np.loadtxt(gt_poses_path.as_posix(), skiprows=1)

            # TODO: finish malaga implementation
            pass
        
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
        ax.set_title("VO Trajectory Positions")

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
        w_t_wc = self.T_wc_array[:, 3].reshape(-1, 3)
        gt_w_t_wc = self.gt_T_wc_array[:, 3].reshape(-1, 3)

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

        """
        # Vector error measures
        delta_R = [gt_T_wc[:, :3] @ T_wc[:, :3].T 
                   for gt_T_wc, T_wc in 
                   zip(self.gt_T_wc_list, self.T_cw_list)]
        delta_p = [gt_T_wc[:, 3] - delta_R @ T_wc[:, 3] 
                   for gt_T_wc, delta_R, T_wc in 
                   zip(self.gt_T_wc_list, delta_R, self.T_wc_list)]
        
        # Compute ATE for rotation
        angle_delta_R_list = [Rodrigues(delta_R)[0]         # answer in degrees
                              for delta_R in delta_R]
        angle_delta_R_array = np.vstack(angle_delta_R_list)
        ate_rot = np.sqrt(np.sum(angle_delta_R_array**2) / self.n_poses)

        # Compute ATE for position
        delta_p_array = np.hstack(delta_p)
        ate_pos = np.sqrt(np.sum(delta_p_array**2) / self.n_poses)
        """

        # Compute the RMSE as shown on the lecture slides

        RMSE = np.sqrt(np.sum((self.gt_T_wc_array[:, 3] 
                               - self.T_wc_array[:, 3])**2) / self.n_poses)
        
        return RMSE

if __name__ == "__main__":
    te = TrajectoryEval(dataset_name="kitti", first_frame=2)
    te.similarity_transform_3d()
    print(te.absolue_trajectory_error())
    te.draw_trajectory(gt=True)