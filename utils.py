"""
File of functions that are needed for both continuous operation and
bootstrapping
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import cv2

linear_LS_triangulation_C = -np.eye(2, 3)
def linear_LS_triangulation(u1, P1, u2, P2):
    """
    Linear Least Squares based triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector will be True for all points.
    """
    A = np.zeros((4, 3))
    b = np.zeros((4, 1))
    
    # Create array of triangulated points
    x = np.zeros((3, len(u1)))
    
    # Initialize C matrices
    C1 = np.array(linear_LS_triangulation_C)
    C2 = np.array(linear_LS_triangulation_C)
    
    for i in range(len(u1)):
        # Derivation of matrices A and b:
        # for each camera following equations hold in case of perfect point matches:
        #     u.x * (P[2,:] * x)     =     P[0,:] * x
        #     u.y * (P[2,:] * x)     =     P[1,:] * x
        # and imposing the constraint:
        #     x = [x.x, x.y, x.z, 1]^T
        # yields:
        #     (u.x * P[2, 0:3] - P[0, 0:3]) * [x.x, x.y, x.z]^T     +     (u.x * P[2, 3] - P[0, 3]) * 1     =     0
        #     (u.y * P[2, 0:3] - P[1, 0:3]) * [x.x, x.y, x.z]^T     +     (u.y * P[2, 3] - P[1, 3]) * 1     =     0
        # and since we have to do this for 2 cameras, and since we imposed the constraint,
        # we have to solve 4 equations in 3 unknowns (in LS sense).

        # Build C matrices, to construct A and b in a concise way
        C1[:, 2] = u1[i, :]
        C2[:, 2] = u2[i, :]
        
        # Build A matrix:
        # [
        #     [ u1.x * P1[2,0] - P1[0,0],    u1.x * P1[2,1] - P1[0,1],    u1.x * P1[2,2] - P1[0,2] ],
        #     [ u1.y * P1[2,0] - P1[1,0],    u1.y * P1[2,1] - P1[1,1],    u1.y * P1[2,2] - P1[1,2] ],
        #     [ u2.x * P2[2,0] - P2[0,0],    u2.x * P2[2,1] - P2[0,1],    u2.x * P2[2,2] - P2[0,2] ],
        #     [ u2.y * P2[2,0] - P2[1,0],    u2.y * P2[2,1] - P2[1,1],    u2.y * P2[2,2] - P2[1,2] ]
        # ]
        A[0:2, :] = C1.dot(P1[0:3, 0:3])    # C1 * R1
        A[2:4, :] = C2.dot(P2[0:3, 0:3])    # C2 * R2
        
        # Build b vector:
        # [
        #     [ -(u1.x * P1[2,3] - P1[0,3]) ],
        #     [ -(u1.y * P1[2,3] - P1[1,3]) ],
        #     [ -(u2.x * P2[2,3] - P2[0,3]) ],
        #     [ -(u2.y * P2[2,3] - P2[1,3]) ]
        # ]
        b[0:2, :] = C1.dot(P1[0:3, 3:4])    # C1 * t1
        b[2:4, :] = C2.dot(P2[0:3, 3:4])    # C2 * t2
        b *= -1
        
        # Solve for x vector
        cv2.solve(A, b, x[:, i:i+1], cv2.DECOMP_SVD)
    
    return x.T.astype(float), np.ones(len(u1), dtype=bool)


def check_inside_FOV(alpha, w, h, p):
    """
    Removes the landmarks that are outside the camers's field of view

    ### Parameters
    - alpha : float
        - focal length in pixels
    - w : int
        - width of the image in pixels
    - h : int
        - height of the image in pixels
    - p : np.array (n x 3)
        - landmarks

    ### Returns
    - boolean np.array shape (n,) 
    """

    # Vectors in the direction of the image corners
    ul = np.array([-0.5 * w, -0.5 * h, alpha])  # upper left
    ll = np.array([-0.5 * w, 0.5 * h, alpha])   # lower left
    ur = np.array([0.5 * w, -0.5 * h, alpha])   # upper right
    lr = np.array([0.5 * w, 0.5 * h, alpha])    # lower right

    # Normal vectors of the planes limiting the FOV
    n_l = np.cross(ll, ul)      # normal vector left
    n_b = np.cross(lr, ll)      # normal vector bottom
    n_r = np.cross(ur, lr)      # normal vector right
    n_t = np.cross(ul, ur)      # normal vecotr top

    # Checking the constraint: dot product with each normal vector must be
    # bigger / equal to zero
    m = np.vstack((n_l, n_b, n_r, n_t))
    product = m @ p.T

    return np.all(product >= 0, axis=0)


def is_new_keypoint(cv2keypoint, existing_keypoints: np.array, threshold):
    """
    Determine if the cv2 keypoint passed in is new or not

    ### Parameters
    1. cv2keypoint : cv2.Keypoint object
        - keypoint to compare, usually result of feature detector
    2. existing_keypoints : np.array (N x 2)
        - array of existing keypoints to compare against
    3. threshold : float
        the threshold value for two keypoints to be considered the same

    ### Returns
    - bool
        - True for inliers and False for outliers
    """
    distances = np.linalg.norm(existing_keypoints - cv2keypoint.pt, axis=1)
    return not np.any(distances < threshold)


def median_outliers(p: np.array, x_tol=None, y_tol=None, z_tol=15) -> np.array:
    """
    Check if the points are outliers by how far they are from the median
    deviation of the median. When the values are all identical, no out-
    liers can be found.

    ### Parameters
    1. p : np.array shape (n x 3)
        - Array of landmark locations
    2. x_tol : int (default 15)
        - Tolerance in terms of the relative deviation of points from the 
          median (x_tol * median of deviations from the median are allowed)
        - Set None for not enforcing any constraint
    3. y_tol : int 
    4. z_tol : int

    ### Returns
    - np.array shape (n,) dtype bool
        - True for inliers and False for outliers
    """

    if x_tol is not None:
        dev = np.abs(p[:, 0] - np.median(p[:, 0]))
        m_of_dev = np.median(dev)
        if m_of_dev != 0.:
            x_bool = (dev / m_of_dev <= x_tol)
    else:
        x_bool = np.ones(p.shape[0], dtype=bool)

    if y_tol is not None:
        dev = np.abs(p[:, 1] - np.median(p[:, 1]))
        m_of_dev = np.median(dev)
        if m_of_dev != 0.:
            y_bool = (dev / m_of_dev <= y_tol)
    else:
        y_bool = np.ones(p.shape[0], dtype=bool)

    if z_tol is not None:
        dev = np.abs(p[:, -1] - np.median(p[:, -1]))
        m_of_dev = np.median(dev)
        if m_of_dev != 0.:
            z_bool = (dev / m_of_dev <= z_tol)
    else:
        z_bool = np.ones(p.shape[0], dtype=bool)

    conditions = np.vstack((x_bool, y_bool, z_bool))
    return np.all(conditions, axis=0)


def multiply_transformation(transform_ab, transform_bc):
    """
    Computes the combined transformation matrix T_ab X T_bc = T_ac
    ### Parameters
    - transform np.array shape (3, 4)
        - A standard transformation matrix T_ab, converting point from frame b to frame a
    - transformations np.array shape (3, 4)
        - A standard transformation matrix T_bc, converting a point from frame c to frame b

    ### Returns
    -  np.array shape (3 x 4)
        - A standard transformation matrix T_ac, converting a point from frame c to frame a
    """
    R_ac = transform_ab[:, :3] @ transform_bc[:, :3]
    t_ac = transform_ab[:, :3] @ transform_bc[:, 3] + transform_ab[:, 3]
    return np.column_stack((R_ac, t_ac))


def inverse_transformation(transformation):
    """
    Computes the inverse of a standard transformation matrix
    ### Parameters
    - transformations np.array shape (3, 4)
        - A standard transformation matrix, converting a point from frame a to frame b

    ### Returns
    -  np.array shape (3 x 4)
        - A standard transformation matrix , converting a point from frame b to frame a
    """
    R_ba = transformation[:, :3]
    b_t_ba = transformation[:, 3]

    R_ab = R_ba.T
    a_t_ab = - R_ab @ b_t_ba
    return np.column_stack((R_ab, a_t_ab))


def triangulate_points(k, r, t, p1, p2):
    """
    ### Parameters
    - p1 np.array shape (n x 2)
        - From the left image I1
    - p2 
        - From the right image I2
    - t np.array shape (3 x 1)
    """

    n_points = p1.shape[0]

    # Get the projection matrix
    rt = np.hstack((r, t))
    io = np.hstack((np.eye(3), np.zeros_like(t)))
    m1 = k @ io
    m2 = k @ rt

    # Fill the triangulated points into this array:
    big_p = np.zeros((n_points, 3))

    for i in range(n_points):

        # cross product p1
        p_1 = p1[i]
        cr_p1 = np.array([[0, -1, p_1[1]],
                          [1, 0, -p_1[0]],
                          [-p_1[1], p_1[0], 0]])
        # cross product p2
        p_2 = p2[i]
        cr_p2 = np.array([[0, -1, p_2[1]],
                          [1, 0, -p_2[0]],
                          [-p_2[1], p_2[0], 0]])

        # Matrix for triangulation:
        triang = np.vstack((cr_p1 @ m1, cr_p2 @ m2))

        # Find eigenvectors
        e = np.linalg.eig(triang.T @ triang)
        min_eval = np.argmin(e.eigenvalues)
        big_p4 = e.eigenvectors[min_eval]
        big_p3 = (big_p4 / big_p4[-1])[:3]
        big_p[i] = big_p3

    return big_p


def rotation_matrix_to_euler_angles(R):
    # Ensure the matrix is valid (orthogonal and determinant of 1)
    # Yaw (ψ), Pitch (θ), Roll (φ) calculations
    pitch = np.arctan2(-R[2, 0], np.sqrt(R[0, 0]**2 + R[1, 0]**2))
    roll = np.arctan2(R[2, 1], R[2, 2])
    yaw = np.arctan2(R[1, 0], R[0, 0])

    # Return as degrees
    return np.degrees(pitch), np.degrees(roll), np.degrees(yaw)


class DrawTrajectory():

    def __init__(self, b, save=False):
        """
        Initialise with the bootstrap

        ### Parameters
        1. b : initialise_vo.Bootstrap
            - Bootstrapping instance that provides:
                - t : np.array. The estimated pose of the camera at the last 
                bootstrapping frame
                - p : np.array. The keypoints in P after bootstrapping
                - x : np.array. The landmarks in X after bootstrapping
                - im : np.array. Provided through b's dataloader; the last
                bootstrapping frame
        2. save : bool
            - Specify whether the plots are to be saved in a dedicated folder
            for later in-depth inspection. When save is true, the plots are
            not displayed.
        """

        # Get information from Bootstrapping object
        p = b.keypoints
        c = []
        x = b.triangulated_points
        self.frame = b.init_frames[-1]
        im = b.data_loader[self.frame]

        c_map = plt.get_cmap("plasma")  # cmap style (if mark_depth)

        # Computing the location of the camera, saving the x and z components
        c_R_cw = b.transformation_matrix[:, :3]
        c_t_cw = b.transformation_matrix[:, -1]
        w_t_wc = -1 * c_R_cw.T @ c_t_cw
        self.w_t_wc_x = [w_t_wc[0]]
        self.w_t_wc_y = [w_t_wc[1]]
        self.w_t_wc_z = [w_t_wc[2]]

        # Computing the depth component
        z_c = (x @ c_R_cw.T)[:, 2]  # compute z_c

        # Setting the directories for saving if desired
        self.save = save
        if self.save:
            plot_dir_general = Path.cwd().joinpath("vo_plots")
            if not plot_dir_general.is_dir():
                plot_dir_general.mkdir()
            dataset_str = b.data_loader.dataset_str
            self.plot_dir_dataset = plot_dir_general.joinpath(dataset_str)
            if not self.plot_dir_dataset.is_dir():
                self.plot_dir_dataset.mkdir()

        # Plot settings
        plt.ion()
        self.fig, axs = plt.subplot_mosaic(
            [["u", "u", "u"], ["m", "l", "r"]],
            figsize=(10, 8),                     # (width x height)
            layout="constrained"
        )
        self.u = axs["u"]
        #self.l = self.fig.add_subplot(2, 3, 5)
        self.l = axs["l"]
        self.r = axs["r"]
        self.m = axs["m"]
        self.fig.suptitle(f"Image i = {self.frame}")

        # Upper plot:
        self.u_im = self.u.imshow(im, cmap="grey")
        self.u_candidatepoints = self.u.scatter(p[:, 0], p[:, 1], marker="x",
                                          alpha=0.5, c='g', s=25)
        self.u_keypoints = self.u.scatter(p[:, 0], p[:, 1], marker="o",
                                          alpha=1, cmap=c_map, c=z_c, s=5)
        self.cbar = self.fig.colorbar(self.u_keypoints, orientation="vertical")
        self.cbar.set_label("Distance from camera, SFM units")
        self.u.set_title("Current image with landmarks")

        # Right: xz landmark graphic
        self.r_landmarks = self.r.scatter(x[:, 0], x[:, 2], marker="o",
                                          alpha=0.5, cmap=c_map, c=z_c)
        self.r_prev_pos = self.r.plot(0, 0, marker="x", c="b", markersize=5,
                                      linestyle=None)[0]
        self.r_pos = self.r.plot(w_t_wc[0], w_t_wc[2], c="b", marker="X",
                                 markersize=10)[0]
        self.r.set_xlabel("$X_w$ - SFM units")
        self.r.set_ylabel("$Z_w$ - SFM units")
        self.r.set_ylim(-200, 200)
        self.r.autoscale(False)
        self.r.grid(visible=True, axis="both")
        self.r.set_title("Landmarks $P$ in world frame")

        # Left: xz camera position
        self.l_pos = self.l.plot(self.w_t_wc_x, self.w_t_wc_z, color='blue', linewidth=2)[0]
        self.l.grid(visible=True)
        self.l.set_xlabel("Camera poses in world frame")
        self.l.set_xlabel("$X_w$ - SFM units")
        self.l.set_ylabel("$Z_w$ - SFM units")


        # camera rotation
        pitch, roll, yaw = rotation_matrix_to_euler_angles(c_R_cw)
        angles = [roll, yaw, pitch]
        labels = ['Pitch', 'Roll', 'Yaw']
        colors = ['r', 'g', 'b']
        self.m_pos = self.m.bar(labels, angles, color=colors)
        self.m.set_title("Pitch, Roll, and Yaw Angles")
        self.m.set_ylabel("Angle (degrees)")
        self.m.set_ylim([-180, 180])
        self.m.grid(True)

        if self.save:
            self.fig.savefig(self.plot_dir_dataset.joinpath(f"{self.frame:0{4}}.jpg"))
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()

    def update_data(self, t: np.array, p: np.array, x: np.array, c: np.array,
                    im: np.array):
        """
        Update the plot

        ### Parameters
        1. t : np.array
            - Latest transformation matrix from homogeneous world coordinates
            to (X, Y, Z) in camera coordinates. A (3 x 4) matrix
        2. p : np.array
            - The current keypoints in set Pi belonging to the landmarks
            xi. Shape (n x 2)
        3. x : np.arrays
            - Np.array containing the landmarks currently used for camera
            localisation on its rows. (n x 3). The landmarks are given in
            the world coordinate system.
        4. im : np.array
            - Image
        """

        ### ------- PROCESS DATA ------- ###

        # Computing the current location of the camera, saving x and z comp.
        c_R_cw = t[:, :3]
        c_t_cw = t[:, -1]
        w_t_wc = -1 * c_R_cw.T @ c_t_cw
        self.w_t_wc_x.append(w_t_wc[0])
        self.w_t_wc_z.append(w_t_wc[2])
        self.w_t_wc_y.append(w_t_wc[1])

        # Computing the depth component
        z_c = (x @ c_R_cw.T)[:, 2]  # compute z_c

        ### ------- UPDATE FIGURE DATA ------- ###

        # Upper plot
        self.u_im.set(data=im)
        self.u_keypoints.set_offsets(p)
        self.u_candidatepoints.set_offsets(c)
        self.r_landmarks.set_array(z_c)  # Upadte the colour mapping
        self.u_keypoints.set_array(z_c)

        # Right plot
        self.r_pos.set_data([w_t_wc[0]], [w_t_wc[2]])
        self.r_landmarks.set_offsets(np.c_[x[:, 0], x[:, 2]])
        self.r_landmarks.set_array(z_c)  # Upadte the colour mapping
        self.r_prev_pos.set_data(self.w_t_wc_x, self.w_t_wc_z)

        # Left plot
        self.l_pos.set_data(self.w_t_wc_x, self.w_t_wc_z)
        #self.l_pos.set_3d_properties(self.w_t_wc_z)

        # rotation plot
        r, p, y = rotation_matrix_to_euler_angles(c_R_cw)
        self.m_pos[0].set_height(r)  # Update the plot data with new values
        self.m_pos[1].set_height(p)
        self.m_pos[2].set_height(y)

        ### ------- UPDATE FIGURE SCALE ------- ###

        # Left plot
        # Compute the limits for the plot; same scale for both axes
        l_xmin, l_xmax = min(self.w_t_wc_x) - 1, max(self.w_t_wc_x) + 1
        l_zmin, l_zmax = min(self.w_t_wc_z) - 1, max(self.w_t_wc_z) + 1
        max_range = max(l_xmax - l_xmin, l_zmax - l_zmin)
        mid_x = 0.5 * (l_xmin + l_xmax)
        mid_z = 0.5 * (l_zmin + l_zmax)
        self.l.set_xlim(mid_x - 0.5 * max_range, mid_x + 0.5 * max_range)
        self.l.set_ylim(mid_z - 0.5 * max_range, mid_z + 0.5 * max_range)

        # Right plot
        # Making sure the landmark positions are visible:
        r_xmax = max(x[:, 0].max(), w_t_wc[0]) + 10
        r_xmin = min(x[:, 0].min(), w_t_wc[0]) - 10
        r_zmax = max(x[:, 2].max(), w_t_wc[2]) + 10
        r_zmin = min(x[:, 2].min(), w_t_wc[2]) - 10
        self.r.set_xlim(r_xmin, r_xmax)
        self.r.set_ylim(r_zmin, r_zmax)

        # Update the colour bar scale
        norm = Normalize(vmin=0, vmax=z_c.max())
        self.u_keypoints.set_norm(norm)
        self.r_landmarks.set_norm(norm)
        self.cbar.update_normal(self.u_keypoints)

        ### ------- UPDATE PLOTS ------- ###

        self.frame += 1
        self.fig.suptitle(f"Image i = {self.frame}")
        if self.save:
            self.fig.savefig(self.plot_dir_dataset.joinpath(f"{self.frame:0{4}}.jpg"))
        else:
            self.fig.canvas.draw()
            self.fig.canvas.flush_events()
