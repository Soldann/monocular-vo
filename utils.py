"""
File of functions that are needed for both continuous operation and
bootstrapping
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize

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
    for point in existing_keypoints:
        distance = np.linalg.norm(cv2keypoint.pt - point)
        if distance < threshold:
            return False
    return True

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
    R_ac = transform_ab[:,:3] @ transform_bc[:,:3]
    t_ac = transform_ab[:,:3] @ transform_bc[:,3] + transform_ab[:,3]
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
    R_ba = transformation[:,:3]
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


class DrawTrajectory():

    def __init__(self, b):
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
        """
        
        # Get information from Bootstrapping object
        p = b.keypoints
        x = b.triangulated_points
        self.frame = b.init_frames[-1]
        im = b.data_loader[self.frame]

        c_map = plt.get_cmap("nipy_spectral")  # cmap style (if mark_depth)

        # Computing the location of the camera, saving the x and z components
        c_R_cw = b.transformation_matrix[:, :3]
        c_t_cw = b.transformation_matrix[:, -1]
        w_t_wc = -1 * c_R_cw.T @ c_t_cw
        self.w_t_wc_x = [w_t_wc[0]]
        self.w_t_wc_z = [w_t_wc[2]]

        # Computing the depth component
        z_c = (x @ c_R_cw.T)[:, 2]  # compute z_c

        # Plot settings
        plt.ion()
        self.fig, axs = plt.subplot_mosaic(
            [["u", "u"], ["l", "r"]], 
            figsize=(10, 8),                     # (width x height)
            layout="constrained"
                                           )
        self.u = axs["u"]
        self.l = axs["l"]
        self.r = axs["r"]
        self.fig.suptitle(f"Image i = {self.frame}")

        # Upper plot:
        self.u_im = self.u.imshow(im, cmap="grey")
        self.u_keypoints = self.u.scatter(p[:, 0], p[:, 1], marker="o", 
                                            alpha=0.5, cmap=c_map, c=z_c, s=2)
        self.cbar = self.fig.colorbar(self.u_keypoints, orientation="vertical")
        self.cbar.set_label("Distance from camera, SFM units")


        # Right: xz landmark graphic
        self.r_landmarks = self.r.scatter(x[:, 0], x[:, 2], marker="o", 
                                            alpha=0.5, cmap=c_map, c=z_c)
        self.r_prev_pos = self.r.plot(0, 0, marker="x", c="b", markersize=5,
                                      linestyle=None)[0]
        self.r_pos = self.r.plot(w_t_wc[0], w_t_wc[2], c="b", marker="X", 
                                 markersize=10)[0]       
        self.r.set_xlabel("$X_c$ - SFM units") 
        self.r.set_ylabel("$Z_c$ - SFM units")
        self.r.grid(visible=True, axis="both")

        # Left: xz camera position
        self.l_pos = self.l.plot(self.w_t_wc_x, self.w_t_wc_z, marker="o")[0]
        self.l.grid(visible=True, axis="both")
        
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()

    def update_data(self, t: np.array, p: np.array, x: np.array,
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

        # Computing the depth component
        z_c = (x @ c_R_cw.T)[:, 2]  # compute z_c

        ### ------- UPDATE FIGURE DATA ------- ###

        # Upper plot
        self.u_im.set(data=im)
        self.u_keypoints.set_offsets(p)
        self.r_landmarks.set_array(z_c)  # Upadte the colour mapping
        self.u_keypoints.set_array(z_c)

        # Right plot
        self.r_pos.set_data([w_t_wc[0]], [w_t_wc[2]])
        self.r_landmarks.set_offsets(np.c_[x[:, 0], x[:, 2]])
        self.r_landmarks.set_array(z_c)  # Upadte the colour mapping
        self.r_prev_pos.set_data(self.w_t_wc_x, self.w_t_wc_z)
        
        # Left plot
        self.l_pos.set_data(self.w_t_wc_x, self.w_t_wc_z)

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
        self.fig.canvas.draw()
        self.fig.canvas.flush_events()
