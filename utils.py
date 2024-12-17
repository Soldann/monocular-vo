"""
File of functions that are needed for both continuous operation and
bootstrapping
"""

# from tqdm import tqdm
import numpy as np

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

