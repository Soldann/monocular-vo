"""
File of functions that are needed for both continuous operation and
bootstrapping
"""

import numpy as np
from matplotlib import pyplot as plt
from matplotlib.colors import Normalize
from pathlib import Path
import cv2
from scipy.linalg import expm, logm
import numpy as np
from matplotlib.patches import FancyArrow

def twist2HomogMatrix(twist):
    """
    twist2HomogMatrix Convert twist coordinates to 4x4 homogeneous matrix
    Input: -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]
    Output: -H(4,4): Euclidean transformation matrix (rigid body motion)
    """
    v = twist[:3]  # linear part
    w = twist[3:]   # angular part

    se_matrix = np.concatenate([cross2Matrix(w), v[:, None]], axis=1)
    se_matrix = np.concatenate([se_matrix, np.zeros([1, 4])], axis=0)

    H = expm(se_matrix)

    return H


def HomogMatrix2twist(H):
    """
    HomogMatrix2twist Convert 4x4 homogeneous matrix to twist coordinates
    Input:
     -H(4,4): Euclidean transformation matrix (rigid body motion)
    Output:
     -twist(6,1): twist coordinates. Stack linear and angular parts [v;w]

    Observe that the same H might be represented by different twist vectors
    Here, twist(4:6) is a rotation vector with norm in [0,pi]
    """

    se_matrix = logm(H)

    # careful for rotations of pi; the top 3x3 submatrix of the returned se_matrix by logm is not
    # skew-symmetric (bad).

    v = se_matrix[:3, 3]

    w = Matrix2Cross(se_matrix[:3, :3])
    twist = np.concatenate([v, w])

    return twist


def cross2Matrix(x):
    """ Antisymmetric matrix corresponding to a 3-vector
     Computes the antisymmetric matrix M corresponding to a 3-vector x such
     that M*y = cross(x,y) for all 3-vectors y.

     Input: 
       - x np.ndarray(3,1) : vector

     Output: 
       - M np.ndarray(3,3) : antisymmetric matrix
    """
    M = np.array([[0,   -x[2], x[1]], 
                  [x[2],  0,  -x[0]],
                  [-x[1], x[0],  0]])
    return M


def Matrix2Cross(M):
    """
    Computes the 3D vector x corresponding to an antisymmetric matrix M such that M*y = cross(x,y)
    for all 3D vectors y.
    Input:
     - M(3,3) : antisymmetric matrix
    Output:
     - x(3,1) : column vector
    See also CROSS2MATRIX
    """
    x = np.array([-M[1, 2], M[0, 2], -M[0, 1]])

    return x

linear_LS_triangulation_C = -np.eye(2, 3)
def linear_LS_triangulation(u1, P1, u2, P2):
    """
    Linear Least Squares based triangulation.
    Relative speed: 0.1
    
    (u1, P1) is the reference pair containing normalized image coordinates (x, y) and the corresponding camera matrix.
    (u2, P2) is the second pair.
    
    u1 and u2 are matrices: amount of points equals #rows and should be equal for u1 and u2.
    
    The status-vector will be True for all points.

    This code is from:
    https://github.com/Eliasvan/Multiple-Quadrotor-SLAM/blob/master/Work/python_libs/triangulation.py

    We felt it is okay to use it, as it does output the same results as the OpenCV triangulation function, just more accurate.

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

def drawCamera(ax, position, direction, length_scale = 1, head_size = 1, 
        equal_axis = True, set_ax_limits = True):
    # Draws a camera consisting of arrows into a 1d Plot
    # ax            axes object, creates as follows
    #                   fig = plt.figure()
    #                   ax = fig.add_subplot(projection='3d')
    # position      np.array(3,) containing the camera position
    # direction     np.array(3,3) where each column corresponds to the [x, y, z]
    #               axis direction
    # length_scale  length scale: the arrows are drawn with length
    #               length_scale * direction
    # head_size     controls the size of the head of the arrows
    # equal_axis    boolean, if set to True (default) the axis are set to an 
    #               equal aspect ratio
    # set_ax_limits if set to false, the plot box is not touched by the function
    head_size = length_scale / 5
    arrow_prop_dict = dict(head_width=head_size, head_length=head_size, color='r')

    a = FancyArrow(position[0], position[2], length_scale * direction[0, 0], length_scale * direction[2, 0], **arrow_prop_dict)
    ax.add_artist(a)
    arrow_prop_dict = dict(head_width=head_size, head_length=head_size, color='g')

    a = FancyArrow(position[0], position[2], length_scale * direction[0, 2], length_scale * direction[2, 2], **arrow_prop_dict)
    ax.add_artist(a)

