

class VO:

    def __init__(self):
                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                  
        # The current state of the pipeline
        #   P_i    In state i: the set of all current 2D keypoints
        #   X_i    In state i: the set of all current 3D landmarks
        #   C_i    In state i: the set of candidate 2D keypoints currently 
        #          being tracked
        #   F_i    In state i: for each candidate keypoint in C_i, its 
        #          position in the first frame it was tracked in
        #   T_i    In state i: the camera pose at the first observation of
        #          each keypoint in C_i
        self.pi = None
        self.ci = None
        self.fi = None
        self.ti = None