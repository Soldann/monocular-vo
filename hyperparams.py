class Hyperparameters:
    def __init__(self):
        self.hyperparams = {}

        parking = HyperparameterInstance()
        kitti = HyperparameterInstance()
        kitti.max_kp = 2500
        kitti.block_radius = 13

        # malaga optimize / not optimize
        malaga = HyperparameterInstance()
        malaga.block_radius = 60
        malaga.max_kp = 1250
        malaga.do_sift = True
        malaga.harris_count = 50
        malaga.harris_threshold = 0.1
        malaga.do_optimize = False

        # Kitti sparse
        # kitti = HyperparameterInstance()
        # kitti.harris_threshold = 0.1
        # kitti.harris_count = 600
        # kitti.max_kp = 500
        # kitti.block_radius = 40

        # Kitti optimize
        # kitti = HyperparameterInstance()
        # kitti.harris_threshold = 0.01
        # kitti.harris_count = 2500
        # kitti.do_sift = False
        # kitti.max_kp = 700
        # kitti.block_radius = 20
        # kitti.w_split = 1
        # kitti.do_optimize = True

        own1 = HyperparameterInstance()

        own2 = HyperparameterInstance()
        own2.do_ransac1 = True

        self.hyperparams["parking"] = parking
        self.hyperparams["kitti"] = kitti
        self.hyperparams["malaga"] = malaga
        self.hyperparams["own_1"] = own1
        self.hyperparams["own_2"] = own2

    def get_params(self, name):
        return self.hyperparams[name]

    def print_params(self, name):
        hp = self.hyperparams[name]
        print(f"--- Hyperparameters for {name} ---")
        hp.print_params()
        print("--- End ---")



class HyperparameterInstance:
    def __init__(self):
        self.max_kp = 700
        self.w_split = 2
        self.h_split = 1
        self.harris_count = 2500
        self.harris_threshold = 0.05
        self.do_sift = False
        self.do_optimize = False
        self.do_ransac1 = False
        self.ransac1_bincount = 30
        self.angle_threshold = 0.01
        self.ransac_acc = 6
        self.block_radius = 20
    
    def print_params(self):
        print("w_split: ", self.w_split)
        print("h_split: ", self.h_split)
        print("harris_count: ", self.harris_count)
        print("harris_threshold: ", self.harris_threshold)
        print("do_sift: ", self.do_sift)
        print("do_optimize: ", self.do_optimize)
        print("do_ransac1: ", self.do_ransac1)
        print("ransac1_bincount: ", self.ransac1_bincount)
        print("angle_threshold: ", self.angle_threshold)
        print("ransac_acc: ", self.ransac_acc)
        print("max_kp: ", self.max_kp)
        print("block_radius: ", self.block_radius)

