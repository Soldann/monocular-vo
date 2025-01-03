
from pathlib import Path
import pickle
from cont_vo import VO
from initialise_vo import Bootstrap, DataLoader
from vo_runner import VoRunner

USER_DS_QUESTION = "Which dataset would you like to use? Options are: 'parking' (1), 'kitti' (2), 'malaga' (3), 'own_1' (4), or 'own_2' (5). Enter your choice as a number: "


def get_ds_name_from_user():
    response = input(USER_DS_QUESTION)
    switcher = {
        '1': "parking",
        '2': "kitti",
        '3': "malaga",
        '4': "own_1",
        '5': "own_2",
    }
    if response in switcher:
        return switcher.get(response)
    else:
        print("Invalid, defaulting to 'kitti'")
        return "kitti"

def init(ds_name):
    dl = DataLoader(ds_name)
    b = Bootstrap(dl, outlier_tolerance=(15, None, 15))
    vo = VO(b)

    return dl, b, vo

def savePoses(poses, dl):
    # Save the pose list 
    trajectory_dir = Path.cwd().joinpath("solution_trajectories")
    if not trajectory_dir.is_dir():
        trajectory_dir.mkdir()
    save_path = trajectory_dir.joinpath(f"{dl.dataset_str}.pkl")
    with open(save_path, "wb") as f:
        pickle.dump(poses, f)


if __name__ == "__main__":
    ds_name = get_ds_name_from_user()
    data_loader, bootstrap, visual_odometry_pipeline = init(ds_name)
    vo_runner = VoRunner(data_loader, bootstrap, visual_odometry_pipeline)
    poses = vo_runner.run_and_visualize(save=False)
    savePoses(poses, data_loader)
