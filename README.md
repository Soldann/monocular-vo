# monocular-vo project VAMR 2024

# Demonstration

A video demonstration of the pipeline in action can be found below ([click here if the embed doesn't work](https://www.youtube.com/watch?v=hEA7zJPG3Gs))

[![YouTube](http://i.ytimg.com/vi/hEA7zJPG3Gs/hqdefault.jpg)](https://www.youtube.com/watch?v=hEA7zJPG3Gs)

## Specifications of the computer from the demo-video
- Ryzen 97950x16/32 cores/threads at 5.2GHz
- The threads of the python threading tool may span over multiple cores, so the exact number of CPU threads is not known, but overall CPU usage was about 30%
- 64GB of 6000MHz dual channel memory, only about 1Gb of RAM usage, however

# Dataset setup

Create a folder called `datasets`, download and unzip the datasets found here into that folder:

Description | Link (size)
------------- | ----------
Parking garage dataset (easy) |	[parking.zip](https://rpg.ifi.uzh.ch/docs/teaching/2024/parking.zip) (208.3 MB)
KITTI 05 dataset (hard)	| [kitti05.zip](https://rpg.ifi.uzh.ch/docs/teaching/2024/kitti05.zip) (1.4 GB)
Malaga 07 dataset (hard) | [malaga-urban-dataset-extract-07.zip](https://rpg.ifi.uzh.ch/docs/teaching/2024/malaga-urban-dataset-extract-07.zip) (2.4 GB)
Own Dataset | https://share.easywin.ch/s/tXC9KEZ0 (1.2 GB)

This should get you folders `datasets/kitti`, `datasets/malaga-urban-dataset-extract-07`, and `datasets/parking`
for use in the pipeline. The additional dataset recorded for this project is in a folder `own_dataset` with subfolders `own_dataset/ds1`, `own_dataset/ds2`, and `own_dataset/calibration`. 

# Running the script

1. ## Setup environment
    A conda environment is provided for installing the dependencies required.
    ```
    conda env create --name vamr --file=vamr-env.yml
    conda activate vamr
    ```

2. ## Run VO pipeline
    Run main.py.
    ```
    python3 main.py
    ```
    You are prompted to insert a number determining what dataset should be evaluated.
    Before VO begins, the raw images are loaded into memory.

3. ## Run Evaluation 
    Run evalution.py.
    ```
    python3 evaluation.py
    ```
    You will be again prompted to insert a number determining what dataset should be evaluated.
    Note that you must have run the pipeline first using `main.py` or this will not work.
