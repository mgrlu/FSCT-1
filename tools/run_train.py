import os
import sys
modulepath = r"C:\Users\gilcher\code\STEMDETECTION\code\FSCT\scripts"
sys.path.append(modulepath)
os.chdir(r"C:\Users\gilcher\code\STEMDETECTION\code\FSCT")
from tools import load_file, save_file, get_fsct_path
from model import Net
from train_datasets import TrainingDataset, ValidationDataset
from fsct_exceptions import NoDataFound
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.data import DataLoader
import glob
import random
import threading
import os
import shutil
from train import *
import plotly.express as px
if __name__ == '__main__':
    datafolder = r"C:\Users\gilcher\code\STEMDETECTION\code\FSCT\data\train\sample_dir\\"
    #load all npy files and print object types and shapes as well as a tabular count for each loaded array of column 3
    files = glob.glob(os.path.join(datafolder, "*.npy"))
    print("Found {} files".format(len(files)))
    for f in files[0:4]:
        data = np.load(f)
        print(f"Loaded {type(data)} with shape {data.shape}")
        print(f"Column 3 has {len(np.unique(data[:,3]))} unique values: {np.unique(data[:,3])}")
        # break

    parameters = dict(
        preprocess_train_datasets=0,
        preprocess_validation_datasets=0,
        clean_sample_directories=0,  # Deletes all samples in the sample directories.
        perform_validation_during_training=0,
        generate_point_cloud_vis=1,  # Useful for visually checking how well the model is learning. Saves a set of samples called "latest_prediction.las" in the "FSCT/data/"" directory. Samples have label and prediction values.
        load_existing_model=1,
        num_epochs=200,
        learning_rate=0.0000025,
        # learning_rate=0.5,
        # learning_rate=0.0000001,
        input_point_cloud=None,
        model_filename="model.pth",
        sample_box_size_m=np.array([6, 6, 6]),
        sample_box_overlap=[0, 0, 0],
        min_points_per_box=200,
        max_points_per_box=1000000,
        subsample=1,
        subsampling_min_spacing=0.025,
        num_cpu_cores_preprocessing=0,  # 0 Means use all available cores.
        num_cpu_cores_deep_learning=1,  # Setting this higher can cause CUDA issues on Windows.
        train_batch_size=2,
        validation_batch_size=10,
        device="cuda",  # set to "cuda" or "cpu"
    )
    run_training = TrainModel(parameters)
    run_training.run_training()
