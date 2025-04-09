import os
import random
import numpy as np
import torch as ch
from glob import glob
from copy import deepcopy
from collections import Counter
from IPython.display import display

import sys
sys.path.append("../src/")
import utils
import matplotlib.pyplot as plt
from robustness_analyzer import RobustnessAnalyzer


from pytorch3d.io import load_obj

data_dir = "../data"

# Envmap(s) we're using during optimization (one or more)
envmap_paths = glob(os.path.join("../data", "environments/*"))[::10]
#envmap_paths = ["../data/environments/goegap_road_2k.exr"]  # This would generate background-neutral (white) renders

true_class = 'tank, army tank, armored combat vehicle, armoured combat vehicle'
target_class = true_class

object_dir = "tank"

params_to_optimize = ["camera"]

kwargs = {
    "obj_path": os.path.join(data_dir, object_dir, "tank.obj"),
    "texture_path": None,
    "envmap_paths": envmap_paths,
    "target_class": target_class,
    "batch_size": 2,  # How many different viewpoints we're optimizing in parallel (* #Environments)
    "params_to_optimize": params_to_optimize,
    "targeted": target_class!=true_class,
    "positive_z": True # constraints to camera z>0
}

robust_analyzer = RobustnessAnalyzer(**kwargs)

#### Run optimization: This will generate num_runs*batch_size viewpoints
results_camera = robust_analyzer.run(num_runs=1, num_iterations=5, lr=5e-3)

