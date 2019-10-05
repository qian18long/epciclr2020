import sys
sys.path.append('..')
import math
import imageio
import tensorflow.contrib.layers as layers
import maddpg_local.common.tf_util as U
import argparse
import numpy as np
import tensorflow as tf
import time
import pickle
import os
import re
import sys
sys.path.append('..')
from .train_helpers import *
# from maddpg_local.common import tf_util as U
from functools import partial



if __name__ == '__main__':
    arglist = parse_args()
    train(arglist)
