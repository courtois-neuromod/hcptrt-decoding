import numpy as np
import pandas as pd
import pickle
import glob
import os
from load_confounds import Params9, Params24
from nilearn.input_data import NiftiLabelsMasker
from sklearn import preprocessing
from numpy import savetxt
from termcolor import colored

sys.path.append(os.path.join("../"))
import utils

"""
Utilities for extracting desired volumes 
and labeling/relabeling data.
The outputs are final post-processed data.
"""