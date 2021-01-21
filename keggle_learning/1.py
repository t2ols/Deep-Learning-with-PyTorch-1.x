#invite people for the Kaggle party
import torch
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from scipy.stats import norm
from sklearn.preprocessing import StandardScaler
from scipy import stats
import warnings
warnings.filterwarnings('ignore')
# %matplotlib inline

is_cuda = False

if torch.cuda.is_available():
    print('CUDA available')
    is_cuda = True

print( is_cuda )    