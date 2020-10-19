import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from fbprophet import Prophet


#####
# Create time series estimate of Australia covid cases to
# compare to my model
#####

## Read in NNDSS data

