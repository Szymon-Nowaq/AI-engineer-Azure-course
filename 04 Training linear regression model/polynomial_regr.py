import pandas as pd
from matplotlib import pyplot as plt
import statsmodels.formula.api as smf
import numpy as np

doggy_df = pd.read_csv('doggy.csv', delimiter = '\t', header='infer')