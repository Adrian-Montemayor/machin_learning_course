import numpy as np 
import matplotlib.pyplot as plt 
import pandas as pd 

##Importing the dataset
dataset = pd.read_csv('Data.csv')
#### Features or independent variable
#### Feature are the columns with which
#### you're going to preditc the dependent variable
x = dataset.iloc[:, :-1].values
y = dataset.iloc[:, -1].values
print(x)
print(y)

###Taking care of missing data
from sklearn.impute import SimpleImputer
"""
If “mean”, then replace missing values using the mean along each column. Can only be used with numeric data.

If “median”, then replace missing values using the median along each column. Can only be used with numeric data.

If “most_frequent”, then replace missing using the most frequent value along each column. Can be used with strings or numeric data.

If “constant”, then replace missing values with fill_value. Can be used with strings or numeric data.
"""
imputer = SimpleImputer(missing_values=np.nan, strategy="mean")
