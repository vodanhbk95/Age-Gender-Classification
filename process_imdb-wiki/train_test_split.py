# Importing dependencies
import pandas as pd
import numpy as np

from sklearn.model_selection import train_test_split

# Loading the data
data = pd.read_csv('data.csv')

data_filtered = data[data['age'] != -1]
train, test = train_test_split(data_filtered, test_size = 0.3, random_state=0)

train.to_csv('train.csv', index=False)
test.to_csv('test.csv', index=False)

