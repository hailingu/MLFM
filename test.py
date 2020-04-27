import os

# working directory
BASEDIR = os.getcwd()
print(BASEDIR)

import pandas as pd
import numpy as np


dataframe = pd.read_csv(BASEDIR + '/assets/datasets/ml-latest-small/ratings.csv')

userId_dict = {}
movieId_dict = {}

userId_unique = dataframe.userId.unique()
movieId_unique = dataframe.movieId.unique()


idx = 0
for n in range(userId_unique.shape[0]):
    userId_dict[userId_unique[idx]] = idx
    idx += 1

idx = 0
for n in range(movieId_unique.shape[0]):
    movieId_dict[movieId_unique[idx]] = idx
    idx += 1

ratings = np.zeros(shape=(len(userId_dict), len(movieId_dict)))


for row in dataframe.itertuples():
    ratings[userId_dict[row.userId], movieId_dict[row.movieId]] = row.rating

print(ratings[0, 0:10])