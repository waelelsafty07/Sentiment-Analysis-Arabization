import pandas as pd
import numpy as np
import json
import re
import string
from sklearn.model_selection import train_test_split
from sklearn.utils import resample
from sklearn.feature_extraction.text import TfidfVectorizer, CountVectorizer
from sklearn.metrics import accuracy_score, f1_score
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import VotingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import StratifiedShuffleSplit
from tqdm.notebook import tqdm
import matplotlib.pyplot as plt
import sys

import pickle


# matplotlib inline

rand_seed = 0  # random state for reproducibility

np.random.seed(rand_seed)


def random_split(data, features, output, fraction, seed=0):
    X_train, X_test, y_train, y_test = train_test_split(data[features],
                                                        data[output],
                                                        stratify=data[output],
                                                        random_state=seed,
                                                        train_size=fraction
                                                        )
    train_data = pd.DataFrame(data=X_train, columns=features)
    train_data[output] = y_train
    test_data = pd.DataFrame(data=X_test, columns=features)
    test_data[output] = y_test

    return train_data, test_data


def modeling(data):
    train_fraction = .80 # use this to split data into training (80%), and tmp (20%)
    val_fraction = .50   # use this to split the tmp data into validation (50%), and 
                         # testing (50%) which means that the validation will be 10% of the original data as well as the

    output = 'Sentiment' # output label column
    features = data.columns.tolist() # the features columns
    features.remove(output)
    print('output:', output)
    print('features:', features)

    train_data, tmp = random_split(data, features, output, train_fraction, rand_seed)
    val_data, test_data = random_split(tmp, features, output, val_fraction, rand_seed)

    print(len(train_data))
    print(len(val_data))
    print(len(test_data))
    print(len(train_data)+len(val_data)+len(test_data))
    print(len(data))

    return train_data, val_data, test_data
    