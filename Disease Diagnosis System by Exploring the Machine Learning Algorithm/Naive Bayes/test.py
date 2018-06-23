import pandas as pd
import numpy as np
from sklearn.feature_extraction.text import CountVectorizer
from dataset_explorer import explore

dataset_df = pd.read_csv("dataset.csv")

print(dataset_df[0:20])
