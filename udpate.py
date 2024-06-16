import pandas as pd
import numpy as np
# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# To produce a matrix where each element (i, j) shows similarity between document i and document j based on their TF-IDF vectors.
from sklearn.metrics.pairwise import linear_kernel

print('Libraries loaded successfully')