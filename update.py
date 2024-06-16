import pandas as pd
import numpy as np
# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# To produce a matrix where each element (i, j) shows similarity between document i and document j based on their TF-IDF vectors.
from sklearn.metrics.pairwise import linear_kernel

print('Libraries loaded successfully')

# Convert JSON to CSV wihtout including dataframe index
df = pd.read_json('anime-offline-database.json')
df.to_csv('anime-database.csv', index=False)

# Import anime titles
titles = pd.read_csv('test-db.csv')
# Remove [ & ' & ] from tags column
titles['tags'] = titles['tags'].str.replace(r'[','').str.replace(r']','').str.replace(r"'","")
# Drop sources picture thumnail relatedAnime
titles.drop(['sources', 'picture', 'thumbnail', 'relatedAnime'], axis=1, inplace=True)
print('Imported all titles, dropped unnecessary columns')


# New DataFrame for TF-IDF generation
all = titles
# Remove commas from tags column
all['tags'] = all['tags'].str.replace(r',','')
all['animeSeason'] = all['animeSeason'].str.replace(r'{','').str.replace("'","").str.replace(r"}","")
print('New dataframe created')

# Compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for every 'tags' column
# Define a TF-IDF Vectorizer Object
tfidf = TfidfVectorizer()

print('TF-IDF vectorizer created')

# Construct required TF-IDF matrix by fitting and transforming the data
tfidf_matrix = tfidf.fit_transform(all['tags'])

print('TF-IDF matrix created')

# Compute the cosine similarity matrix
cosine_sim = linear_kernel(tfidf_matrix, tfidf_matrix)
print('Cosine similarity calculated')

# Write cosine_sim to a csv file
cosine_sim_df = pd.DataFrame(cosine_sim)
cosine_sim_df.to_csv('cosine-sim.csv')
print('Written Cosine similarity matrix to csv file')