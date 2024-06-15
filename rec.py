# guide - https://www.kaggle.com/code/dgoenrique/a-simple-movie-tv-show-recommendation-system

import pandas as pd
import numpy as np

# I have no idea what these libraries are
import ast
import re

import matplotlib.pyplot as plt
import missingno as msno

# for TF-IDF vector. Advanced mafs shit man, I should look this up again and again.
from sklearn.feature_extraction.text import TfidfVectorizer
# to produce a matrix where each eleemnt (i, j) shows similarity between document i and document j based on their TF-IDF vectors.
from sklearn.metrics.pairwise import linear_kernel

print("All libraries loaded succeesfully!")

# ------------- IGNORE ALL THIS CLUTTER ------------
# converting json files to csv
# read json file into a pandas DataFrame
#df = pd.read_json('anime-offline-database.json')

# convert DataFrame to csv without including the index of DataFrame
#df.to_csv('anime-database.csv', index=False)

# --------------------------------------------------

# import anime tiles
titles = pd.read_csv('anime-database.csv')

# The anime listing is well sorted and does not have any duplicates, still leaving this here for future reference
# look for duplicates and output the first 4 rows
# titles[titles.duplicated() == True].head(5)

# remove duplicates, but do the operation on the original DataFrame that is tiles
# titles.drop_duplicates(inplace=True)

# print(titles.head())

# Show information about the DataFrame
# print(titles.info())

# Do a sum of all not assigned columns
# print(titles.isna().sum())

# print(titles['tags'].str.replace(r'[','').str.replace(r"'",""))

# isolate the first tag from the columnt 'tags' and add it to the DataFrame 
titles['tags'] = titles['tags'].str.replace(r'[','').str.replace(r"'","").str.replace(r"]","")
titles['tag'] = titles['tags'].str.split(",").str[0]

# print(titles['tag'].unique())
# print(titles['type'].unique())

# Drop sources picture thumbnail relatedAnime  note -- axis=0 refers to rows and axis=1 refers to columns in pandas
titles.drop(['sources', 'picture', 'thumbnail', 'relatedAnime'], axis=1, inplace=True)

# Separate into ALL SPECIAL MOVIE OVA TV ONA UNKNOWN
# after which drop the index column from each
all = titles.head()

special = titles[titles['type'] == 'SPECIAL'].copy().reset_index()
special.drop(['index'], axis=1, inplace=True)

movie = titles[titles['type'] == 'MOVIE'].copy().reset_index()
movie.drop(['index'], axis=1, inplace=True)

ova = titles[titles['type'] == 'OVA'].copy().reset_index()
ova.drop(['index'], axis=1, inplace=True)

tv = titles[titles['type'] == 'TV'].copy().reset_index()
tv.drop(['index'], axis=1, inplace=True)

ona = titles[titles['type'] == 'ONA'].copy().reset_index()
ona.drop(['index'], axis=1, inplace=True)

unknown = titles[titles['type'] == 'UNKNOWN'].copy().reset_index()
unknown.drop(['index'], axis=1, inplace=True)

# Compute Term Frequency-Inverse Document Frequency (TF-IDF) vectors for every 'tags' column
# Define a TD-IDF Vectorizer Object
tfidf = TfidfVectorizer()

# Construct required TF-IDF matrix by fitting and transforming the data
tfidf_matrix_all = tfidf.fit_transform(all['tags'].str.replace(r",",""))
tfidf_matrix_special = tfidf.fit_transform(special['tags'].str.replace(r",",""))
tfidf_matrix_movie = tfidf.fit_transform(movie['tags'].str.replace(r",",""))
tfidf_matrix_ova = tfidf.fit_transform(ova['tags'].str.replace(r",",""))
tfidf_matrix_tv = tfidf.fit_transform(tv['tags'].str.replace(r",",""))
tfidf_matrix_ona = tfidf.fit_transform(ona['tags'].str.replace(r",",""))
tfidf_matrix_unknown = tfidf.fit_transform(unknown['tags'].str.replace(r",",""))

# Oh man, mafs again
# Compute the cosine similarity matrix
cosine_sim_all = linear_kernel(tfidf_matrix_all, tfidf_matrix_all)
cosine_sim_special = linear_kernel(tfidf_matrix_special, tfidf_matrix_special)
cosine_sim_movie = linear_kernel(tfidf_matrix_movie, tfidf_matrix_movie)
cosine_sim_ova = linear_kernel(tfidf_matrix_ova, tfidf_matrix_ova)
cosine_sim_tv = linear_kernel(tfidf_matrix_tv, tfidf_matrix_tv)
cosine_sim_ona = linear_kernel(tfidf_matrix_ona, tfidf_matrix_ona)
cosine_sim_unknown = linear_kernel(tfidf_matrix_unknown, tfidf_matrix_unknown)

# Identify the index of of show in the data given its title
# Create a new pandas Series with <database>.index as the values and <database>['title'] as the index.
# In other words, each title will be mapped to its corresponding index in the <database> DataFrame.
indices_all = pd.Series(all.index, index=all['title'])
indices_special = pd.Series(special.index, index=special['title'])
indices_movie = pd.Series(movie.index, index=movie['title'])
indices_ova = pd.Series(ova.index, index=ova['title'])
indices_tv = pd.Series(tv.index, index=tv['title'])
indices_ona = pd.Series(ona.index, index=ona['title'])
indices_unknown = pd.Series(unknown.index, index=unknown['title'])

# Function that gets the index searcher and searches the title index
def get_title(title, indices):
    try:
        index = indices[title]
    except:
        print('\n Title not found')
        return None
    
    # Checking if index is of instance (type) of NumPy's int64 type else ask to select a title
    if isinstance(index, np.int64):
        return index
    else:
        rt = 0
        print('Select a title: ')
        if indices == indices_all:
            print(f"{i} - {all['title'].iloc[index[i]]}", end=" ")
            # imported ast and re and extracting Year somehow, chat GPT's method:
            df = all
            df['year'] = df['animeSeason'].apply(lambda x: ast.literal_eval(x)['year'])
            print(f"({df['year'].iloc[index[i]]})")
        elif indices == indices_special:
            print(f"{i} - {special['title'].iloc[index[i]]}", end=" ")
            # imported ast and re and extracting Year somehow, chat GPT's method:
            df = special
            df['year'] = df['animeSeason'].apply(lambda x: ast.literal_eval(x)['year'])
            print(f"({df['year'].iloc[index[i]]})")
        elif indices == indices_movie:
            print(f"{i} - {movie['title'].iloc[index[i]]}", end=" ")
            # imported ast and re and extracting Year somehow, chat GPT's method:
            df = movie
            df['year'] = df['animeSeason'].apply(lambda x: ast.literal_eval(x)['year'])
            print(f"({df['year'].iloc[index[i]]})")
        elif indices == indices_ova:
            print(f"{i} - {ova['title'].iloc[index[i]]}", end=" ")
            # imported ast and re and extracting Year somehow, chat GPT's method:
            df = ova
            df['year'] = df['animeSeason'].apply(lambda x: ast.literal_eval(x)['year'])
            print(f"({df['year'].iloc[index[i]]})")
        elif indices == indices_tv:
            print(f"{i} - {tv['title'].iloc[index[i]]}", end=" ")
            # imported ast and re and extracting Year somehow, chat GPT's method:
            df = tv
            df['year'] = df['animeSeason'].apply(lambda x: ast.literal_eval(x)['year'])
            print(f"({df['year'].iloc[index[i]]})")
        elif indices == indices_ona:
            print(f"{i} - {ona['title'].iloc[index[i]]}", end=" ")
            # imported ast and re and extracting Year somehow, chat GPT's method:
            df = ona
            df['year'] = df['animeSeason'].apply(lambda x: ast.literal_eval(x)['year'])
            print(f"({df['year'].iloc[index[i]]})")
        else:
            print(f"{i} - {unknown['title'].iloc[index[i]]}", end=" ")
            # imported ast and re and extracting Year somehow, chat GPT's method:
            df = unknown
            df['year'] = df['animeSeason'].apply(lambda x: ast.literal_eval(x)['year'])
            print(f"({df['year'].iloc[index[i]]})")
        rt = int(input())
        return index[rt]

print('EOP')
