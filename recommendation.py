import pandas as pd
import numpy as np
# TF-IDF Vectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
# To produce a matrix where each element (i, j) shows similarity between document i and document j based on their TF-IDF vectors.
from sklearn.metrics.pairwise import linear_kernel

print('Libraries loaded')



# Import all titles
titles = pd.read_csv('anime-database.csv')
# Remove [ & ' & ] from tags column
titles['tags'] = titles['tags'].str.replace(r'[','').str.replace(r']','').str.replace(r"'","")
# Drop sources picture thumnail relatedAnime
titles.drop(['sources', 'picture', 'thumbnail', 'relatedAnime'], axis=1, inplace=True)

print('Imported all titles, dropped columns')


# New DataFrame for TF-IDF generation
all = titles
# Remove commas from tags column
all['tags'] = all['tags'].str.replace(r',','')
all['animeSeason'] = all['animeSeason'].str.replace(r'{','').str.replace("'","").str.replace(r"}","")

print('New daataframe created')

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

# Function to get the index of the searched title
def get_title(title, df):
    # Find which rows have that given title and print them
    rows = []
    for i, row in df.iterrows():
        if title.lower() in row['title'].lower():
            rows.append(i)

    if rows:
        print('The following titles matched:')
        for i in rows:
            print(f"Index: {i} -- {df.loc[i, 'title']} [ {df.loc[i, 'animeSeason'].upper()} ]")
    else:
        print('No titles matched!')
        return None
    
get_title('Pokemon', all)




print('End of script')