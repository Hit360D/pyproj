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
        print('**' * 40)
        for count, i in enumerate(rows, 1):
            print(f"{count}. {i} -- {df.loc[i, 'title']}       [ {df.loc[i, 'type']} ] [ {df.loc[i, 'animeSeason'].upper()} ]")
        print('**' * 40)

        # Take input, print which title selected and return that title
        num = input('Select the title: ')
        try:
            num = int(num)
            print(f"You selected: {num}. {df.loc[rows[num-1], 'title']}")
            return rows[num-1]
        except ValueError:
            print('Invalid input, enter an integer.')
    else:
        print('No titles matched!')
        return None

    
# print(get_title('komi-san', all))

# Function to get recommendations, takes input title and prints top 10 most similar titles based on input tags
def get_recommendations(title, cosine_sim=cosine_sim):
    index = get_title(title, all)
    if index == None:
        return
    
    print('**' * 40)

    # Get the pairwsie similarity scores of all animes with that anime
    sim_scores = list(enumerate(cosine_sim[index]))

    # Sort the animes based on the similarity scores
    sim_scores = sorted(sim_scores, key=lambda x: x[1], reverse=True)

    # Get the scores of the 10 most similar animes
    sim_scores = sim_scores[1:11]

    # Get the anime indices
    anime_index = []
    for i in sim_scores:
        anime_index.append(i[0])

    for count, i in enumerate(anime_index, 1):
        print(f"{count}. {all.loc[i, 'title']} [{all.loc[i, 'type']}]  [ {all.loc[i, 'status']} ]  [ {all.loc[i, 'animeSeason']} ]")

user_input = input('Search for a title: ')
get_recommendations(user_input)
# get_recommendations('komi-san')



print('End of script')