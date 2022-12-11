"""
Get Word2vec embeddings of Reddit text

RESOURCES:
- https://radimrehurek.com/gensim/auto_examples/core/run_core_concepts.html#sphx-glr-auto-examples-core-run-core-concepts-py
- https://radimrehurek.com/gensim/auto_examples/tutorials/run_word2vec.html#sphx-glr-auto-examples-tutorials-run-word2vec-py
- https://www.geeksforgeeks.org/removing-stop-words-nltk-python/
- https://stackoverflow.com/questions/15547409/how-to-get-rid-of-punctuation-using-nltk-tokenizer
"""

# Load dependencies
import os
import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import RegexpTokenizer
import gensim.downloader as api

# Set file path
my_path = os.getcwd()

# Import data
df_truth = pd.read_csv(my_path + '/data/cleaned/features_temp/df_truth_dass.csv')
df_primary = pd.read_csv(my_path + '/data/cleaned/features_temp/df_primary_dass.csv')

#region GROUND TRUTH DATASET

#region PREPROCESS TEXT

# Create empty list
corpus_truth = []

# Set the stop words from NLTK
stop_words = set(stopwords.words('english'))

# Create a custom tokenizer to remove punctuation
tokenizer = RegexpTokenizer(r'\w+')

# Create corpus
for string in df_truth['text'].tolist():

    # Remove strange characters
    string = string.replace('\r', '')
    string = string.replace('*', '')

    # Get tokens (i.e., individual words)
    tokens = tokenizer.tokenize(string)

    # Set a list holder
    filtered_sentence = []

    # For each token, remove the stop words
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # Save list of tokens (i.e., sentences) to preprocessed corpus
    corpus_truth.append(filtered_sentence)

#endregion

#region WORD2VEC MODEL

# Load the Word2vec model
wv = api.load('word2vec-google-news-300')

# List embeddings for each post
post_embeddings = []

# For every word in every sentence within the corpus
for sentence in corpus_truth:

    # List of word embeddings
    w2v_embeddings = []

    # Get the word embeddings for each word
    for word in sentence:

        # See if there is a pretrained word embedding
        try:
            vector_representation = wv[word]
            w2v_embeddings.append(vector_representation)

        # If there is no pretrained word embedding
        except KeyError:
            vector_representation = np.repeat(0, 300)
            w2v_embeddings.append(vector_representation)

    # Save the word embeddings at the post level
    post_embeddings.append(w2v_embeddings)

# Set a holder variable
avg_post_embeddings = []

# Aggregate word embeddings
for post in post_embeddings:

    # Transform embedding into data frame where each row is a word and each column is the embedding dimension
    df = pd.DataFrame(post)

    # Square each element in the data frame to remove negatives
    df = df.apply(np.square)

    # Get the mean of each embedding dimension
    df = df.apply(np.mean, axis=0)

    # The average word embedding for the entire Reddit post
    avg_embedding = df.tolist()

    # Append to list
    avg_post_embeddings.append(avg_embedding)

# Create a dataframe with the average word embeddings of each post
embedding_df = pd.DataFrame(avg_post_embeddings)

# Rename the columns
embedding_df = embedding_df.add_prefix('w2v_')

#endregion

# Add average word embeddings to the ground truth data set
df_truth1 = pd.concat([df_truth, embedding_df], axis=1)

# Save to file
df_truth1.to_csv(my_path + '/data/cleaned/with_features/df_truth.csv')

#endregion

#region PRIMARY DATASET

#region PREPROCESS TEXT

# Create empty list
corpus_primary = []

# Set the stop words from NLTK
stop_words = set(stopwords.words('english'))

# Create a custom tokenizer to remove punctuation
tokenizer = RegexpTokenizer(r'\w+')

# Create corpus
for string in df_primary['text'].tolist():

    # Remove strange characters
    string = string.replace('\r', '')
    string = string.replace('*', '')

    # Get tokens (i.e., individual words)
    tokens = tokenizer.tokenize(string)

    # Set a list holder
    filtered_sentence = []

    # For each token, remove the stop words
    for w in tokens:
        if w not in stop_words:
            filtered_sentence.append(w)

    # Save list of tokens (i.e., sentences) to preprocessed corpus
    corpus_primary.append(filtered_sentence)

#endregion

#region WORD2VEC MODEL

# Load the Word2vec model
wv = api.load('word2vec-google-news-300')

# List embeddings for each post
post_embeddings = []

# For every word in every sentence within the corpus
for sentence in corpus_primary:

    # List of word embeddings
    w2v_embeddings = []

    # Get the word embeddings for each word
    for word in sentence:

        # See if there is a pretrained word embedding
        try:
            vector_representation = wv[word]
            w2v_embeddings.append(vector_representation)

        # If there is no pretrained word embedding
        except KeyError:
            vector_representation = np.repeat(0, 300)
            w2v_embeddings.append(vector_representation)

    # Save the word embeddings at the post level
    post_embeddings.append(w2v_embeddings)

# Set a holder variable
avg_post_embeddings = []

# Aggregate word embeddings
for post in post_embeddings:

    # Transform embedding into data frame where each row is a word and each column is the embedding dimension
    df = pd.DataFrame(post)

    # Square each element in the data frame to remove negatives
    df = df.apply(np.square)

    # Get the mean of each embedding dimension
    df = df.apply(np.mean, axis=0)

    # The average word embedding for the entire Reddit post
    avg_embedding = df.tolist()

    # Append to list
    avg_post_embeddings.append(avg_embedding)

# Create a dataframe with the average word embeddings of each post
embedding_df = pd.DataFrame(avg_post_embeddings)

# Rename the columns
embedding_df = embedding_df.add_prefix('w2v_')

#endregion

# Add average word embeddings to the primary data set
df_primary1 = pd.concat([df_primary, embedding_df], axis=1)

# Save to file
df_primary1.to_csv(my_path + '/data/cleaned/with_features/df_primary.csv')

#endregion
