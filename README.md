# Gender Dysphoria and Machine Learning
Classifying gender dysphoria disclosures on Reddit using machine learning (ML) techniques. Describing the language of gender dysphoria disclosures with natural language processing (NLP).

## Data
* **Ground Truth Set**. The df_truth.csv is labeled. It has ~600 labels from human and the rest are from different subreddits. If from r/GenderDysphoria, then labeled 1; if from r/askscience, labeled 0.

* **Primary Set**. The df_primary.csv is not yet labeled. However, we will develop an ML classifier to machine-label these data and then perform error analysis. For an example, see Saha et al. (2019). The primary dataset is comprised of posts from different transgender-specific subreddits.

## Features
* **Clinical Keywords**. Taken from the top five non-common (e.g., "individuals", "female") keywords featured in the DSM-5.
  * *n_features* += 1.

* **Mental Health Distress**. Generated from a classifier trained on data from subreddits, similar to the DASS approach in Saha et al. (2019). Labels from this classifier used as features.
  * *n_features* += 4.
  * *Types of features*. Depression, anxiety, stress, suicide.

* **Word Embeddings**. Used Word2Vec and the Gensim library; add the 300 dimensions as features.
  * *n_features* += 300.

* **Psycholinguistic Attributes**. Taken from the Linguistic Inquiry and Word Count (LIWC) lexicon.
  * *n_features* += 93.

* **Sentiment**. Valence of the sentiment of the Reddit post.
  * *n_features* += 1.

* **Top n-grams**. Found the top n-grams (n = 1, 2, 3).
  * *n_features* += 550.
  * After cleaning, found top 250 unigrams, 250 bigrams, and 50 trigrams (i.e., only 50 because trigrams used less frequently and inspection revealed non-informative trigrams) based on ordering ngrams by tf_idf scores.
  