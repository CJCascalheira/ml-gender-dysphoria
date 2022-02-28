# Gender Dysphoria and Machine Learning
Classifying gender dysphoria disclosures on Reddit using machine learning (ML) techniques.

## Data
* **Training Set**. The df_truth.csv is labeled. It has ~600 labels from human and the rest are from different subreddits. If from r/GenderDysphoria, then labeled 1; if from r/askscience, labeled 2.

* **Testing Set**. The df_primary.csv is not yet labeled. However, we will develop an ML classifier to machine-label these data and then perform error analysis. For an example, see Saha et al. (2019). The testing dataset is comprised of posts from different transgender-specific subreddits.

## Features
* **Clinical Keywords**. Taken from the top five non-common (e.g., "individuals", "female") keywords featured in the DSM-5.
  * *n_features* += 1.

* **Mental Health Distress**

* **Word Embeddings**

* **Psycholinguistic Attributes**. Taken from the Linguistic Inquiry and Word Count (LIWC) lexicon.
  * *n_features* += 93.

* **Sentiment**. Valence of the sentiment of the Reddit post.
  * *n_features* += 1.

* **Top n-grams**. Found the top n-grams (n = 1, 2, 3).
  * *n_features* += 3.
  * After cleaning, found top 250 unigrams, 250 bigrams, and 100 trigrams (i.e., only 100 because trigrams used less than once after 100) based on ordering ngrams by tf_idf scores.