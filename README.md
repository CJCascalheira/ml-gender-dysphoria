# Gender Dysphoria and Machine Learning
Classifying gender dysphoria disclosures on Reddit using machine learning (ML) techniques.

## Data
* **Training Set**. The df_train_raw.csv is labeled. It will have 1,000 instances before the end of class. Colleagues are finishing their labeling now.
* **Testing Set**. The df_test_raw.csv is not yet labeled. However, we will develop an ML classifier to machine-label these data and then perform error analysis. For an example, see Saha et al. (2019). 

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