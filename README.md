# Gender Dysphoria and Machine Learning
Classifying gender dysphoria disclosures on Reddit using machine learning (ML) techniques.

## Data
* **Training Set**. The df_train_raw.csv is labeled. It will have 1,000 instances before the end of class. Colleagues are finishing their labeling now.
* **Testing Set**. The df_test_raw.csv is not yet labeled. However, we will develop an ML classifier to machine-label these data and then perform error analysis. For an example, see Saha et al. (2019). 

## Features
* **Clinical Keywords**. Taken from the top five non-common (e.g., "individuals", "female") keywords featured in the DSM-5.
* **Mental Health Distress**
* **Word Embeddings**
* **Psycholinguistic Attributes**. Taken from the Linguistic Inquiry and Word Count (LIWC) lexicon.
* **Sentiment**
* **Top n-grams**