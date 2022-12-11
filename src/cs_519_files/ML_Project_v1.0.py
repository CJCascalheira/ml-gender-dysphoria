import numpy as np
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
import matplotlib.pyplot as plt
import contractions                                 # Expanding contractions
from nltk.stem.wordnet import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.tree import DecisionTreeClassifier
from sklearn.svm import SVC
from sklearn.linear_model import SGDClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score
from sklearn.metrics import classification_report

print(' ------------------------------------------')
print('| Classifying Gender Dysphoria Disclosures |')
print('|  on Social Media with Machine Learning.  |')
print(' ------------------------------------------')
print()
print('Team members:    Cory J. Cascalheira')
print('                 Ivan Nieto Gomez   ')
print('                 Edgar Corrales Sotelo')
print()
print('Data Processing....')
print()

#num_of_lines = 2

dataset = pd.read_csv('df_truth.csv')
dataset.tail()
#print('Dataset size: ',dataset.shape)
# ------ ORIGINAL DATA --------
#print('Original Dataset: \n',dataset)
headers = list(dataset.columns.values)
#print(headers)

text = dataset.iloc[:,1]            # text = dataset['text']
#print(text.shape)
#print(text)

# ---------------- EXPANDING CONTRACTIONS -------------------
n_text = []
expanded_words = []
for i in range(len(text)):
    a = str(text[i])
    # -------------- LOWERCASE ----------
    a_lower = a.lower()
    line = a_lower.split()
    for h in line:
        expanded_words.append(contractions.fix(h))
    expanded_text = ' '.join(expanded_words)
    n_text.append(expanded_text)
    expanded_words.clear()                  # Clearing List
#print(n_text)
#print('Original text: ' + text)
#print('Expanded_text: ' + n_text)

mySeries = pd.Series(n_text)
#print(mySeries)                                                                                                                            
# ----------------------------------------------------------

new_text = []
w_stopwords_text = []
for k in range(len(mySeries)):
    a = str(mySeries[k])
    # ----------------- REMOVING NUMBERS --------
    text_ = ''.join([i for i in a if not i.isdigit()])
    # -------- REMOVING SPECIAL CHARACTERS AND PUNCTUATION --------
    punc = '''!()-[]{};:'"\,“”<>’./?@#$%^&*ðÿ˜=∆+_~'''
    for j in text_:
        if j in punc:
            text_ = text_.replace(j,'')
    #print(text_)
    new_text.append(text_)
#print(new_text)

# -------------------- REMOVING STOP WORDS -------------------
for j in range(len(new_text)):
    text_tokens = word_tokenize(new_text[j])
    tokens_without_sw = [word for word in text_tokens if not word in stopwords.words('english')]
    filtered_sentence = (" ").join(tokens_without_sw)
    w_stopwords_text.append(filtered_sentence)
#print(w_stopwords_text)

col_text = pd.DataFrame(w_stopwords_text)
final_text = col_text[0]
#print(final_text)

# -------------------------------- NORMALIZING WORDS VIA LEMMATIZATION ---------------------------------
f_sent = []
xxx = []
yyy = []
for count in range(len(w_stopwords_text)):
    b = str(w_stopwords_text[count]) 
    words_sent = b.split()
    for j in words_sent:
        lemmatizer = WordNetLemmatizer()
        lem_sent = lemmatizer.lemmatize(j)  
        f_sent.append(lem_sent)
    xxx = ' '.join(f_sent)
    yyy.append(xxx)
    f_sent.clear()
#print(yyy)

col_text = pd.DataFrame(yyy)
final_text = col_text[0]

# --------------- CLEANED DATA PLACED IN COLUMN #2 -----------
dataset.insert(2,'new_text',final_text) 

#print('Clean Dataset: \n',dataset['new_text'].values)
print('1. Text Preprocessing Done!')

X = dataset['new_text'].values
y = dataset['dysphoria'].values
y_labels = np.unique(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.30, random_state=1)
#print(X_train.shape)
#print(X_test.shape)

vectorizer = TfidfVectorizer()
X_train = vectorizer.fit_transform(X_train)
X_test = vectorizer.transform(X_test)

# ---------------------------------------------------------------------------------
print('2. Classifiers')
print()

# ---------------------------------------------------------------------------------
print('2.1. Support Vector Machine (SVM - RBF)')
print()

svm = SVC(kernel = 'rbf', gamma = 0.1, C = 10.0, random_state = 1)
svm.fit(X_train,y_train)
y_pred = svm.predict(X_test)
svm_predictions = svm.predict(X_test)

print('     Misclassified samples (linear model): %d'%(y_test!=y_pred).sum())
print('     Accuracy: %.3f'%accuracy_score(y_test,y_pred))
print(classification_report(y_test, svm_predictions))
# ---------------------------------------------------------------------------------
print('2.2. Decision Tree')
print()

dt = DecisionTreeClassifier(criterion="entropy", random_state = 1)
dt.fit(X_train,y_train)
y_pred = dt.predict(X_test)
dt_predictions = dt.predict(X_test)

print('     Misclassified samples: %d'%(y_test!=y_pred).sum())
print('     Accuracy: %.2f'%accuracy_score(y_test,y_pred))
print(classification_report(y_test, dt_predictions))
print()

# ---------------------------------------------------------------------------------
print('2.3. Logistic Regression')
print()
log_reg = LogisticRegression(penalty='l2', C = 10, random_state = 1)
log_reg.fit(X_train, y_train)
y_pred = log_reg.predict(X_test)
log_reg_predictions = log_reg.predict(X_test)

print('     Misclassified samples: %d'%(y_test!=y_pred).sum())
print('     Accuracy: %.2f'%accuracy_score(y_test,y_pred))
print(classification_report(y_test, log_reg_predictions))
print()

# ---------------------------------------------------------------------------------
#print('2.4. Linear Regression')
#print()
#lr = LogisticRegression()
#lr.fit(X_train, y_train)
#y_pred = lr.predict(X_test)
#lr_predictions = lr.predict(X_test)

#print('     Misclassified samples: %d'%(y_test!=y_pred).sum())
#print('     Accuracy: %.2f'%accuracy_score(y_test,y_pred))
#print(classification_report(y_test, lr_predictions))
#print()
# ---------------------------------------------------------------------------------




