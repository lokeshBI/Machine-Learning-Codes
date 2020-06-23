
# import packages
import pandas as pd
import numpy as np

import spacy 

# Need to load the large model to get the vectors
from spacy.cli.download import download
download('en_core_web_lg')

nlp = spacy.load('en_core_web_lg')

# sample check for wordvector
text1 = "hello world"


doc = nlp("hello world of ML")

for i in doc:
    print(i)


with nlp.disable_pipes():
    vectors = np.array([token.vector for token in nlp(text1)])

vectors.shape

# load the spam data
spam_data = pd.read_csv("D:\\Machine-Learning-A-Z-New\\Machine Learning A-Z New\\spam detection\\datasets_483_982_spam.csv", engine= 'python')

spam_data.head()

spam = spam_data[['v1','v2']]
spam.head()

spam.shape

with nlp.disable_pipes():
    doc_vectors = np.array([nlp(token).vector for token in spam.v2])


doc_vectors

doc_vectors.shape

# train and test split

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(doc_vectors, spam.v1, test_size=0.1, random_state=1)

# model training
from sklearn.svm import LinearSVC
# Set dual=False to speed up training, and it's not needed
svc = LinearSVC(random_state=1, dual=False, max_iter=10000)
svc.fit(X_train, y_train)

# predictions
y_pred = svc.predict(X_test)

# model evaluation
from sklearn.metrics import confusion_matrix
cm = confusion_matrix(y_test,y_pred)
cm

# accuracy score 
from sklearn.metrics import accuracy_score
accuracy_score(y_test, y_pred)

# accuracy from cm
(484+59)/(484+59+15)



# cosine similarity
from sklearn.metrics.pairwise import cosine_similarity


a = "This is a wild cat from india"
b = "This is a tiger from india belongs to cat family" 

documents = [a,b]

# Scikit Learn
from sklearn.feature_extraction.text import CountVectorizer

# Create the Document Term Matrix
count_vectorizer = CountVectorizer(stop_words='english')
count_vectorizer = CountVectorizer()
sparse_matrix = count_vectorizer.fit_transform(documents)

# OPTIONAL: Convert Sparse Matrix to Pandas Dataframe if you want to see the word frequencies.
doc_term_matrix = sparse_matrix.todense()
df = pd.DataFrame(doc_term_matrix, 
                  columns=count_vectorizer.get_feature_names(), 
                  index=['doc_a', 'doc_b'])
df                  


print(cosine_similarity(df, df))
