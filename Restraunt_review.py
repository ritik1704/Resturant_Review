import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
df= pd.read_csv("Restaurant_Reviews.tsv",delimiter= '\t',quoting=3)
import re
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
from nltk.stem.porter import PorterStemmer
corpus=[]
for i in range(0,1000):
    computed_review=[]
    review = re.sub('[^a-zA-Z]',' ',df['Review'][i])
    review=review.lower()
    review=review.split()
    ps=PorterStemmer()
    for word in review:
        if  not word in set(stopwords.words('english')) :
            computed_review.append(ps.stem(word))
    computed_review=' '.join(computed_review)
    corpus.append(computed_review)

from sklearn.feature_extraction.text import CountVectorizer
cv=CountVectorizer(max_features=1500)
x= cv.fit_transform(corpus).toarray()
y=df.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train,x_test,y_train,y_test=train_test_split(x,y,test_size=0.1,random_state=0)



from sklearn.naive_bayes import GaussianNB
classifier=GaussianNB()
classifier.fit(x_train,y_train)

ypred=classifier.predict(x_test)

from sklearn.metrics import confusion_matrix
cm=confusion_matrix(y_test,ypred)

# ENTER BY THE USER

text=input('enter the comment')
corpus2 = []
review2 = re.sub("[^a-zA-z]", ' ', text)
review2 = review2.lower()
review2 = review2.split()
ps2 = PorterStemmer()
computed_review=[]
for word in review2:
        if  not word in set(stopwords.words('english')) :
            computed_review.append(ps.stem(word))
computed_review=' '.join(computed_review)
corpus2.append(computed_review)
review2 = " ".join(review2)
corpus2.append(review2)
from sklearn.feature_extraction.text import CountVectorizer
cv2 = CountVectorizer(max_features = 1500)
x2 = cv2.fit_transform(corpus + corpus2).toarray()
my = x2[-1].reshape(1, -1)
result = classifier.predict(my)
if result == 1:
    ans = "Positive review given by the person"
else:
    ans = "Negative review given by the person"
    
print(ans)
