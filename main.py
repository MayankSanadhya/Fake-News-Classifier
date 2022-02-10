# %%writefile app.py
import streamlit as st
import pickle
import re
from nltk.corpus import stopwords
import string
import nltk
from nltk.stem import PorterStemmer


cv = pickle.load(open('vectorizer.pkl','rb'))
classifier = pickle.load(open('model.pkl','rb'))

st.title("Fake News Classifier")

input_news = st.text_area("Enter the title of the news")

if st.button('Predict'):

    # 1. preprocess
    ps = PorterStemmer()
    # corpus = []


    def transform_text(x):
        #     corpus = []
#         for i in range(0, len(x)):
            review = re.sub('[^a-zA-Z]', ' ', x)
            review = review.lower()
            review = review.split()

            review = [ps.stem(word) for word in review if word not in stopwords.words('english')]
            review = ' '.join(review)
#             corpus.append(review)
            return review

    transform_text(input_news)
    transformed_news = transform_text(input_news)

    # 2. vectorize
    vector_input = cv.transform([transformed_news])

    # 3. predict
    result = classifier.predict(vector_input)
    # 4. display

    if result == 1:
        st.header('Real News')
    else:
        st.header('Fake News')



