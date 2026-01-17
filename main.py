import nltk
import streamlit as st
import pickle as pkl
import string
import nltk
nltk.download('punkt')
nltk.download('punkt_tab')
nltk.download('stopwords')
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
ps = PorterStemmer()
from nltk.corpus import stopwords



def transform_text(text):
    text = text.lower()
    text = nltk.word_tokenize(text)
    y = []
    for i in text:
        if i.isalnum():
            y.append(i)
    text = y[:]
    y.clear()

    for i in text:
        if i not in stopwords.words('english') and i not in string.punctuation:
            y.append(i)

    text = y[:]
    y.clear()

    for i in text:
        y.append(ps.stem(i))

    return " ".join(y)

tfidf=pkl.load(open('vectorizer.pkl','rb'))
model = pkl.load(open('model.pkl','rb'))

st.title("Email/sms spam Classifier")

input_sms=st.text_area("Enter message")

if st.button("Predict"):
    transformed_sms=transform_text(input_sms)

    vector_input=tfidf.transform([transformed_sms])

    result=model.predict(vector_input)[0]

    if result==1:
        st.header("Spam Detected")
    else:
        st.header("Not spam ")

