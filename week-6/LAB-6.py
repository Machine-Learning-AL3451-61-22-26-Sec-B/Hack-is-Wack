import streamlit as st
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

def main():
    st.title('Sentiment Analysis with Classifier')

    default_file_path = r"C:\Users\TUF\Downloads\document (1).csv"

    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")

    delimiter = st.selectbox("Select delimiter for CSV file", [",", ";", "\t"])

    if uploaded_file is not None:
        msg = pd.read_csv(uploaded_file, names=['message', 'label'], delimiter=delimiter)
    else:
        msg = pd.read_csv(default_file_path, names=['message', 'label'])

    st.write("Total Instances of Dataset:", msg.shape[0])

    show_sample = st.checkbox("Show sample of original dataset")
    if show_sample:
        st.write("Sample of Original Dataset:")
        st.write(msg.head())

    msg['labelnum'] = msg.label.map({'pos': 1, 'neg': 0})

    test_size = st.slider("Test Size Ratio:", min_value=0.1, max_value=0.5, value=0.2, step=0.05)
    X = msg.message
    y = msg.labelnum
    Xtrain, Xtest, ytrain, ytest = train_test_split(X, y, test_size=test_size, random_state=42)

    handle_nan_inf = st.checkbox("Handle NaN or infinite values in target variable (ytrain)")
    if handle_nan_inf:
        if np.isnan(ytrain).any() or np.isinf(ytrain).any():
            mask = ~np.isnan(ytrain) & ~np.isinf(ytrain)
            Xtrain = Xtrain[mask]
            ytrain = ytrain[mask]

    count_v = CountVectorizer()
    Xtrain_dm = count_v.fit_transform(Xtrain)
    Xtest_dm = count_v.transform(Xtest)

    df = pd.DataFrame(Xtrain_dm.toarray(), columns=count_v.get_feature_names_out())
    st.write("Sample of Vectorized Training Data:")
    st.write(df.head())

    classifier = st.selectbox("Choose Classifier:", ["Multinomial Naive Bayes", "Support Vector Machine", "Random Forest"])

    if classifier == "Multinomial Naive Bayes":
        clf = MultinomialNB()
    elif classifier == "Support Vector Machine":
        clf = SVC()
    elif classifier == "Random Forest":
        clf = RandomForestClassifier()

    clf.fit(Xtrain_dm, ytrain)
    pred = clf.predict(Xtest_dm)

    st.write('Sample Predictions:')
    for doc, p in zip(Xtest, pred):
        p = 'pos' if p == 1 else 'neg'
        st.write(f"{doc} -> {p}")

    st.write('Accuracy Metrics:')
    metrics = st.multiselect("Choose Metrics:", ["Accuracy", "Recall", "Precision", "Confusion Matrix"])
    if "Accuracy" in metrics:
        st.write('Accuracy:', accuracy_score(ytest, pred))
    if "Recall" in metrics:
        st.write('Recall:', recall_score(ytest, pred))
    if "Precision" in metrics:
        st.write('Precision:', precision_score(ytest, pred))
    if "Confusion Matrix" in metrics:
        st.write('Confusion Matrix:\n', confusion_matrix(ytest, pred))

if __name__ == '__main__':
    main()
