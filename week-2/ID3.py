import streamlit as st
import pandas as pd
import numpy as np

def generate_synthetic_data():
    # Generate synthetic data
    np.random.seed(42)
    n_samples = 100
    n_features = 4
    X = np.random.randn(n_samples, n_features)
    y = np.random.randint(0, 3, n_samples)  # Three classes
    df = pd.DataFrame(X, columns=[f"Feature_{i+1}" for i in range(n_features)])
    df['target'] = y
    return df

def main():
    st.title("Dummy Classifier")
    st.write("This app demonstrates a dummy classifier using synthetic data.")

    # Generate synthetic data
    df = generate_synthetic_data()

    # Display dataset
    st.subheader("Synthetic Dataset")
    st.write(df)

    # Train model
    most_common_class = df['target'].mode().iloc[0]

    # Evaluate model
    y_true = df['target']
    y_pred = np.full_like(y_true, most_common_class)
    accuracy = np.mean(y_true == y_pred)
    report = f"Accuracy: {accuracy:.2f}"

    # Display evaluation results
    st.subheader("Model Evaluation")
    st.write(report)

if __name__ == "__main__":
    main()
