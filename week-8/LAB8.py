from sklearn.datasets import load_iris
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split

# Load the Iris dataset
dataset = load_iris()

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(dataset.data, dataset.target, random_state=0)

# Initialize the K-Nearest Neighbors classifier with k=1
kn = KNeighborsClassifier(n_neighbors=1)

# Train the classifier
kn.fit(X_train, y_train)

# Make predictions and print results
for i, x in enumerate(X_test):
    x_new = x.reshape(1, -1)  # Reshape input for prediction
    prediction = kn.predict(x_new)
    print(f"TARGET={y_test[i]} {dataset.target_names[y_test[i]]} PREDICTED={prediction} {dataset.target_names[prediction]}")

# Calculate and print the accuracy score of the classifier on the test set
accuracy = kn.score(X_test, y_test)
print("Accuracy:", accuracy)
