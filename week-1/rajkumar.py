import numpy as np

class CandidateElimination:
    def __init__(self, num_features):
        self.num_features = num_features
        self.S = [np.zeros(num_features, dtype=int)]  # Specific Boundary
        self.G = [np.full(num_features, None)]  # General Boundary

    def fit(self, X, y):
        for i in range(len(X)):
            x = X[i]
            label = y[i]

            if label == 1:  # Positive example
                self._remove_inconsistent_hypotheses(x)
            else:  # Negative example
                self._generalize_hypotheses(x)

    def _remove_inconsistent_hypotheses(self, x):
        S_prime = []
        for s in self.S:
            if np.array_equal(s, x):
                continue
            if np.any(s != x):
                S_prime.append(s)
        self.S = S_prime

        self._specialize_hypotheses(x)

    def _specialize_hypotheses(self, x):
        G_prime = []
        for g in self.G:
            for i in range(len(x)):
                if g[i] is None or g[i] == x[i]:
                    continue
                g_copy = np.copy(g)
                g_copy[i] = None
                if not self._is_more_general(g_copy):
                    G_prime.append(g_copy)
        self.G = G_prime

    def _generalize_hypotheses(self, x):
        S_prime = []
        for s in self.S:
            if np.array_equal(s, np.zeros(self.num_features)):
                continue
            s_copy = np.copy(s)
            for i in range(len(x)):
                if s[i] is None or s[i] == x[i]:
                    continue
                s_copy[i] = None
            if not self._is_more_general(s_copy):
                S_prime.append(s_copy)
        self.S = S_prime

        self._remove_inconsistent_hypotheses(x)

    def _is_more_general(self, hypothesis):
        for s in self.S:
            if all((s_val is None or s_val == h_val) for s_val, h_val in zip(s, hypothesis)):
                return True
        return False

    def get_hypotheses(self):
        return self.S, self.G

# Example usage:
X = np.array([
    [1, 1, 0, 0],
    [0, 0, 1, 0],
    [1, 0, 0, 0],
    [0, 0, 0, 1]
])
y = np.array([1, 0, 1, 0])

ce = CandidateElimination(num_features=X.shape[1])
ce.fit(X, y)
print("Final Specific Hypotheses:", ce.get_hypotheses()[0])
print("Final General Hypotheses:", ce.get_hypotheses()[1])



# Streamlit app
st.title('Candidate-Elimination Algorithm')

# Upload CSV file
uploaded_file = st.file_uploader("Upload CSV file", type=['csv'])

if uploaded_file is not None:
    # Load data from uploaded file
    data = load_data(uploaded_file)

    # Separating concept features from Target
    concepts = np.array(data.iloc[:, 0:-1])
    target = np.array(data.iloc[:, -1])

    # Execute the algorithm
    s_final, g_final = learn(concepts, target)

    # Display the training data
    st.subheader('Training Data')
    st.write(data)

    # Display final specific hypothesis
    st.subheader('Final Specific Hypothesis')
    st.write(s_final)

    # Convert final general hypotheses to DataFrame for tabular display
    g_final_df = pd.DataFrame(g_final, columns=data.columns[:-1])
    
    # Display final general hypotheses
    st.subheader('Final General Hypotheses')
    st.write(g_final_df)
