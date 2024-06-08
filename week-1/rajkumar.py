import numpy as np
import pandas as pd

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
data = {
    'Feature1': [1, 0, 1, 0],
    'Feature2': [1, 0, 0, 0],
    'Feature3': [0, 1, 0, 0],
    'Feature4': [0, 0, 0, 1],
    'Label': [1, 0, 1, 0]
}
df = pd.DataFrame(data)

X = df[['Feature1', 'Feature2', 'Feature3', 'Feature4']].values
y = df['Label'].values

ce = CandidateElimination(num_features=X.shape[1])
ce.fit(X, y)

print("Final Specific Hypotheses:", ce.get_hypotheses()[0])
print("Final General Hypotheses:", ce.get_hypotheses()[1])
