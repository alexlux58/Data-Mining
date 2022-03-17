import numpy as np

class NaiveBayes():
    def __init__(self, X, y):
        self.num_examples, self.num_features = X.shape
        self.num_classes = len(np.unique(y))
        self.eps = 1e-6
    
    def fit(self, X, y):
        self.classes_mean = {}
        self.classes_variance = {}
        self.classes_prior = {}
        
        for c in range(self.num_classes):
            X_c = X[y==c]
            self.classes_mean[str(c)] = np.mean(X_c, axis=0)
            self.classes_variance[str(c)] = np.var(X_c, axis=0)
            self.classes_prior[str(c)] = X_c.shape[0] / self.num_examples
    
    def predict(self, X):
        probs = np.zeros((self.num_examples, self.num_classes))
        for c in range(self.num_classes):
            prior = self.classes_prior[str(c)]
            probs_c = self.density_function(X, self.classes_mean[str(c)], self.classes_variance[str(c)])
            probs[:, c] = probs_c + np.log(prior)
        return np.argmax(probs, 1)
    
    def density_function(self, x, mean, sigma):
        # calculate probability from gaussian density function
        constant = -self.num_features / 2 * np.log(2 * np.pi) - 0.5 * np.sum(np.log(sigma + self.eps))
        probabilities = 0.5*np.sum(np.power(x - mean, 2) / (sigma + self.eps), 1)
        return constant - probabilities
    
if __name__ == '__main__':
    # X = np.loadtxt('data.txt', delimiter=',')
    # y = np.loadtxt('targets.txt') - 1
    
    X = np.load('X.npy')
    y = np.load('y.npy')
    
    print(X.shape)
    print(y.shape)
    
    NB = NaiveBayes(X, y)
    NB.fit(X, y)
    y_pred = NB.predict(X)
    
    print(sum(y_pred == y) / X.shape[0])
    
    print("Spam % = ", sum(y) / y.shape[0])