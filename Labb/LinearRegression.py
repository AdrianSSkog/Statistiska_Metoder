import numpy as np
import scipy.stats as st

def train_test_split(data, train=80, test=20): 
    if train + test != 100:
        raise ValueError("Training data and test data must add up to 100%")
    
    data = np.asarray(data).copy()
    n = data.shape[0]
    split_index = int(n * (train/100))

    np.random.shuffle(data)

    trainingData = data[:split_index,:]
    testData = data[split_index:,:]

    return trainingData, testData

def make_numeric(data):
    for col in data.columns:
        if data[col].dtype == object:
            try:
                data[col] = data[col].astype(float)
            except:
                pass
    return data

   

class LinearRegression():
    def __init__(self, confidence_level= 0.95):
        self.confidence_level = confidence_level
        self.b = None #coefficient
        self.intercept = None
        self.d = None #number of features
        self.n = None #number of samples
        self.SSE = None 
        self.SSR = None
        self.Syy = None
        self.sigma2 = None
        self.residuals = None
        self.se = None
        self.SST = None
        self.Rsqr = None
        self.adjRsqr = None
        self.std = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.n, self.d = X.shape
        X_matrix = np.column_stack((np.ones(X.shape[0]), X))
        X_Pinv = np.linalg.pinv(X_matrix)
        beta = X_Pinv @ y
        self.intercept = beta[0]
        self.b = beta[1:]
        

        y_hat = X_matrix @ beta
        self.residuals = y - y_hat

        self.SSE = np.sum((self.residuals)**2) 
        self.SSR = np.sum((y_hat - np.mean(y))**2)
        self.Syy = np.sum((y - np.mean(y))**2)
        self.SST = self.SSE + self.SSR
        self.Rsqr = self.SSR / self.SST

        DF = self.n - self.d - 1

        if DF <= 0:
            raise ValueError("Not enough samples to estimate variance")
        
        self.sigma2 = self.SSE / DF
        self.adjRsqr = 1 - ((1-self.Rsqr)*((self.n - 1) /DF))
        self.std = np.sqrt(self.sigma2)

        cov_beta = self.sigma2 * (np.linalg.inv(X_Pinv @ X_Pinv.T))
        self.se = np.sqrt(np.diag(cov_beta))

        return self.b, self.intercept
    
    def predict(self, X):
        if self.b is None:
            raise ValueError("Model has not been fitted")
        
        X = np.asarray(X)
        if X.ndim == 1:
            X = X.reshape(-1, 1)   

        Y = X @ self.b + self.intercept
        return Y
    
    def rmse(self, X=None, Y=None):
        if X is None or Y is None:
            return np.sqrt(self.SSE / self.n)
        else:
            Y_pred = self.predict(X)
            return np.sqrt(np.mean((Y - Y_pred)**2))

    def correlation(self, data=None):
        n, d = data.shape
        X = data.to_numpy(dtype=float)
        corr = []

        means = np.mean(X, axis=0)
        variances = np.var(X, axis=0, ddof=1)

        for i in range(d):
            corr_row = []
            for j in range(d):
                cov = np.sum((X[:,i] - means[i])*(X[:,j] - means[j]))/(n-1)
                r = cov/np.sqrt(variances[i] * variances[j]) 
                corr_row.append(round(r, 4))
            corr.append(corr_row)
        corr = np.asarray(corr)
        return corr


    

