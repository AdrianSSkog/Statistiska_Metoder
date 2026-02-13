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

class OneHotEncoder():
    def __init__(self, drop_first=True):
        self.drop_first = drop_first
        self.categories = {}
        self.fitted = False
        self.feature_names_ = []
    
    def fit(self, data, column_names=None):
        X = np.asarray(data)
        self.feature_names_ = []

        if column_names is None:
            column_names = [f"{j}" for j in range(X.shape[1])]

        for j in range(X.shape[1]):
            col = X[:,j]
            if isinstance(col[0], (str, np.str_)):
                cats = np.unique(col)
                start = 1 if self.drop_first else 0
                self.categories[j] = np.unique(col)

                for k in range(start, len(cats)):
                    self.feature_names_.append(f"{column_names[j]}_{cats[k]}")
            else:
                self.feature_names_.append(column_names[j])
        self.fitted = True
        return self
   
    def transform(self, data):
        if self.fitted == False:
            raise ValueError("Encoder has not been fitted")
        
        X = np.asarray(data)
        n = X.shape[0]
        output_columns = []

        for j in range(X.shape[1]):
            col = X[:,j]
            if j in self.categories:
                cats = self.categories[j]
                one_hot = np.zeros((n, len(cats) - self.drop_first))

                for i in range(n):
                    if col[i] not in cats:
                        raise ValueError(f"Category {col[i]} is not recognised")
                    index = np.where(cats == col[i])[0][0]
                    if index >= self.drop_first:
                        one_hot[i, index - self.drop_first] = 1
                output_columns.append(one_hot)
            else:
                output_columns.append(col.reshape(-1,1))
        return np.hstack(output_columns).astype(float)

    def fit_transform(self, data, column_names=None):
        return self.fit(data, column_names=column_names).transform(data)

class LinearRegression():
    def __init__(self, confidence_level= 0.95):
        self.confidence_level = confidence_level
        self.b = None 
        self.intercept = None
        self.d = None 
        self.n = None 
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
        self.DoF = None
        self.fullData = None

    def fit(self, X, y):
        X = np.asarray(X)
        y = np.asarray(y)

        if X.ndim == 1:
            X = X.reshape(-1, 1)

        self.fullData = np.column_stack((X, y))

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

        self.DoF = self.n - self.d - 1

        if self.DoF <= 0:
            raise ValueError("Not enough samples to estimate variance")
        
        self.sigma2 = self.SSE / self.DoF
        self.adjRsqr = 1 - ((1-self.Rsqr)*((self.n - 1) /self.DoF))
        self.std = np.sqrt(self.sigma2)

        cov_beta = self.sigma2 * (X_Pinv @ X_Pinv.T)
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
            return np.sqrt(self.SSE / self.DoF)
        else:
            Y_pred = self.predict(X)
            return np.sqrt(np.mean((Y - Y_pred)**2))

    def correlation(self, data=None):
        if data is None:
            X = self.fullData
        
        else:
            if isinstance(data, np.ndarray):
                X = data
            else:
                X = data.to_numpy(dtype=float)
        
        n, d = X.shape
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
    
    def F_test(self):
        if self.SSE is None or self.SSR is None:
            raise ValueError("Model has not been fitted")

        MSR = self.SSR / self.d
        MSE = self.SSE / self.DoF

        F = MSR / MSE
        pValue = 1 - st.f.cdf(F, self.d, self.DoF)


        return F, pValue
    
    def t_test(self):
        t_values = self.b / self.se[1:]
        p_values = 2 * (1 - st.t.cdf(np.abs(t_values), self.DoF)) 
        alpha = 1 - self.confidence_level
        significant = p_values < alpha
        return t_values, p_values, significant

    
    def confidence_intervals(self, confidence_level = None):
        if confidence_level is None:
            CL = self.confidence_level
        else:
            CL = confidence_level

        if CL < 0 or CL > 1:
            raise ValueError("Confidence level must be a float between 0 and 1")
        
        alpha = 1 - CL
        critical_t = st.t.ppf(1 - alpha/2, df=self.DoF)
        CI_lower = self.b - critical_t*self.se[1:]
        CI_upper = self.b + critical_t*self.se[1:]

        return CI_lower, CI_upper


    def summary(self, feature_names=None, target_name="target variable", testX=None, testy=None):
        if self.b is None:
            raise ValueError("Model has not been fitted")
        
        if feature_names is None:
            feature_names = [f"feature {i+1}" for i in range(self.d)]
        
        print("Linear regression summary")
        print("="*26)
        print(f"Target variable:            {target_name}")
        print(f"Sample size:                {self.n}")
        print(f"Number of features:         {self.d}")
        print()
        print(f"variance:                   {self.sigma2:.4f}")
        print(f"Standard deviation:         {self.std:.4f}")
        print(f"R square:                   {self.Rsqr:.4f}")
        print(f"Adjusted R square:          {self.adjRsqr:.4f}")
        if testX is None or testy is None:
            print(f"train RMSE:                 {self.rmse():.4f}")
        else: 
            print(f"test RMSE:                  {self.rmse(testX, testy):.4f}")
        f_stat, fstat_p = self.F_test()
        print(f"F-statistic:                {f_stat:.4f}")
        print(f"p value (F-statistic):      {fstat_p:.4f}")
        print()
        print(f"Confidence level:   {self.confidence_level:.1%}")
        
        CI_L, CI_U = self.confidence_intervals()
        t, p, sig = self.t_test()
        print("Feature                 CI lower   coefficient   CI upper  t-value   p-value   significant")
        print("="*90)
        for i, feat in enumerate(feature_names):
            sign = "Yes" if sig[i]==True else "No"
            print(f"{feat:<26}{CI_L[i]:>10.2f}{self.b[i]:>10.2f}{CI_U[i]:>10.2f}{t[i]:>10.2f}{p[i]:>10.2f}{sign:>10}")

        print()
        print("Pearson correlation:")
        print()
        corr = self.correlation()
        columnNames = feature_names + [target_name]
        for i, feat in enumerate(columnNames):
            print(f"{feat:<26} {[f"{x:>5.2f}"for x in corr[i,:]]}")


    


    

