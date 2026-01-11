import cvxopt
import numpy as np

class SupportVectorMachine:
    '''
    Binary Classifier using Support Vector Machine
    '''
    def __init__(self):
        self.alpha = None
        self.w = None
        self.b = None
        self.X_train = None
        self.y_train = None
        self.gamma = None
        
    def gaussian_kernel(self,X1, X2, gamma):
        '''
        Compute the Gaussian kernel matrix between X1 and X2 efficiently.
        
        Args:
            X1: np.array of shape (N1, D)
            X2: np.array of shape (N2, D)
            gamma: float, gamma parameter for the Gaussian kernel
            
        Returns:
            Kernel matrix of shape (N1, N2)
        '''
        self.gamma = gamma
        # Compute squared Euclidean distances efficiently
        X1_norm = np.sum(X1**2, axis=1).reshape(-1, 1)
        X2_norm = np.sum(X2**2, axis=1).reshape(1, -1)
        distances = X1_norm + X2_norm - 2 * np.dot(X1, X2.T)
        
        # Compute the Gaussian kernel
        K = np.exp(-gamma * distances)
        return K
        
    def fit(self, X, y, kernel = 'linear', C = 1.0, gamma = 0.001):
        '''
        Learn the parameters from the given training data
        Classes are 0 or 1
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
            y: np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the ith sample
                
            kernel: str
                The kernel to be used. Can be 'linear' or 'gaussian'
                
            C: float
                The regularization parameter
                
            gamma: float
                The gamma parameter for gaussian kernel, ignored for linear kernel
        '''
        N, D = X.shape
        y = y.astype(np.double).reshape(-1, 1) 
        
        # Compute Gram matrix (for linear kernel)
        if kernel == 'linear':
            K = X @ X.T
        elif kernel == 'gaussian':
            K = self.gaussian_kernel(X, X, gamma)
        
        P = cvxopt.matrix(np.outer(y, y) * K)
        q = cvxopt.matrix(-np.ones((N, 1)))
        G = cvxopt.matrix(np.vstack((-np.eye(N), np.eye(N))))
        h = cvxopt.matrix(np.hstack((np.zeros(N), np.ones(N) * C)))
        A = cvxopt.matrix(y.reshape(1, -1))
        b = cvxopt.matrix(0.0)

        # Solve QP problem
        solution = cvxopt.solvers.qp(P, q, G, h, A, b)
        alpha = np.ravel(solution['x'])
        
        # Select support vectors
        sv = alpha > 1e-5
        self.alpha = alpha[sv]
        self.X_train = X[sv]
        self.y_train = y[sv]

        # Compute w and b for linear kernel
        if kernel == 'linear':
            self.w = np.sum(self.alpha[:, None] * self.y_train * self.X_train, axis=0)
            self.b = np.mean(self.y_train - (self.w @ self.X_train.T))
        elif kernel == 'gaussian':
            # Compute bias term (b) for Gaussian kernel
            sv_indices = np.where(sv)[0]
            self.b = 0
            for i in range(len(self.alpha)):
                self.b += self.y_train[i] - np.sum(self.alpha * self.y_train * K[sv_indices[i], sv])
            self.b /= len(self.alpha)
            
        num_support_vectors = np.sum(sv)
        support_vector_percentage = (num_support_vectors / len(y)) * 100
        print(f"Number of Support Vectors: {num_support_vectors}")
        print(f"Percentage of Support Vectors: {support_vector_percentage:.2f}%")
        
        
    def predict(self, X):
        '''
        Predict the class of the input data
        
        Args:
            X: np.array of shape (N, D) 
                where N is the number of samples and D is the flattened dimension of each image
                
        Returns:
            np.array of shape (N,)
                where N is the number of samples and y[i] is the class of the
                ith sample (0 or 1)
        '''
        
        if self.w is not None:  # Linear Kernel
            print(self.w)
            print(self.b)
            return (X @ self.w + self.b >= 0).astype(int)
        # Gaussian Kernel (optimized computation)
        gamma = self.gamma  # Ensure gamma is set correctly
        X_norm = np.sum(X**2, axis=1).reshape(-1, 1)  # (N_test, 1)
        sv_norm = np.sum(self.X_train**2, axis=1).reshape(1, -1)  # (1, N_train)
        print(self.w)
        print(self.b)
        # Compute Gaussian Kernel correctly (N_test, N_train)
        K = np.exp(-gamma * (X_norm + sv_norm - 2 * np.dot(X, self.X_train.T))) - self.b  # (N_test, N_train)
        
        # Ensure correct broadcasting
        decision_values = np.sum((self.alpha * self.y_train.T) * K, axis=1)  # (N_test,)
        
        return (decision_values >= 0).astype(int)