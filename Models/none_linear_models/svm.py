import numpy as np
from Models.model import Model
from Models.utils.svm_kernels import linear_kernel

class SVM(Model):
    """
    Support Vector Machine implementation using Sequential Minimal Optimization (SMO)
    """
    def __init__(self, kernel_func=None, C=1.0, tol=1e-3, max_passes=5):
        """
        Initialize SVM with custom kernel
        :param kernel_func: Kernel function that takes two vectors and returns their similarity
        :param C: Regularization parameter
        :param tol: Numerical tolerance
        :param max_passes: Maximum number of optimization passes
        """
        super().__init__()
        self.kernel_func = kernel_func if kernel_func is not None else linear_kernel
        self.C = C
        self.tol = tol
        self.max_passes = max_passes
        self.alphas = None
        self.b = 0
        self.support_vectors = None
        self.support_vector_labels = None
        self.support_vector_indices = None
    def compute_kernel_matrix(self, X1, X2=None):
        """
        Compute kernel matrix for given data points using the provided kernel function
        """
        if X2 is None:
            X2 = X1
        
        n1, n2 = X1.shape[0], X2.shape[0]
        K = np.zeros((n1, n2))
        for i in range(n1):
            for j in range(n2):
                K[i,j] = self.kernel_func(X1[i], X2[j])
        return K
    def compute_gradients(self, X, y):
        """
        Not directly used in SMO algorithm, but implemented for consistency
        """
        pass
    def fit(self, X, y, optimizer=None, num_epochs=None):
        """
        Fit SVM using Sequential Minimal Optimization (SMO)
        """
        n_samples = X.shape[0]
        self.alphas = np.zeros(n_samples)
        self.b = 0
        
        # Compute kernel matrix
        K = self.compute_kernel_matrix(X)
        
        # SMO optimization
        passes = 0
        while passes < self.max_passes:
            num_changed_alphas = 0
            
            for i in range(n_samples):
                # Calculate Ei
                Ei = np.sum(self.alphas * y * K[i]) + self.b - y[i]
                
                if ((y[i] * Ei < -self.tol and self.alphas[i] < self.C) or 
                    (y[i] * Ei > self.tol and self.alphas[i] > 0)):
                    
                    # Select j randomly
                    j = i
                    while j == i:
                        j = np.random.randint(n_samples)
                    
                    # Calculate Ej
                    Ej = np.sum(self.alphas * y * K[j]) + self.b - y[j]
                    
                    # Save old alphas
                    alpha_i_old = self.alphas[i]
                    alpha_j_old = self.alphas[j]
                    
                    # Compute bounds L and H
                    if y[i] != y[j]:
                        L = max(0, self.alphas[j] - self.alphas[i])
                        H = min(self.C, self.C + self.alphas[j] - self.alphas[i])
                    else:
                        L = max(0, self.alphas[i] + self.alphas[j] - self.C)
                        H = min(self.C, self.alphas[i] + self.alphas[j])
                    
                    if L == H:
                        continue
                    
                    # Compute eta
                    eta = 2 * K[i,j] - K[i,i] - K[j,j]
                    if eta >= 0:
                        continue
                    
                    # Update alpha j
                    self.alphas[j] = alpha_j_old - y[j] * (Ei - Ej) / eta
                    self.alphas[j] = min(H, max(L, self.alphas[j]))
                    
                    if abs(self.alphas[j] - alpha_j_old) < 1e-5:
                        continue
                    
                    # Update alpha i
                    self.alphas[i] = alpha_i_old + y[i] * y[j] * (alpha_j_old - self.alphas[j])
                    
                    # Update b
                    b1 = self.b - Ei - y[i] * (self.alphas[i] - alpha_i_old) * K[i,i] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * K[i,j]
                    b2 = self.b - Ej - y[i] * (self.alphas[i] - alpha_i_old) * K[i,j] \
                         - y[j] * (self.alphas[j] - alpha_j_old) * K[j,j]
                    
                    if 0 < self.alphas[i] < self.C:
                        self.b = b1
                    elif 0 < self.alphas[j] < self.C:
                        self.b = b2
                    else:
                        self.b = (b1 + b2) / 2
                    
                    num_changed_alphas += 1
            
            if num_changed_alphas == 0:
                passes += 1
            else:
                passes = 0
        
        # Store support vectors
        self.support_vector_indices = np.where(self.alphas > 1e-5)[0]
        self.support_vectors = X[self.support_vector_indices]
        self.support_vector_labels = y[self.support_vector_indices]
        
        return self
    def predict(self, X):
        """
        Make predictions for input data X
        """
        if self.alphas is None:
            raise ValueError("Model not fitted yet!")
            
        K = self.compute_kernel_matrix(X, self.support_vectors)
        y_pred = np.zeros(X.shape[0])
        
        for i in range(X.shape[0]):
            y_pred[i] = np.sum(self.alphas[self.support_vector_indices] * 
                              self.support_vector_labels * K[i]) + self.b
            
        return np.sign(y_pred)