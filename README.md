# Regularized Non Negative Matrix Factorization
Optimization in RegularizedΝon-negativeΜatrixFactorization. 
- A feature engineering algorithm reducing the number of features while retaining the basis information necessary to reconstruct the original data.
- A matrix V is factorized into (usually) two matrices W and H (i call it C), with the property that all three matrices have no negative elements (PCA does not do that).Also the factorization explains better the dimensions of sparse data than the Principal Component Analysis.
- The optimization developed using the gradient descent steps.
    
    - C [t+1] = C [t] − n_t ∇ C [t] l(W, C [t] )
    
    - W [t+1] = W [t] − n_t ∇ W [t] l(W [t], C)
    
    Where we keep one parameter variable and the other fixed in each equation. Iteratively we change the variable parameter based on the Gradient Descent Step. 
    The input is an X table of data and the output the optimized lower rank W and C in order to minimize the least square error: min l(W, C) = ∥X − WC∥^2  (
    the projection).
    - The method #yields the values:    W, C, epsilon value (value compared with the error for the early stop), the iteration it stopped.
