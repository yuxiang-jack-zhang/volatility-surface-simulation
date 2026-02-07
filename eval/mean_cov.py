from sklearn.covariance import LedoitWolf
import numpy as np

def olse_mean_estimator(Y, mu0=None, use_pinv=False):
    """
    Optimal Linear Shrinkage Estimator (OLSE) for the mean vector.

    Parameters:
    - Y: (n x p) data matrix, each row is an observation.
    - mu0: (p x 1) shrinkage target mean vector.
    - use_pinv: whether to force use pseudo-inverse (useful for p > n)

    Returns:
    - mu_hat: (p,) estimated mean vector
    """
    n, p = Y.shape
    y_bar = np.mean(Y, axis=0)  # Sample mean: shape (p,1)
    
    # Sample covariance (bias=True gives MLE version)
    S = np.cov(Y.T, bias=True)
    
    # Choose inverse method
    if use_pinv or p > n:
        S_inv = np.linalg.pinv(S)
        inv_type = "pseudo-inverse"
    else:
        S_inv = np.linalg.inv(S)
        inv_type = "inverse"

    if mu0 is None:
        mu0 = np.ones(p)

    # Compute inner products
    y_bar_T_Sinv = y_bar.T @ S_inv
    mu0_T_Sinv = mu0.T @ S_inv

    ySy = float(y_bar_T_Sinv @ y_bar)
    mSm = float(mu0_T_Sinv @ mu0)
    ySm = float(y_bar_T_Sinv @ mu0)

    # Correction term: unbiased estimator of trace(S^{-1}) is (p / (n - p)) when c < 1
    # In large sample limit, this is the "shrinkage factor" penalty
    if p < n:
        correction = p / (n - p)
    else:
        correction = 0  # In p > n case, we omit this or can regularize further

    numerator_alpha = ySy - correction
    denominator = ySy * mSm - ySm**2
    alpha = (numerator_alpha * mSm - ySm**2) / denominator
    beta = (1 - alpha) * ySm / mSm

    mu_hat = alpha * y_bar + beta * mu0

    return mu_hat.flatten(), inv_type

def jorion_bayes_stein_mu(Y):
    """
    Implements Jorion (1986) Bayes–Stein estimator as per equations (4) and (5).

    Parameters:
    Y: ndarray of shape (M, N)
        Asset returns data, N assets over M time periods.

    Returns:
    mu_bs: ndarray of shape (N,)
        Shrinkage estimator of mean return vector.
    """

    M, N = Y.shape
    mu_hat = np.mean(Y, axis=0, keepdims=True).T         # Sample mean vector (N x 1)
    Sigma_hat = np.cov(Y.T, bias=True)                   # Sample covariance matrix (N x N)

    # Compute minimum variance portfolio weights
    ones = np.ones((N, 1))
    if N > M:
        Sigma_inv = np.linalg.pinv(Sigma_hat)
    else:
        Sigma_inv = np.linalg.inv(Sigma_hat)

    w_min = Sigma_inv @ ones / (ones.T @ Sigma_inv @ ones)  # (N x 1)
    # Target mean = expected return of minimum variance portfolio
    mu_min = w_min.T @ mu_hat                  # scalar
    mu_min_vec = mu_min * np.ones((N, 1))              # (N x 1)

    # Compute shrinkage intensity
    delta = mu_hat - mu_min_vec
    shrinkage_denom = delta.T @ Sigma_inv @ delta
    phi_hat = (N + 2) / ((N + 2) + M * shrinkage_denom)
    phi_hat = max(0, min(1, phi_hat))  # ensure [0, 1]

    # Final Bayes–Stein estimator
    mu_bs = (1 - phi_hat) * mu_hat + phi_hat * mu_min_vec

    return mu_bs.flatten(), phi_hat[0][0]

def calculate_mean_cov(data, BS=False, OLSE=False, LW=False, clip=None):
    """
    Calculate mean and covariance matrix for input data.
    
    Args:
        data (numpy.ndarray): Input data array
        shr (bool): Whether to use LedoitWolf shrinkage, default True
        clip (float or None): If not None, winsorize data at this quantile level (e.g., 0.01 for 1% quantile)
        
    Returns:
        tuple: (mean_vector, cov_matrix)
            - mean_vector: Mean vector of the data
            - cov_matrix: Covariance matrix of the data
    """
    # Winsorize data if requested
    if clip is not None:
        lower_quantile = np.quantile(data, clip, axis=0)
        upper_quantile = np.quantile(data, 1-clip, axis=0)
        data = np.clip(data, lower_quantile, upper_quantile)
    
    # Calculate mean vector
    if BS:
        # Use Bayes-Stein shrinkage
        mean_vector = jorion_bayes_stein_mu(data)[0]
    elif OLSE:
        # Use Optimal Linear Shrinkage Estimator
        mean_vector = olse_mean_estimator(data)[0]
    else:
        # Use sample mean
        mean_vector = np.mean(data, axis=0)
    
    # Calculate covariance matrix
    if LW:
        # Use LedoitWolf shrinkage for better estimation
        cov_matrix = LedoitWolf().fit(data).covariance_
    else:
        # Use sample covariance matrix
        cov_matrix = np.cov(data.T)
    
    return mean_vector, cov_matrix
