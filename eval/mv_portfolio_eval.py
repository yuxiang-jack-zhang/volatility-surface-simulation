import torch
import pandas as pd
import numpy as np
import numba
import scipy
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from sklearn.covariance import LedoitWolf
from mosek.fusion import *
import mosek.fusion.pythonic
import sys
import seaborn as sns
import os

sns.set_style('white')

def set_seed(seed):
        torch.manual_seed(seed)
        torch.cuda.manual_seed(seed)
        np.random.seed(seed)

def comparision_histplot(stock_i, training_data_path, generated_data_path, bins_num=50, x_bound=3, y_bound=0.04, zoomin_bound=0.5):
    """
    Plot histogram comparison between generated and training data for a given stock.
    
    Args:
        stock_i (int): Index of the stock to plot
        training_data_path (str): Path to training data file
        generated_data_path (str): Path to generated data file
        bins_num (int): Number of bins for histogram
        x_bound (float): X-axis limit
        y_bound (float): Y-axis limit
        zoomin_bound (float): Zoom-in area boundary
    """
    # Load data
    training_return_data = np.load(training_data_path)
    generated_return_data = np.load(generated_data_path)

    # Print statistics
    print("Generated Samples:", generated_return_data[:, stock_i].min(), generated_return_data[:, stock_i].max(),
          np.round(generated_return_data[:, stock_i].mean(), 3), np.round(generated_return_data[:, stock_i].var(), 3))
    print("Training Samples:", training_return_data[:, stock_i].min(), training_return_data[:, stock_i].max(),
          np.round(training_return_data[:, stock_i].mean().item(), 3), np.round(training_return_data[:, stock_i].var().item(), 3))

    # Create figure
    fig, axes = plt.subplots(1, 2, figsize=(8, 3), dpi=400)
    
    # Plot generated data
    sns.histplot(ax=axes[0], data=generated_return_data[:, stock_i], bins=bins_num, alpha=1,
                stat="proportion", color="C0", label="Generated")
    bin_edges = np.histogram(generated_return_data[:, stock_i], bins=bins_num, density=False)[1]
    
    # Plot training data
    sns.histplot(ax=axes[1], data=training_return_data[:, stock_i], bins=bin_edges, alpha=1,
                stat="proportion", color="C2", label="Training")

    # Configure axes
    for ax in axes:
        ax.set_xlim(-x_bound, x_bound)
        ax.set_xticks(range(-x_bound, x_bound+1, 1))
        ax.tick_params(axis='x', labelsize=12)
        ax.set_ylim(0, y_bound)
        ax.set_yticks(np.linspace(0, y_bound, 4))
        ax.tick_params(axis='y', labelsize=12)
        ax.set_ylabel("Frequency", fontsize=12)
        ax.legend(fontsize=12, loc='upper right')
        
        # Set border style
        for spine in ax.spines.values():
            spine.set_color("black")
            spine.set_linestyle("-")
            spine.set_linewidth(1)

    # Add zoom-in effect if requested
    if zoomin_bound > 0:
        def custom_formatter(x, pos):
            return f"{x:.2f}"

        for ax in axes:
            axins = ax.inset_axes([0.1, 0.4, 0.28, 0.5])
            sns.histplot(ax=axins, data=generated_return_data[:, stock_i] if ax == axes[0] else training_return_data[:, stock_i],
                        bins=bins_num if ax == axes[0] else bin_edges, alpha=1,
                        color="C0" if ax == axes[0] else "C2", stat="proportion")
            
            axins.set_xlim(-zoomin_bound, zoomin_bound)
            axins.set_ylim(0, y_bound)
            axins.set_yticks([])
            axins.set_ylabel("")
            axins.yaxis.set_major_formatter(plt.FuncFormatter(custom_formatter))
            
            for spine in axins.spines.values():
                spine.set_linestyle((0, (5, 4, 1, 4)))
                spine.set_linewidth(1)

    plt.tight_layout()
    plt.show()
    
    return bin_edges


def calculate_objective_constrained(mu, sigma, eta=1, lower_bound=-np.inf, upper_bound=np.inf):
    """
    Solve the constrained optimization problem using MOSEK:
    max w^T * mu - 0.5 * eta * lambda * w^T * sigma * w
    subject to sum(w) = 1.
    """
    n = len(mu)  # Number of asset
    mu = mu.astype(np.float64)
    sigma = sigma.astype(np.float64)

    with Model("Portfolio Optimization") as M:
        
        # Defines the variables (holdings). Shortselling is not allowed.
        w = M.variable("w", n, Domain.greaterThan(lower_bound)) # Portfolio variables
        s = M.variable("s", 1, Domain.unbounded())      # Variance variable
        GT = np.linalg.cholesky(sigma).T

        # Total budget constraint
        M.constraint('budget', Expr.sum(w), Domain.equalsTo(1.0))
        M.constraint('max_w', w, Domain.lessThan(upper_bound))

        # Computes the risk
        M.constraint('variance', Expr.vstack(s, 0.5, GT @ w), Domain.inRotatedQCone())

        # Define objective as a weighted combination of return and variance
        M.objective('obj', ObjectiveSense.Maximize, w.T @ mu - s * 0.5 * eta)

        # Solve the problem
        M.solve()

        # Obtain the result
        w_optimal = w.level()

    return w_optimal


def calculate_all_weights(mean_paths, cov_paths, bound=0.05, eta=3, year=2001):
    """
    Calculate portfolio weights using various combinations of mean and covariance paths.
    
    Parameters:
    -----------
    mean_paths : dict
        Dictionary mapping strategy keys to paths of mean .npy files or .csv (for VW)
    cov_paths : dict
        Dictionary mapping strategy keys to paths of covariance .npy files
    bound : float
        Weight constraint for optimizer (symmetric)
    eta : float
        Risk aversion parameter
    year : int
        Year used to determine VW market value file
    
    Returns:
    --------
    dict
        Portfolio weights for all strategy combinations
    """
    results = {
        # Real data
        'Real Emp+Real Emp': [], 'Real Emp+Real LW': [],
        'Real BS+Real Emp': [], 'Real BS+Real LW': [],
        'Real OLSE+Real Emp': [], 'Real OLSE+Real LW': [],
        # Diffusion
        'Diff Emp+Diff Emp': [], 'Diff Emp+Diff LW': [],
        'Diff BS+Diff Emp': [], 'Diff BS+Diff LW': [],
        'Diff OLSE+Diff Emp': [], 'Diff OLSE+Diff LW': [],
        # Hybrid
        'Real Emp+Diff Emp': [], 'Diff Emp+Real Emp': [],
        # Benchmark
        'EW': [], 'VW': []
    }
    
    def compute_and_store_weight(mean_key, cov_key, result_key):
        mean = np.load(mean_paths[mean_key])
        cov = np.load(cov_paths[cov_key])
        w = calculate_objective_constrained(mean, cov, eta, -bound, bound)
        results[result_key].append(w / w.sum())

    # Real
    compute_and_store_weight('real_emp', 'real_emp', 'Real_Emp+Real_Emp')
    compute_and_store_weight('real_emp', 'real_lw', 'Real_Emp+Real_LW')
    compute_and_store_weight('real_bs', 'real_emp', 'Real_BS+Real_Emp')
    compute_and_store_weight('real_bs', 'real_lw', 'Real_BS+Real_LW')
    compute_and_store_weight('real_olse', 'real_emp', 'Real_OLSE+Real_Emp')
    compute_and_store_weight('real_olse', 'real_lw', 'Real_OLSE+Real_LW')

    # Diffusion
    compute_and_store_weight('diff_emp', 'diff_emp', 'Diff_Emp+Diff_Emp')
    compute_and_store_weight('diff_emp', 'diff_lw', 'Diff_Emp+Diff_LW')
    compute_and_store_weight('diff_bs', 'diff_emp', 'Diff_BS+Diff_Emp')
    compute_and_store_weight('diff_bs', 'diff_lw', 'Diff_BS+Diff_LW')
    compute_and_store_weight('diff_olse', 'diff_emp', 'Diff_OLSE+Diff_Emp')
    compute_and_store_weight('diff_olse', 'diff_lw', 'Diff_OLSE+Diff_LW')

    # Hybrid
    compute_and_store_weight('real_emp', 'diff_emp', 'Real_Emp+Diff_Emp')
    compute_and_store_weight('diff_emp', 'real_emp', 'Diff_Emp+Real_Emp')

    # Equal Weight
    dim = np.load(mean_paths['real_emp']).shape[0]
    results['EW'].append(np.ones(dim) / dim)

    # Value Weight
    vw_df = pd.read_csv(mean_paths['vw'], index_col=0)
    vw = vw_df.iloc[0, :].values
    vw = vw / vw.sum()
    vw[vw > 0.05] = 0.05
    results['VW'].append(vw / vw.sum())
    
    return results

def calculate_portfolio_returns(returns, weights, transaction_fee_rate=0.002):
    """
    Calculate portfolio returns with transaction costs.
    
    Args:
        returns (np.ndarray): Daily returns matrix of shape (n_days, n_assets)
        weights (np.ndarray): Portfolio weights matrix of shape (n_days, n_assets)
        transaction_fee_rate (float): Transaction fee rate per trade, default 0.002 (0.2%)
    
    Returns:
        tuple: (portfolio_returns, total_transaction_cost, turnover)
            - portfolio_returns (np.ndarray): Daily portfolio returns
            - total_transaction_cost (float): Total transaction costs incurred
            - turnover (float): Average daily turnover percentage
    """
    n_days, n_assets = returns.shape
    portfolio_returns = np.zeros(n_days)
    holdings = 1
    total_transaction_cost = 0
    turnover = 0

    for t in range(n_days-1):
        current_weight = weights[t] * (1 + returns[t])
        target_weight = weights[t+1]
        
        daily_transaction_cost = transaction_fee_rate * np.sum(np.abs(target_weight * np.sum(current_weight) - current_weight))
        total_transaction_cost += daily_transaction_cost
        
        portfolio_returns[t] = np.sum(weights[t] * returns[t]) - daily_transaction_cost
        holdings *= (1 + portfolio_returns[t])
        
        if np.sum(current_weight) > 0:
            turnover += 100 * np.sum(np.abs(target_weight - current_weight / np.sum(current_weight)))
    
    portfolio_returns[-1] = np.sum(weights[-1] * returns[-1])
    
    return portfolio_returns, total_transaction_cost, turnover / (n_days - 1) if n_days > 1 else 0

def calculate_portfolio_statistics(portfolio_returns, risk_free_data=None, eta=3):
    """
    Calculate portfolio performance statistics.
    
    Args:
        portfolio_returns (np.ndarray): Daily portfolio returns
        risk_free_data (np.ndarray, optional): Daily risk-free rate data
        eta (float): Risk aversion parameter, default 3
    
    Returns:
        dict: Dictionary containing portfolio statistics
            - Mean: Annualized mean return
            - Std: Annualized standard deviation
            - SR: Sharpe ratio
            - CER: Certainty equivalent return
            - CR: Cumulative return
            - MDD (%): Maximum drawdown percentage
    """
    # Calculate mean return
    if risk_free_data is not None:
        mean_return = np.mean(portfolio_returns - risk_free_data) * 252
    else:
        mean_return = np.mean(portfolio_returns) * 252
    
    # Calculate standard deviation
    std_dev = np.std(portfolio_returns) * np.sqrt(252)
    
    # Calculate Sharpe ratio
    sharpe_ratio = mean_return / std_dev
    
    # Calculate maximum drawdown
    cumulative_returns = np.cumprod(1 + portfolio_returns)
    peak = np.maximum.accumulate(cumulative_returns)
    drawdown = 100 * (peak - cumulative_returns) / (peak + 1e-5)
    max_drawdown = np.max(drawdown)
    
    # Calculate CER and cumulative return
    cer = mean_return - eta / 2 * (std_dev**2)
    cumulative_return = cumulative_returns[-1]
    
    return {
        "Mean": mean_return,
        "Std": std_dev,
        "SR": sharpe_ratio,
        "CER": cer,
        "CR": cumulative_return,
        "MDD (\%)": max_drawdown
    }

def calculate_portfolio_metrics(returns, weights, risk_free_data=None, transaction_fee_rate=0.002, eta=3):
    """
    Calculate comprehensive portfolio metrics.
    
    Args:
        returns (np.ndarray): Daily returns matrix of shape (n_days, n_assets)
        weights (np.ndarray): Portfolio weights matrix of shape (n_days, n_assets)
        risk_free_data (np.ndarray, optional): Daily risk-free rate data
        transaction_fee_rate (float): Transaction fee rate per trade, default 0.002 (0.2%)
        eta (float): Risk aversion parameter, default 3
    
    Returns:
        tuple: (portfolio_returns, metrics)
            - portfolio_returns (np.ndarray): Daily portfolio returns
            - metrics (dict): Dictionary containing portfolio statistics
    """
    portfolio_returns, _, turnover = calculate_portfolio_returns(returns, weights, transaction_fee_rate)
    metrics = calculate_portfolio_statistics(portfolio_returns, risk_free_data, eta)
    metrics["TO (\%)"] = turnover
    return portfolio_returns, metrics


def test_period(start_year, end_year, test_data_path, mean_paths_template, cov_paths_template, eta=3, fee=0.002):
    """
    Test performance of a comprehensive set of real/diffusion combinations.

    Parameters:
        start_year (int): Starting year for testing
        end_year (int): Ending year for testing
        test_data_path (str): Template string with {year} and {next_year} placeholders
        mean_paths_template (dict): Dict of method name -> template path for mean, with {year}
        cov_paths_template  (dict): Dict of method name -> template path for cov, with {year}
        eta (float): Risk aversion parameter
        fee (float): Transaction fee rate

    Returns:
        tuple: (portfolio_df, metrics_df)
    """
    # Define all method combinations
    method_names = [
        'Real Emp+Real Emp', 'Real Emp+Real LW', 'Real BS+Real Emp', 'Real BS+Real LW',
        'Real OLSE+Real Emp', 'Real OLSE+Real LW',
        'Diff Emp+Diff Emp', 'Diff Emp+Diff LW', 'Diff BS+Diff Emp', 'Diff BS+Diff LW',
        'Diff OLSE+Diff Emp', 'Diff OLSE+Diff LW',
        'Real Emp+Diff Emp', 'Diff Emp+Real Emp',
        'EW', 'VW'
    ]

    final_dfs = {name: pd.DataFrame() for name in method_names}
    final_test_data = pd.DataFrame()

    for year in range(start_year, end_year):
        # Load test data
        test_data = pd.read_csv(
            test_data_path.format(year=year+5, next_year=year+6), index_col=0
        )
        test_data.index = pd.to_datetime(test_data.index)
        test_data.columns = test_data.columns.astype("int64")
        final_test_data = pd.concat([final_test_data, test_data], axis=0).fillna(0.0)

        # Generate actual file paths for this year
        mean_paths = {k: v.format(year=year) for k, v in mean_paths_template.items()}
        cov_paths = {k: v.format(year=year) for k, v in cov_paths_template.items()}

        # Compute weights
        weight_results = calculate_all_weights(
            mean_paths, cov_paths, bound=0.05, eta=eta, year=year, df=test_data
        )

        for method_name, weight_df in zip(method_names, weight_results):
            final_dfs[method_name] = pd.concat(
                [final_dfs[method_name], weight_df], axis=0
            ).fillna(0.0)

    # Compute returns and metrics
    portfolio_results = []
    summary_results = []

    for method_name in method_names:
        weights_df = final_dfs[method_name]
        returns, metrics = calculate_portfolio_metrics(
            final_test_data.values, weights_df.values,
            transaction_fee_rate=fee, eta=eta
        )

        portfolio_results.append(returns)
        summary_results.append(metrics)

    # Construct output DataFrames
    portfolio_df = pd.DataFrame(
        portfolio_results, index=method_names, columns=final_test_data.index
    )
    metrics_df = pd.DataFrame(summary_results, index=method_names)

    print(metrics_df.to_latex(float_format="%.3f"))
    return portfolio_df, metrics_df

def plot_cumulative_log_returns(portfolio_df, risk_free_path, strategy_names, 
                                      display_names, sharpe_ratios,
                                      figsize=(6, 4), dpi=300):
    """
    Plot cumulative log returns for selected portfolio strategies using Sharpe-adjusted linewidths.

    Parameters:
    -----------
    portfolio_df : pd.DataFrame
        DataFrame with datetime index and strategy columns
    risk_free_path : str
        Path to .npy file containing risk-free rate (daily)
    strategy_names : list of str
        List of column names in portfolio_df to plot
    display_names : list of str
        List of names to display in legend (must match order of strategy_names)
    sharpe_ratios : dict
        Dictionary mapping strategy name to its Sharpe ratio
    figsize : tuple, optional
        Figure size
    dpi : int, optional
        Figure DPI
    """
    plt.figure(figsize=figsize, dpi=dpi)

    # Compute cumulative excess log-returns
    rf = np.load(risk_free_path)
    selected_returns = portfolio_df.loc[strategy_names].T
    excess_returns = selected_returns - rf[:, np.newaxis]
    cumulative_returns = 1 + np.log(1 + excess_returns).cumsum()
    cumulative_returns.iloc[0, :] = 1

    n_strategies = len(strategy_names)

    # Use fixed colors
    line_colors = ['C' + str(i % 10) for i in range(n_strategies)]
    line_styles = ['-'] * n_strategies

    # Normalize Sharpe ratios for linewidth (between 1 and 3)
    sharpe_values = np.array([sharpe_ratios[name] for name in strategy_names])
    min_sharpe, max_sharpe = sharpe_values.min(), sharpe_values.max()
    if max_sharpe - min_sharpe > 1e-6:
        norm_linewidths = 1 + 2 * (sharpe_values - min_sharpe) / (max_sharpe - min_sharpe)
    else:
        norm_linewidths = np.full_like(sharpe_values, fill_value=2.0)
    
    # Alpha: optional, e.g., fade lower-sharpe strategies
    norm_alphas = 0.6 + 0.4 * (sharpe_values - min_sharpe) / (max_sharpe - min_sharpe) if max_sharpe > min_sharpe else np.ones_like(sharpe_values)

    for i, name in enumerate(strategy_names):
        plt.plot(
            cumulative_returns[name],
            linewidth=norm_linewidths[i],
            linestyle=line_styles[i],
            color=line_colors[i],
            alpha=norm_alphas[i],
            label=display_names[i]
        )

    ax = plt.gca()
    ax.xaxis.set_major_locator(mdates.MonthLocator(interval=36))
    ax.xaxis.set_major_formatter(mdates.DateFormatter('%Y'))
    plt.xticks(rotation=0, fontsize=12)

    plt.legend(fontsize=10.5, loc='upper left')
    plt.yticks(fontsize=12)
    plt.ylabel("Cumulative Log-Return", fontsize=16)
    plt.yticks(np.arange(0, 5, 1))
    plt.tight_layout()
    plt.show()
