import numpy as np
from scipy.stats import genpareto
from scipy.optimize import minimize
import matplotlib.pyplot as plt

def fit_gpd(target_mean=1.34, target_upper_ci=2.5, initial_shape=0.2, initial_scale=0.2, loc_param=1.0, sample_size=1000, method='Powell'):
    """
    Fits a Generalized Pareto Distribution (GPD) to match a target mean and upper CI.
    
    Parameters:
    - target_mean: Desired mean of the distribution.
    - target_upper_ci: Desired upper bound of the confidence interval.
    - initial_shape: Initial guess for the shape parameter.
    - initial_scale: Initial guess for the scale parameter.
    - loc_param: Location parameter (fixed).
    - sample_size: Number of samples to generate for fitting.
    - method: Optimization method (default is 'Powell').
    
    Returns:
    - shape_opt: Optimized shape parameter.
    - scale_opt: Optimized scale parameter.
    - fitted_mean: Mean of the fitted distribution.
    - fitted_low_99: Lower bound of the 99% confidence interval.
    - fitted_high_99: Upper bound of the 99% confidence interval.
    """

    # Define a function to optimize both the mean and upper CI
    def objective_gpd(params):
        shape_param, scale_param = params
        samples = genpareto.rvs(c=shape_param, loc=loc_param, scale=scale_param, size=sample_size)
        mean = np.mean(samples)
        high_99 = np.percentile(samples, 99.5)
        penalty = 0

        # Add penalties if parameters go out of expected bounds
        if shape_param < 0 or scale_param < 0:
            penalty += 1e6  # Large penalty for invalid parameters
        
        return (mean - target_mean) ** 2 + (high_99 - target_upper_ci) ** 2 + penalty

    # Initial guess for shape and scale
    initial_guess = [initial_shape, initial_scale]

    # Define bounds for the optimizer (e.g., shape and scale should be positive)
    bounds = [(0, None), (0, None)]  # No upper bound, but both should be >= 0

    # Optimize for the mean and upper CI being close to the target values
    result = minimize(objective_gpd, initial_guess, method=method, bounds=bounds)
    shape_opt, scale_opt = result.x

    # Generate sample data from the optimized GPD
    samples_optimized = genpareto.rvs(c=shape_opt, loc=loc_param, scale=scale_opt, size=sample_size)

    # Calculate mean and confidence intervals for the optimized distribution
    fitted_mean = np.mean(samples_optimized)
    fitted_low_99 = np.percentile(samples_optimized, 0.5)
    fitted_high_99 = np.percentile(samples_optimized, 99.5)

    return shape_opt, scale_opt, fitted_mean, fitted_low_99, fitted_high_99, loc_param


def plot_gpd(samples_optimized, shape_opt, loc_param, scale_opt, target_mean, target_upper_ci):
    """
    Plot the optimized Generalized Pareto Distribution (GPD) and the histogram of the samples.

    Parameters:
    - samples_optimized (ndarray): Array of samples from the optimized GPD.
    - shape_opt (float): Optimized shape parameter of the GPD.
    - loc_param (float): Location parameter of the GPD.
    - scale_opt (float): Optimized scale parameter of the GPD.
    - target_mean (float): Desired mean of the distribution.
    - target_upper_ci (float): Desired upper bound of the confidence interval.
    """
    # Plot the histogram of the samples
    plt.hist(samples_optimized, bins=30, density=True, alpha=0.6, color='b')

    # Plot the optimized GPD distribution
    xmin, xmax = plt.xlim()
    x = np.linspace(xmin, xmax, 100)
    p = genpareto.pdf(x, c=shape_opt, loc=loc_param, scale=scale_opt)
    plt.plot(x, p, 'k', linewidth=2)

    # Set the title and labels
    plt.title(f"Optimized Generalized Pareto Distribution (Mean Target: {target_mean}, Upper CI Target: {target_upper_ci})")
    plt.xlabel("Value")
    plt.ylabel("Density")

    # Show the plot
    plt.show()

def sample_gpd(shape_param, scale_param, loc_param=1.0, n=1000):
    """
    Generate random samples from a Generalized Pareto Distribution (GPD).

    Parameters:
    - shape_param (float): The shape parameter of the GPD.
    - scale_param (float): The scale parameter of the GPD.
    - loc_param (float): The location parameter of the GPD (default is 1.0).
    - n (int): The number of samples to generate.

    Returns:
    - samples (ndarray): An array of random samples from the GPD.
    """
    samples = genpareto.rvs(c=shape_param, loc=loc_param, scale=scale_param, size=n)
    return samples


# Example usage of the function
shape_opt, scale_opt, fitted_mean, fitted_low_99, fitted_high_99, loc_param = fit_gpd(
    target_mean=1.34,
    target_upper_ci=2.3,
    initial_shape=0.2,
    initial_scale=0.2,
    loc_param=1.0,
    sample_size=10000,
    method='Nelder-Mead'  # More robust method than 'Nelder-Mead'
)

# Print the results
print(f"Fitted Mean: {fitted_mean}")
print(f"Lower 99% Confidence Interval: {fitted_low_99}")
print(f"Upper 99% Confidence Interval: {fitted_high_99}")
print(f"Location Parameter: {loc_param}")
print(f"Optimized Shape Parameter: {shape_opt}")
print(f"Optimized Scale Parameter: {scale_opt}")



# shape = 0.20007812499999994
# scale = 0.19953125000000005
# loc = 1.0

# Generate and plot samples
# samples_gpd = sample_gpd(shape_param=shape_opt, scale_param=scale_opt, loc_param=loc_param, n=1000)
# plot_gpd(samples_gpd, shape_opt, loc_param, scale_opt, target_mean=1.34, target_upper_ci=2.4)
