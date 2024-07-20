import numpy as np
import pandas as pd
from bayesian_network import BayesianNetwork
from sklearn.model_selection import train_test_split
import time
import logging

logging.basicConfig(filename='logs/bayesian_network_log.txt', level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger()

def generate_synthetic_data(n_samples=5000):
    A = np.random.normal(0, 1, n_samples)
    B = 0.5 * A + np.random.normal(0, 0.5, n_samples)
    C = 0.3 * A + 0.7 * B + np.random.normal(0, 0.3, n_samples)
    return pd.DataFrame({'A': A, 'B': B, 'C': C})

def progress_callback(progress):
    print(f"Progress: {progress*100:.0f}%")

if __name__ == "__main__":
    try:
        start_time = time.time()

        print("Generating synthetic data...")
        data = generate_synthetic_data(n_samples=5000)
        train_data, test_data = train_test_split(data, test_size=0.2, random_state=42)
        print(f"Data generated: {len(data)} samples, {len(data.columns)} features")

        print("Creating and fitting Bayesian Network...")
        bn = BayesianNetwork(method='hill_climb', max_parents=2)
        prior_edges = [('A', 'B'), ('A', 'C'), ('B', 'C')]
        bn.fit(train_data, prior_edges, progress_callback=progress_callback)

        print("Evaluating model...")
        test_ll = bn.log_likelihood(test_data)
        print(f"Test log-likelihood: {test_ll:.4f}")

        print("Performing cross-validation...")
        mean_ll, std_ll = bn.cross_validate(data, k_folds=5)
        print(f"Cross-validation: Mean LL = {mean_ll:.4f}, Std = {std_ll:.4f}")

        print("Computing sensitivity (this may take a while)...")
        sensitivity = bn.compute_sensitivity('C', num_samples=1000)  # Reduced number of samples
        top_sensitivities = sorted(sensitivity.items(), key=lambda x: x[1], reverse=True)[:3]
        for node, value in top_sensitivities:
            print(f"Sensitivity of C to {node}: {value:.4f}")

        print("Performing Metropolis-Hastings sampling...")
        observed_data = {'A': 0.5}
        mh_samples = bn.metropolis_hastings(observed_data, num_samples=1000)  # Reduced number of samples
        for node, samples in mh_samples.items():
            if node not in observed_data:
                print(f"{node}: Mean = {np.mean(samples):.4f}, Std = {np.std(samples):.4f}")

        print(f"Total execution time: {time.time() - start_time:.2f} seconds")

    except Exception as e:
        print(f"An error occurred: {str(e)}")
        logger.error(f"Error in main execution: {str(e)}", exc_info=True)

    finally:
        # Ensure all logs are flushed to file
        for handler in logger.handlers:
            handler.flush()
            handler.close()