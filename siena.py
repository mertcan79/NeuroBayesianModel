from eeg import SymbolicEEG
from data.data_processing_siena import prepare_eeg_dataset
import numpy as np
import pandas as pd

def alpha_rule(data, params):
    alpha_power = np.mean(data[:, [col for col in data.columns if 'alpha' in col]])
    return np.where(alpha_power > 0.5, 0.0, -1e10)

def beta_theta_ratio_rule(data, params):
    beta_power = np.mean(data[:, [col for col in data.columns if 'beta' in col]])
    theta_power = np.mean(data[:, [col for col in data.columns if 'theta' in col]])
    return np.where(beta_power / theta_power > 1.5, 0.0, -1e10)

def alpha_theta_ratio_rule(data, params):
    alpha_power = np.mean(data[:, [col for col in data.columns if 'alpha' in col]])
    theta_power = np.mean(data[:, [col for col in data.columns if 'theta' in col]])
    ratio = alpha_power / theta_power
    return np.where(ratio > 1, 0.0, -1e10)  # Favors states where alpha power is greater than theta power

def main():
    # Prepare dataset
    eeg_dataset = prepare_eeg_dataset('/path/to/siena/dataset')
    
    # Initialize and fit model (this step would be done offline in a real app)
    model = SymbolicEEG(num_channels=32, num_features=len(eeg_dataset.columns))
    model.add_eeg_symbolic_rule(alpha_rule)
    model.add_eeg_symbolic_rule(beta_theta_ratio_rule)
    model.add_symbolic_rule(alpha_theta_ratio_rule)
    model.fit_eeg(eeg_dataset)

    # Generate insights for a single sample
    sample = eeg_dataset.iloc[0]
    insights = []

    # Analyze frequency dynamics
    freq_dynamics = model.analyze_frequency_dynamics(sample.values, 250)  # Assuming 250 Hz sampling rate
    insights.append(f"Frequency dynamics: {freq_dynamics}")

    # Classify cognitive state
    cognitive_state = model.classify_cognitive_state(sample)
    insights.append(f"Cognitive state: {cognitive_state}")

    # Estimate cognitive load
    cognitive_load = model.estimate_cognitive_load(sample)
    insights.append(f"Cognitive load: {cognitive_load}")

    # Analyze sleep stage
    sleep_stage = model.analyze_sleep_stages(sample)
    insights.append(f"Sleep stage: {sleep_stage}")

    # Analyze meditation depth
    meditation_depth = model.analyze_meditation_depth(sample)
    insights.append(f"Meditation depth: {meditation_depth}")

    # Analyze stress level
    stress_level = model.analyze_stress_level(sample)
    insights.append(f"Stress level: {stress_level}")

    # Perform causal inference
    causal_effect = model.perform_causal_inference_eeg(('alpha_ch1', 0.5), 'beta_ch1', sample.values, 250)
    insights.append(f"Causal effect of increasing alpha in channel 1 on beta in channel 1: {causal_effect}")

    # Generate symbolic insights
    symbolic_insights = model.generate_symbolic_insights(sample)
    insights.extend(symbolic_insights)

    # Print insights
    for insight in insights:
        print(insight)

if __name__ == "__main__":
    main()