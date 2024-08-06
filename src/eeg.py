from symbolic_network import SymbolicBayesianNetwork
import numpy as np
import pandas as pd
from scipy.signal import welch
import mne
from scipy import signal
from scipy import stats
from sklearn.mixture import BayesianGaussianMixture
import matplotlib.pyplot as plt
import ruptures as rpt

class SymbolicEEG(SymbolicBayesianNetwork):
    def __init__(self, num_channels, num_bands=5, time_window=1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.num_channels = num_channels
        self.num_bands = num_bands
        self.time_window = time_window
        self.feature_names = self._generate_feature_names()

    def fit(self, eeg_data, sfreq, behavioral_data=None):
        preprocessed_eeg = self.preprocess_eeg(eeg_data, sfreq)
        eeg_features = self.extract_features(preprocessed_eeg, sfreq)
        if behavioral_data is not None:
            combined_data = pd.concat([pd.DataFrame(eeg_features), behavioral_data], axis=1)
        else:
            combined_data = pd.DataFrame(eeg_features)
        super().fit(combined_data)

    def _generate_feature_names(self):
        bands = ['delta', 'theta', 'alpha', 'beta', 'gamma']
        features = [f'{band}_ch{i}_t{t}' for band in bands[:self.num_bands]
                    for i in range(self.num_channels)
                    for t in range(self.time_window)]
        features += [f'coherence_{i}_{j}_t{t}' for i in range(self.num_channels)
                     for j in range(i+1, self.num_channels)
                     for t in range(self.time_window)]
        return features

    def preprocess_eeg(self, raw_eeg, sfreq):
        """
        Preprocess the raw EEG data by applying filtering and artifact rejection.

        Args:
            raw_eeg (np.ndarray): Raw EEG data.
            sfreq (float): Sampling frequency of the EEG data.

        Returns:
            np.ndarray: Preprocessed EEG data.
        """
        # Apply bandpass filtering
        filtered_eeg = self._filter_eeg(raw_eeg, sfreq)

        # Apply notch filter to remove line noise (e.g., 50 Hz or 60 Hz)
        notch_filtered_eeg = mne.filter.notch_filter(filtered_eeg, Fs=sfreq, freqs=50)

        # Additional preprocessing steps can be added here, such as artifact rejection

        return notch_filtered_eeg

    def _filter_eeg(self, eeg, sfreq):
        """
        Apply a bandpass filter to the EEG data to remove noise and artifacts outside the frequency range of interest.

        Args:
            eeg (np.ndarray): EEG data.
            sfreq (float): Sampling frequency of the EEG data.

        Returns:
            np.ndarray: Filtered EEG data.
        """
        # Apply bandpass filtering (1-40 Hz)
        filtered_eeg = mne.filter.filter_data(eeg, sfreq, l_freq=1, h_freq=40)
        return filtered_eeg

    def extract_features(self, eeg, sfreq):
        features = []
        for t in range(self.time_window):
            window = eeg[:, t*sfreq:(t+1)*sfreq]
            band_powers = self._compute_band_powers(window, sfreq)
            coherence = self._compute_coherence(window, sfreq)
            features.extend(band_powers)
            features.extend(coherence)
        return np.array(features)

    def _compute_band_powers(self, eeg, sfreq):
        bands = [(1, 4), (4, 8), (8, 13), (13, 30), (30, 40)]
        powers = []
        for ch in range(self.num_channels):
            f, psd = welch(eeg[ch], fs=sfreq, nperseg=sfreq)
            for low, high in bands[:self.num_bands]:
                power = np.mean(psd[(f >= low) & (f < high)])
                powers.append(power)
        return powers

    def _compute_coherence(self, eeg, sfreq):
        # Compute coherence
        coh, freqs = self.spectral_connectivity(eeg, method='coh', fmin=1, fmax=40, fs=sfreq)

        # Extract the coherence values for each pair of channels
        coherence_values = []
        for i in range(self.num_channels):
            for j in range(i + 1, self.num_channels):
                coherence_values.append(np.mean(coh[i, j]))  # Average over frequency bands

        return coherence_values

    def fit_eeg(self, eeg_data, sfreq, behavioral_data=None):
        preprocessed_eeg = self.preprocess_eeg(eeg_data, sfreq)
        eeg_features = self.extract_features(preprocessed_eeg, sfreq)
        if behavioral_data is not None:
            combined_data = pd.concat([pd.DataFrame(eeg_features), behavioral_data], axis=1)
        else:
            combined_data = pd.DataFrame(eeg_features)
        super().fit(combined_data)

    def predict_eeg(self, eeg_data, sfreq):
        preprocessed_eeg = self.preprocess_eeg(eeg_data, sfreq)
        eeg_features = self.extract_features(preprocessed_eeg, sfreq)
        return super().predict(pd.DataFrame(eeg_features))

    def add_eeg_symbolic_rule(self, rule_function):
        self.add_symbolic_rule({
            'type': 'custom',
            'function': rule_function
        })  

    def analyze_frequency_dynamics(self, eeg_data, sfreq):
        f, t, Sxx = signal.spectrogram(eeg_data, fs=sfreq, nperseg=sfreq)

        delta = np.mean(Sxx[(f >= 1) & (f < 4)], axis=0)
        theta = np.mean(Sxx[(f >= 4) & (f < 8)], axis=0)
        alpha = np.mean(Sxx[(f >= 8) & (f < 13)], axis=0)
        beta = np.mean(Sxx[(f >= 13) & (f < 30)], axis=0)

        return {
            'delta': delta,
            'theta': theta,
            'alpha': alpha,
            'beta': beta,
            'time': t
        }

    def classify_cognitive_state(self, eeg_features):
        alpha_power = np.mean([eeg_features[f'alpha_ch{i}'] for i in range(self.num_channels)])
        beta_power = np.mean([eeg_features[f'beta_ch{i}'] for i in range(self.num_channels)])
        theta_power = np.mean([eeg_features[f'theta_ch{i}'] for i in range(self.num_channels)])

        if alpha_power > beta_power and alpha_power > theta_power:
            return "Relaxed"
        elif beta_power > alpha_power and beta_power > theta_power:
            return "Focused"
        elif theta_power > alpha_power and theta_power > beta_power:
            return "Drowsy"
        else:
            return "Neutral"

    def analyze_seizure_types(self, dataset):
        seizure_counts = dataset['seizure_type'].value_counts()
        return f"Seizure type distribution:\n{seizure_counts}"

    def analyze_age_related_differences(self, dataset):
        young = dataset[dataset['age'] < 30]
        old = dataset[dataset['age'] >= 30]
        
        young_power = young[[col for col in young.columns if 'power_' in col]].mean()
        old_power = old[[col for col in old.columns if 'power_' in col]].mean()
        
        t_stats, p_values = stats.ttest_ind(young_power, old_power)
        
        return f"Age-related differences in spectral power:\nt-statistic: {t_stats}\np-values: {p_values}"

    def analyze_gender_differences(self, dataset):
        male = dataset[dataset['sex'] == 'm']
        female = dataset[dataset['sex'] == 'f']
        
        male_pli = male[[col for col in male.columns if 'pli_' in col]].mean()
        female_pli = female[[col for col in female.columns if 'pli_' in col]].mean()
        
        t_stats, p_values = stats.ttest_ind(male_pli, female_pli)
        
        return f"Gender differences in PLI:\nt-statistic: {t_stats}\np-values: {p_values}"

    def analyze_sleep_stages(self, eeg_features):
        delta_power = np.mean([eeg_features[f'delta_ch{i}'] for i in range(self.num_channels)])
        theta_power = np.mean([eeg_features[f'theta_ch{i}'] for i in range(self.num_channels)])
        alpha_power = np.mean([eeg_features[f'alpha_ch{i}'] for i in range(self.num_channels)])
        
        if delta_power > 0.5:
            return "Deep Sleep"
        elif theta_power > 0.5:
            return "Light Sleep"
        elif alpha_power > 0.5:
            return "REM"
        else:
            return "Awake"

    def analyze_meditation_depth(self, eeg_features):
        alpha_power = np.mean([eeg_features[f'alpha_ch{i}'] for i in range(self.num_channels)])
        theta_power = np.mean([eeg_features[f'theta_ch{i}'] for i in range(self.num_channels)])
        
        meditation_index = (alpha_power + theta_power) / 2
        if meditation_index > 0.7:
            return "Deep Meditation"
        elif meditation_index > 0.4:
            return "Light Meditation"
        else:
            return "Not Meditating"

    def analyze_stress_level(self, eeg_features):
        beta_power = np.mean([eeg_features[f'beta_ch{i}'] for i in range(self.num_channels)])
        alpha_power = np.mean([eeg_features[f'alpha_ch{i}'] for i in range(self.num_channels)])
        
        stress_index = beta_power / alpha_power
        if stress_index > 2:
            return "High Stress"
        elif stress_index > 1.5:
            return "Moderate Stress"
        else:
            return "Low Stress"

    def perform_causal_inference_eeg(self, intervention, target, eeg_data, sfreq):
        baseline_features = self.extract_features(eeg_data, sfreq)
        intervention_data = eeg_data.copy()
        intervention_data[self.feature_names.index(intervention[0])] = intervention[1]
        intervention_features = self.extract_features(intervention_data, sfreq)

        baseline_prediction = self.predict(pd.DataFrame([baseline_features]))
        intervention_prediction = self.predict(pd.DataFrame([intervention_features]))

        return intervention_prediction[target] - baseline_prediction[target]

    def spectral_connectivity(data, method='coh', fmin=0, fmax=np.inf, fs=1.0):
        n_channels = data.shape[1]
        f, Pxx = signal.welch(data, fs=fs, nperseg=min(data.shape[0], 256))
        
        freq_mask = (f >= fmin) & (f <= fmax)
        Pxx = Pxx[:, freq_mask]
        f = f[freq_mask]
        
        conn = np.zeros((n_channels, n_channels, len(f)))
        
        for i in range(n_channels):
            for j in range(i+1, n_channels):
                if method == 'coh':
                    f, Cxy = signal.coherence(data[:, i], data[:, j], fs=fs, nperseg=min(data.shape[0], 256))
                    conn[i, j] = Cxy[freq_mask]
                    conn[j, i] = conn[i, j]
        
        return conn, f

    def extract_motor_imagery_features(self, eeg_data, sfreq):
        features = []
        for ch in range(eeg_data.shape[0]):
            f, psd = welch(eeg_data[ch], fs=sfreq, nperseg=sfreq)
            mu_power = np.mean(psd[(f >= 8) & (f <= 12)])
            beta_power = np.mean(psd[(f >= 13) & (f <= 30)])
            features.extend([mu_power, beta_power])
        return np.array(features)

    def predict_motor_imagery(self, eeg_data, sfreq):
        motor_features = self.extract_motor_imagery_features(eeg_data, sfreq)
        return self.predict(motor_features)

    def detect_change_points(self, eeg_data, sfreq):
        features = self.extract_motor_imagery_features(eeg_data, sfreq)
        algo = rpt.Pelt(model="rbf").fit(features)
        change_points = algo.predict(pen=10)
        return change_points
    
    def nonparametric_clustering(self, features, n_components=10):
        dpgmm = BayesianGaussianMixture(n_components=n_components, weight_concentration_prior_type='dirichlet_process')
        dpgmm.fit(features)
        return dpgmm
    
    def model_criticism(self, X, y):
        y_pred = self.predict(X)
        residuals = y - y_pred.mean(axis=0)
        plt.scatter(y_pred.mean(axis=0), residuals)
        plt.xlabel('Predicted')
        plt.ylabel('Residuals')
        plt.title('Residual Plot')
        plt.show()