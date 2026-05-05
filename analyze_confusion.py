import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import warnings
warnings.filterwarnings('ignore')

# Load metadata
df = pd.read_csv('Metadata_Train.csv')

# Get some drum and violin samples
drum_samples = df[df['Class'] == 'Sound_Drum'].head(10)['FileName'].tolist()
violin_samples = df[df['Class'] == 'Sound_Violin'].head(10)['FileName'].tolist()

print('Analyzing feature differences between Drums and Violins...')
print(f'Drum samples: {len(drum_samples)}')
print(f'Violin samples: {len(violin_samples)}')

# Feature extraction function (simplified version)
def extract_features(image_array):
    if image_array is None:
        return None

    try:
        # Perform SVD
        U, S, Vt = np.linalg.svd(image_array, full_matrices=False)

        # Normalize singular values
        S_sum = np.sum(S)
        if S_sum == 0:
            S_sum = 1e-10
        S_normalized = S / S_sum

        features = []

        # Top 10 singular values
        n_sv = min(10, len(S_normalized))
        sv_features = np.zeros(10)
        sv_features[:n_sv] = S_normalized[:n_sv]
        features.append(sv_features)

        # Energy concentration (first 5 thresholds)
        cumsum_S = np.cumsum(S_normalized)
        thresholds = [0.5, 0.7, 0.8, 0.9, 0.95]
        cumsum_features = []
        for thresh in thresholds:
            idx = np.where(cumsum_S >= thresh)[0]
            if len(idx) > 0:
                cumsum_features.append(idx[0] / max(len(S), 1.0))
            else:
                cumsum_features.append(1.0)
        features.append(np.array(cumsum_features))

        # Entropy
        entropy = -np.sum((S_normalized + 1e-10) * np.log(S_normalized + 1e-10))
        features.append(np.array([entropy]))

        # Image statistics
        image_stats = np.array([
            np.mean(image_array),
            np.std(image_array),
            np.max(image_array),
            np.percentile(image_array, 75)
        ])
        features.append(image_stats)

        return np.concatenate(features)

    except Exception as e:
        print(f'Error extracting features: {e}')
        return None

# Extract features for both classes
drum_features = []
violin_features = []

# Use the same feature extraction as the main classifier
def extract_advanced_features(image_array):
    """Extract the same features as the main classifier"""
    if image_array is None:
        return None
    
    try:
        # Perform SVD on the image
        U, S, Vt = np.linalg.svd(image_array, full_matrices=False)
        
        # Normalize singular values
        S_sum = np.sum(S)
        if S_sum == 0:
            S_sum = 1e-10
        S_normalized = S / S_sum
        
        features = []
        
        # ===== Feature Group 1: Singular Values (40 features) =====
        n_sv = min(40, len(S_normalized))
        sv_features = np.zeros(40)
        sv_features[:n_sv] = S_normalized[:n_sv]
        features.append(sv_features)
        
        # ===== Feature Group 2: Energy Concentration (8 features) =====
        cumsum_S = np.cumsum(S_normalized)
        thresholds = [0.3, 0.5, 0.7, 0.8, 0.9, 0.95, 0.99, 0.999]
        cumsum_features = []
        for thresh in thresholds:
            idx = np.where(cumsum_S >= thresh)[0]
            if len(idx) > 0:
                cumsum_features.append(idx[0] / max(len(S), 1.0))
            else:
                cumsum_features.append(1.0)
        features.append(np.array(cumsum_features))
        
        # ===== Feature Group 3: Spectral Entropy (3 features) =====
        entropy = -np.sum((S_normalized + 1e-10) * np.log(S_normalized + 1e-10))
        max_entropy = np.log(len(S)) if len(S) > 1 else 1
        renyi_entropy = np.log(np.sum(S_normalized ** 2)) / (1 - 2)
        entropy_features = np.array([entropy, entropy / max_entropy if max_entropy > 0 else 0, renyi_entropy])
        features.append(entropy_features)
        
        # ===== Feature Group 4: Decay Characteristics (10 features) =====
        decay_features = []
        if len(S_normalized) > 1:
            diffs = np.diff(S_normalized[:min(50, len(S_normalized))])
            decay_features.extend([np.mean(diffs), np.std(diffs), np.max(diffs), np.min(diffs)])
        else:
            decay_features.extend([0, 0, 0, 0])
        
        if len(S_normalized) > 2:
            diffs2 = np.diff(diffs)
            decay_features.extend([np.mean(diffs2) if len(diffs2) > 0 else 0, np.std(diffs2) if len(diffs2) > 0 else 0])
        else:
            decay_features.extend([0, 0])
        
        if len(S_normalized) > 5:
            decay_features.extend([
                S_normalized[0] / (S_normalized[1] + 1e-10),
                S_normalized[0] / (S_normalized[min(4, len(S_normalized)-1)] + 1e-10),
                S_normalized[min(4, len(S_normalized)-1)] / (S_normalized[min(9, len(S_normalized)-1)] + 1e-10)
            ])
        else:
            decay_features.extend([1, 1, 1])
        
        features.append(np.array(decay_features))
        
        # ===== Feature Group 5: Image Statistics (8 features) =====
        image_stats = np.array([
            np.mean(image_array), np.std(image_array), np.max(image_array), np.min(image_array),
            np.percentile(image_array, 25), np.percentile(image_array, 50), np.percentile(image_array, 75), np.percentile(image_array, 95)
        ])
        features.append(image_stats)
        
        # ===== Feature Group 6: U matrix characteristics (5 features) =====
        U_stats = np.array([
            np.mean(np.abs(U[:, 0])), np.std(U[:, 0]),
            np.mean(np.abs(U[:min(50, U.shape[1]), 0])),
            np.max(np.abs(U[:, 0])),
            np.sum(S_normalized[:5]) / np.sum(S_normalized)
        ])
        features.append(U_stats)
        
        # ===== NEW: Feature Group 7: Temporal Dynamics (11 features) =====
        temporal_features = []
        n_time_segments = 8
        time_step = image_array.shape[1] // n_time_segments
        
        for i in range(n_time_segments):
            start_col = i * time_step
            end_col = (i + 1) * time_step if i < n_time_segments - 1 else image_array.shape[1]
            segment = image_array[:, start_col:end_col]
            segment_energy = np.sum(segment)
            temporal_features.append(segment_energy)
        
        temporal_features = np.array(temporal_features)
        if np.sum(temporal_features) > 0:
            temporal_features = temporal_features / np.sum(temporal_features)
        
        temporal_features = np.concatenate([
            temporal_features,
            [np.std(temporal_features), np.max(temporal_features) - np.min(temporal_features),
             np.where(temporal_features == np.max(temporal_features))[0][0] / len(temporal_features)]
        ])
        features.append(temporal_features)
        
        # ===== NEW: Feature Group 8: Spectral Characteristics (7 features) =====
        spectral_features = []
        freq_bins = np.arange(image_array.shape[0])
        total_energy = np.sum(image_array, axis=1)
        if np.sum(total_energy) > 0:
            spectral_centroid = np.sum(freq_bins * total_energy) / np.sum(total_energy)
            spectral_features.append(spectral_centroid / image_array.shape[0])
        else:
            spectral_features.append(0)
        
        if np.sum(total_energy) > 0:
            spectral_spread = np.sqrt(np.sum(((freq_bins - spectral_centroid) ** 2) * total_energy) / np.sum(total_energy))
            spectral_features.append(spectral_spread / image_array.shape[0])
        else:
            spectral_features.append(0)
        
        cumsum_energy = np.cumsum(total_energy)
        if cumsum_energy[-1] > 0:
            rolloff_idx = np.where(cumsum_energy >= 0.85 * cumsum_energy[-1])[0]
            rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else len(freq_bins) - 1
            spectral_features.append(rolloff / image_array.shape[0])
        else:
            spectral_features.append(1.0)
        
        band_size = image_array.shape[0] // 4
        for i in range(4):
            start_freq = i * band_size
            end_freq = (i + 1) * band_size if i < 3 else image_array.shape[0]
            band_energy = np.sum(total_energy[start_freq:end_freq])
            spectral_features.append(band_energy)
        
        band_energies = np.array(spectral_features[-4:])
        if np.sum(band_energies) > 0:
            band_energies = band_energies / np.sum(band_energies)
            spectral_features[-4:] = band_energies
        
        features.append(np.array(spectral_features))
        
        # ===== NEW: Feature Group 9: Attack/Decay Characteristics (6 features) =====
        attack_decay_features = []
        time_energy = np.sum(image_array, axis=0)
        
        if len(time_energy) > 10:
            kernel_size = min(5, len(time_energy) // 2)
            if kernel_size > 0:
                kernel = np.ones(kernel_size) / kernel_size
                smoothed_energy = np.convolve(time_energy, kernel, mode='valid')
            else:
                smoothed_energy = time_energy
            
            attack_portion = smoothed_energy[:len(smoothed_energy)//4]
            if len(attack_portion) > 1:
                attack_slope = (attack_portion[-1] - attack_portion[0]) / len(attack_portion)
                attack_decay_features.append(attack_slope)
            else:
                attack_decay_features.append(0)
            
            decay_portion = smoothed_energy[len(smoothed_energy)//2:]
            if len(decay_portion) > 1:
                decay_slope = (decay_portion[-1] - decay_portion[0]) / len(decay_portion)
                attack_decay_features.append(decay_slope)
            else:
                attack_decay_features.append(0)
            
            peak_idx = np.argmax(smoothed_energy)
            peak_value = smoothed_energy[peak_idx]
            attack_decay_features.extend([
                peak_idx / len(smoothed_energy),
                peak_value / np.max(smoothed_energy) if np.max(smoothed_energy) > 0 else 0,
                np.std(smoothed_energy) / np.mean(smoothed_energy) if np.mean(smoothed_energy) > 0 else 0
            ])
        else:
            attack_decay_features.extend([0, 0, 0, 0, 0, 0])
        
        features.append(np.array(attack_decay_features))
        
        # ===== NEW: Feature Group 10: Harmonic Structure (4 features) =====
        harmonic_features = []
        freq_profile = np.sum(image_array, axis=1)
        
        if len(freq_profile) > 10:
            from scipy.signal import find_peaks
            peaks, properties = find_peaks(freq_profile, height=np.mean(freq_profile), distance=5)
            
            if len(peaks) > 0:
                if len(peaks) > 1:
                    peak_positions = peaks
                    spacings = np.diff(peak_positions)
                    spacing_std = np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else 1
                    harmonic_features.append(1 - min(spacing_std, 1))
                else:
                    harmonic_features.append(0)
                
                prominences = properties['peak_heights']
                avg_prominence = np.mean(prominences) / np.max(freq_profile) if np.max(freq_profile) > 0 else 0
                harmonic_features.append(avg_prominence)
                harmonic_features.append(len(peaks) / 20)
                
                peak_energy = np.sum(properties['peak_heights'])
                total_energy = np.sum(freq_profile)
                harmonic_features.append(peak_energy / total_energy if total_energy > 0 else 0)
            else:
                harmonic_features.extend([0, 0, 0, 0])
        else:
            harmonic_features.extend([0, 0, 0, 0])
        
        features.append(np.array(harmonic_features))
        
        return np.concatenate(features)
    
    except Exception as e:
        print(f'Error extracting features: {e}')
        return None

for filename in drum_samples:
    image_path = f'Train_submission/Train_Spectrograms/{filename.replace(".wav", ".png")}'
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        features = extract_advanced_features(img_array)
        if features is not None:
            drum_features.append(features)
    except Exception as e:
        print(f'Error processing drum {filename}: {e}')

for filename in violin_samples:
    image_path = f'Train_submission/Train_Spectrograms/{filename.replace(".wav", ".png")}'
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32) / 255.0
        features = extract_advanced_features(img_array)
        if features is not None:
            violin_features.append(features)
    except Exception as e:
        print(f'Error processing violin {filename}: {e}')

drum_features = np.array(drum_features)
violin_features = np.array(violin_features)

print(f'\nDrum features shape: {drum_features.shape}')
print(f'Violin features shape: {violin_features.shape}')

# Calculate feature statistics
drum_means = np.mean(drum_features, axis=0)
violin_means = np.mean(violin_features, axis=0)
drum_stds = np.std(drum_features, axis=0)
violin_stds = np.std(violin_features, axis=0)

print('\n=== FEATURE ANALYSIS (ADVANCED FEATURES) ===')
print('Now analyzing all 102 features including temporal, spectral, and harmonic features...')

# Focus on the new features (Groups 7-10)
feature_groups = {
    'Temporal Dynamics': (74, 85),  # 11 features
    'Spectral Characteristics': (85, 92),  # 7 features  
    'Attack/Decay': (92, 98),  # 6 features
    'Harmonic Structure': (98, 102)  # 4 features
}

for group_name, (start, end) in feature_groups.items():
    print(f'\n--- {group_name} Features ---')
    group_drum = drum_means[start:end]
    group_violin = violin_means[start:end]
    group_drum_std = drum_stds[start:end]
    group_violin_std = violin_stds[start:end]
    
    for i, (d_mean, v_mean, d_std, v_std) in enumerate(zip(group_drum, group_violin, group_drum_std, group_violin_std)):
        diff = abs(d_mean - v_mean)
        overlap = 1 - diff / (d_std + v_std) if (d_std + v_std) > 0 else 0
        overlap = max(0, min(1, overlap))
        status = "HIGH OVERLAP" if overlap > 0.7 else "MEDIUM" if overlap > 0.5 else "LOW OVERLAP"
        print(f'  Feature {i+1}: Drum={d_mean:.4f}±{d_std:.4f}, Violin={v_mean:.4f}±{v_std:.4f}, Diff={diff:.4f}, Overlap={overlap:.3f} ({status})')

# Overall analysis
overlaps = []
for i in range(len(drum_means)):
    d_mean, d_std = drum_means[i], drum_stds[i]
    v_mean, v_std = violin_means[i], violin_stds[i]
    diff = abs(d_mean - v_mean)
    if d_std + v_std > 0:
        overlap = 1 - diff / (d_std + v_std)
        overlap = max(0, min(1, overlap))
    else:
        overlap = 0
    overlaps.append(overlap)

print(f'\n=== OVERALL FEATURE OVERLAP ANALYSIS ===')
print(f'Average feature overlap: {np.mean(overlaps):.3f}')
print(f'Max overlap: {np.max(overlaps):.3f}')
print(f'Min overlap: {np.min(overlaps):.3f}')

# Find most discriminative features
discriminative_features = []
for i, overlap in enumerate(overlaps):
    if overlap < 0.5:  # Low overlap = discriminative
        discriminative_features.append((i, overlap))

print(f'\nMost discriminative features (overlap < 0.5): {len(discriminative_features)}')
for idx, overlap in sorted(discriminative_features, key=lambda x: x[1])[:10]:  # Top 10 most discriminative
    print(f'  Feature {idx}: overlap = {overlap:.3f}')