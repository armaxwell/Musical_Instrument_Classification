"""
SVD-based Image Classification for Instrument Spectrograms
Classifies spectrograms into: Piano, Drum, Guitar, Violin using Singular Value Decomposition
Advanced optimization with feature engineering and hyperparameter tuning
"""

import os
import pandas as pd
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.base import clone
from sklearn.svm import SVC
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from scipy.signal import find_peaks
import warnings
warnings.filterwarnings('ignore')

class SpectrogramSVDClassifier:
    """Advanced SVD-based classifier for spectrogram images"""
    
    def __init__(self, image_size=(200, 200), n_components=120):
        """
        Initialize the classifier with advanced parameters
        Args:
            image_size: Tuple of (height, width) for resized images
            n_components: Number of features for feature extraction
        """
        self.image_size = image_size
        self.n_components = n_components
        self.scaler = StandardScaler()
        self.pca = PCA(n_components=min(90, n_components-20), random_state=42)  # Reduce noise, increased from 80
        
        # Optimized SVM with tuned hyperparameters
        self.classifier = SVC(
            kernel='rbf',
            C=50,  # Increased regularization for less overfitting
            gamma=0.0005,  # Fine-tuned gamma for RBF kernel
            probability=True,
            class_weight='balanced',  # Handle class imbalance
            random_state=42,
            max_iter=3000,
            tol=1e-3
        )
        
        self.label_mapping = {}
        self.reverse_label_mapping = {}
        self.cv_scores = None
        
    def load_image(self, image_path):
        """Load and preprocess an image"""
        try:
            img = Image.open(image_path).convert('L')  # Convert to grayscale
            img = img.resize(self.image_size)
            return np.array(img, dtype=np.float32) / 255.0  # Normalize to [0, 1]
        except Exception as e:
            print(f"Error loading {image_path}: {e}")
            return None
    
    def extract_svd_features(self, image_array):
        """Extract advanced SVD features from image with multiple analysis techniques"""
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
            # Thresholds for energy concentration
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
            # Also compute Rényi entropy for different alpha
            renyi_entropy = np.log(np.sum(S_normalized ** 2)) / (1 - 2)
            entropy_features = np.array([
                entropy,
                entropy / max_entropy if max_entropy > 0 else 0,
                renyi_entropy
            ])
            features.append(entropy_features)
            
            # ===== Feature Group 4: Decay Characteristics (10 features) =====
            decay_features = []
            
            # First-order differences decay
            if len(S_normalized) > 1:
                diffs = np.diff(S_normalized[:min(50, len(S_normalized))])
                decay_features.extend([
                    np.mean(diffs),
                    np.std(diffs),
                    np.max(diffs),
                    np.min(diffs)
                ])
            else:
                decay_features.extend([0, 0, 0, 0])
            
            # Curvature (second-order differences)
            if len(S_normalized) > 2:
                diffs2 = np.diff(diffs)
                decay_features.extend([
                    np.mean(diffs2) if len(diffs2) > 0 else 0,
                    np.std(diffs2) if len(diffs2) > 0 else 0,
                ])
            else:
                decay_features.extend([0, 0])
            
            # Specific singular value ratios
            if len(S_normalized) > 5:
                decay_features.extend([
                    S_normalized[0] / (S_normalized[1] + 1e-10),  # Ratio of first two
                    S_normalized[0] / (S_normalized[min(4, len(S_normalized)-1)] + 1e-10),  # First to 5th
                    S_normalized[min(4, len(S_normalized)-1)] / (S_normalized[min(9, len(S_normalized)-1)] + 1e-10)  # 5th to 10th
                ])
            else:
                decay_features.extend([1, 1, 1])
            
            features.append(np.array(decay_features))
            
            # ===== Feature Group 5: Image Statistics (8 features) =====
            # Directly from image pixel values
            image_stats = np.array([
                np.mean(image_array),
                np.std(image_array),
                np.max(image_array),
                np.min(image_array),
                np.percentile(image_array, 25),
                np.percentile(image_array, 50),
                np.percentile(image_array, 75),
                np.percentile(image_array, 95)
            ])
            features.append(image_stats)
            
            # ===== Feature Group 6: U matrix characteristics (5 features) =====
            # From the left singular vectors
            U_stats = np.array([
                np.mean(np.abs(U[:, 0])),  # Mean of first left singular vector
                np.std(U[:, 0]),
                np.mean(np.abs(U[:min(50, U.shape[1]), 0])),
                np.max(np.abs(U[:, 0])),
                np.sum(S_normalized[:5]) / np.sum(S_normalized)  # Cumulative first 5 singular values ratio
            ])
            features.append(U_stats)
            
            # ===== NEW: Feature Group 7: Temporal Dynamics (12 features) =====
            # Analyze energy distribution across time (horizontal axis)
            temporal_features = []
            
            # Divide spectrogram into time segments
            n_time_segments = 8
            time_step = image_array.shape[1] // n_time_segments
            
            for i in range(n_time_segments):
                start_col = i * time_step
                end_col = (i + 1) * time_step if i < n_time_segments - 1 else image_array.shape[1]
                segment = image_array[:, start_col:end_col]
                
                # Energy in this time segment
                segment_energy = np.sum(segment)
                temporal_features.append(segment_energy)
            
            # Normalize temporal features
            temporal_features = np.array(temporal_features)
            if np.sum(temporal_features) > 0:
                temporal_features = temporal_features / np.sum(temporal_features)
            
            # Add temporal statistics
            temporal_features = np.concatenate([
                temporal_features,  # 8 normalized energy values
                [np.std(temporal_features),  # Temporal variance
                 np.max(temporal_features) - np.min(temporal_features),  # Peak-to-valley
                 np.where(temporal_features == np.max(temporal_features))[0][0] / len(temporal_features)]  # Peak position
            ])
            features.append(temporal_features)
            
            # ===== NEW: Feature Group 8: Spectral Characteristics (8 features) =====
            # Analyze frequency domain characteristics
            spectral_features = []
            
            # Spectral centroid (weighted average of frequency bins)
            freq_bins = np.arange(image_array.shape[0])
            total_energy = np.sum(image_array, axis=1)
            if np.sum(total_energy) > 0:
                spectral_centroid = np.sum(freq_bins * total_energy) / np.sum(total_energy)
                spectral_features.append(spectral_centroid / image_array.shape[0])  # Normalized
            else:
                spectral_features.append(0)
            
            # Spectral spread (variance around centroid)
            if np.sum(total_energy) > 0:
                spectral_spread = np.sqrt(np.sum(((freq_bins - spectral_centroid) ** 2) * total_energy) / np.sum(total_energy))
                spectral_features.append(spectral_spread / image_array.shape[0])  # Normalized
            else:
                spectral_features.append(0)
            
            # Spectral rolloff (frequency below which 85% of energy lies)
            cumsum_energy = np.cumsum(total_energy)
            if cumsum_energy[-1] > 0:
                rolloff_idx = np.where(cumsum_energy >= 0.85 * cumsum_energy[-1])[0]
                rolloff = rolloff_idx[0] if len(rolloff_idx) > 0 else len(freq_bins) - 1
                spectral_features.append(rolloff / image_array.shape[0])  # Normalized
            else:
                spectral_features.append(1.0)
            
            # Frequency band energies (divide into 4 bands)
            band_size = image_array.shape[0] // 4
            for i in range(4):
                start_freq = i * band_size
                end_freq = (i + 1) * band_size if i < 3 else image_array.shape[0]
                band_energy = np.sum(total_energy[start_freq:end_freq])
                spectral_features.append(band_energy)
            
            # Normalize band energies
            band_energies = np.array(spectral_features[-4:])
            if np.sum(band_energies) > 0:
                band_energies = band_energies / np.sum(band_energies)
                spectral_features[-4:] = band_energies
            
            features.append(np.array(spectral_features))
            
            # ===== NEW: Feature Group 9: Attack/Decay Characteristics (6 features) =====
            # Analyze how energy changes over time (attack/decay patterns)
            attack_decay_features = []
            
            # Time series of total energy per column (time evolution)
            time_energy = np.sum(image_array, axis=0)
            
            if len(time_energy) > 10:
                # Smooth the time energy signal
                kernel_size = min(5, len(time_energy) // 2)
                if kernel_size > 0:
                    kernel = np.ones(kernel_size) / kernel_size
                    smoothed_energy = np.convolve(time_energy, kernel, mode='valid')
                else:
                    smoothed_energy = time_energy
                
                # Attack: initial rise
                attack_portion = smoothed_energy[:len(smoothed_energy)//4]
                if len(attack_portion) > 1:
                    attack_slope = (attack_portion[-1] - attack_portion[0]) / len(attack_portion)
                    attack_decay_features.append(attack_slope)
                else:
                    attack_decay_features.append(0)
                
                # Decay: later decay
                decay_portion = smoothed_energy[len(smoothed_energy)//2:]
                if len(decay_portion) > 1:
                    decay_slope = (decay_portion[-1] - decay_portion[0]) / len(decay_portion)
                    attack_decay_features.append(decay_slope)
                else:
                    attack_decay_features.append(0)
                
                # Peak characteristics
                peak_idx = np.argmax(smoothed_energy)
                peak_value = smoothed_energy[peak_idx]
                attack_decay_features.extend([
                    peak_idx / len(smoothed_energy),  # Peak position (normalized)
                    peak_value / np.max(smoothed_energy) if np.max(smoothed_energy) > 0 else 0,  # Peak prominence
                    np.std(smoothed_energy) / np.mean(smoothed_energy) if np.mean(smoothed_energy) > 0 else 0  # Variability
                ])
            else:
                attack_decay_features.extend([0, 0, 0, 0, 0, 0])
            
            features.append(np.array(attack_decay_features))
            
            # ===== NEW: Feature Group 10: Harmonic Structure (4 features) =====
            # Analyze harmonic vs noise characteristics
            harmonic_features = []
            
            # Harmonic-to-noise ratio approximation
            # Look for peaks in the frequency domain
            freq_profile = np.sum(image_array, axis=1)
            
            if len(freq_profile) > 10:
                # Find local maxima (potential harmonics)
                from scipy.signal import find_peaks
                peaks, properties = find_peaks(freq_profile, height=np.mean(freq_profile), distance=5)
                
                if len(peaks) > 0:
                    # Harmonic spacing regularity
                    if len(peaks) > 1:
                        peak_positions = peaks
                        spacings = np.diff(peak_positions)
                        spacing_std = np.std(spacings) / np.mean(spacings) if np.mean(spacings) > 0 else 1
                        harmonic_features.append(1 - min(spacing_std, 1))  # Regularity score
                    else:
                        harmonic_features.append(0)
                    
                    # Average peak prominence
                    prominences = properties['peak_heights']
                    avg_prominence = np.mean(prominences) / np.max(freq_profile) if np.max(freq_profile) > 0 else 0
                    harmonic_features.append(avg_prominence)
                    
                    # Number of significant peaks
                    harmonic_features.append(len(peaks) / 20)  # Normalized by expected max
                    
                    # Energy concentration in peaks
                    peak_energy = np.sum(properties['peak_heights'])
                    total_energy = np.sum(freq_profile)
                    harmonic_features.append(peak_energy / total_energy if total_energy > 0 else 0)
                else:
                    harmonic_features.extend([0, 0, 0, 0])
            else:
                harmonic_features.extend([0, 0, 0, 0])
            
            features.append(np.array(harmonic_features))
            
            # Concatenate all features
            all_features = np.concatenate(features)
            
            # Ensure exact length
            if len(all_features) < self.n_components:
                all_features = np.pad(all_features, (0, self.n_components - len(all_features)), mode='constant')
            else:
                all_features = all_features[:self.n_components]
            
            return all_features
            
        except Exception as e:
            print(f"Error extracting SVD features: {e}")
            return None
    
    def load_data(self, spectrogram_dir, metadata_file):
        """Load spectrograms and labels from directory"""
        print(f"Loading metadata from {metadata_file}...")
        df = pd.read_csv(metadata_file)
        
        print(f"Loading spectrograms from {spectrogram_dir}...")
        X = []  # Features
        y = []  # Labels
        filenames = []
        
        total_samples = len(df)
        loaded_count = 0
        
        for idx, row in df.iterrows():
            wav_filename = row['FileName']
            label = row['Class']
            
            # Convert .wav to .png
            image_filename = wav_filename.replace('.wav', '.png')
            image_path = os.path.join(spectrogram_dir, image_filename)
            
            if os.path.exists(image_path):
                # Load image
                img_array = self.load_image(image_path)
                
                # Extract SVD features
                features = self.extract_svd_features(img_array)
                
                if features is not None:
                    X.append(features)
                    y.append(label)
                    filenames.append(image_filename)
                    loaded_count += 1
                    
                    if (idx + 1) % 100 == 0:
                        print(f"Processed {idx + 1}/{total_samples} samples...")
            else:
                print(f"Warning: Image not found - {image_path}")
        
        print(f"Successfully loaded {loaded_count}/{total_samples} samples")
        
        # Create label mapping
        unique_labels = list(set(y))
        self.label_mapping = {label: idx for idx, label in enumerate(unique_labels)}
        self.reverse_label_mapping = {idx: label for label, idx in self.label_mapping.items()}
        
        # Convert labels to integers
        y_encoded = np.array([self.label_mapping[label] for label in y])
        
        return np.array(X), y_encoded, y, filenames
    
    def train(self, X, y):
        """Train the SVM classifier with PCA and cross-validation"""
        print("\nTraining advanced SVM classifier...")
        
        # Standardize features
        X_scaled = self.scaler.fit_transform(X)
        
        # Apply PCA for noise reduction and dimensionality reduction
        print("Applying PCA for noise reduction...")
        X_pca = self.pca.fit_transform(X_scaled)
        explained_var = np.sum(self.pca.explained_variance_ratio_)
        print(f"PCA: {X_scaled.shape[1]} → {X_pca.shape[1]} features (explains {explained_var*100:.1f}% variance)")
        
        # Perform cross-validation
        print("Performing 5-fold cross-validation...")
        self.cv_scores = cross_val_score(self.classifier, X_pca, y, cv=5, 
                                         scoring='accuracy', n_jobs=-1)
        print(f"CV scores: {[f'{s:.4f}' for s in self.cv_scores]}")
        print(f"Mean CV accuracy: {self.cv_scores.mean():.4f} (+/- {self.cv_scores.std():.4f})")
        
        # Train classifier on PCA-transformed data
        print("\nTraining on full training set...")
        self.classifier.fit(X_pca, y)
        
        # Get training accuracy
        train_pred = self.classifier.predict(X_pca)
        train_accuracy = accuracy_score(y, train_pred)
        train_f1 = f1_score(y, train_pred, average='weighted')
        
        print(f"Training accuracy: {train_accuracy:.4f}")
        print(f"Training F1-score (weighted): {train_f1:.4f}")
        
        return train_accuracy
    
    def plot_cv_and_training_curves(self, X_train, y_train, X_test, y_test, train_sizes=None):
        """Plot both CV fold accuracy and training accuracy curve side-by-side."""
        if self.cv_scores is None:
            print("No CV scores available. Run train() before plotting.")
            return

        print("\nGenerating combined training curves plot...")
        if train_sizes is None:
            train_sizes = np.linspace(0.1, 1.0, 10)

        train_sizes = np.clip(train_sizes, 0.05, 1.0)
        train_sizes = np.unique(train_sizes)

        train_acc = []
        test_acc = []
        sample_counts = []

        for frac in train_sizes:
            if float(frac) >= 1.0:
                X_sub = X_train
                y_sub = y_train
            else:
                X_sub, _, y_sub, _ = train_test_split(
                    X_train, y_train, train_size=float(frac), random_state=42, stratify=y_train
                )
            sample_counts.append(len(X_sub))

            local_scaler = clone(self.scaler)
            local_pca = clone(self.pca)
            local_clf = clone(self.classifier)

            X_sub_scaled = local_scaler.fit_transform(X_sub)
            X_sub_pca = local_pca.fit_transform(X_sub_scaled)
            local_clf.fit(X_sub_pca, y_sub)

            train_pred = local_clf.predict(X_sub_pca)
            train_acc.append(accuracy_score(y_sub, train_pred))

            X_test_scaled = local_scaler.transform(X_test)
            X_test_pca = local_pca.transform(X_test_scaled)
            test_pred = local_clf.predict(X_test_pca)
            test_acc.append(accuracy_score(y_test, test_pred))

            print(f"  Training size {len(X_sub)} → train acc {train_acc[-1]:.4f}, test acc {test_acc[-1]:.4f}")

        # Create figure with 2 subplots side-by-side
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

        # Left subplot: CV fold accuracy
        fold_numbers = np.arange(1, len(self.cv_scores) + 1)
        ax1.plot(fold_numbers, self.cv_scores, marker='o', linestyle='-', color='tab:blue', linewidth=2, markersize=8)
        ax1.set_title('Cross-Validation Fold Accuracy', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Fold', fontsize=12)
        ax1.set_ylabel('Accuracy', fontsize=12)
        ax1.set_xticks(fold_numbers)
        ax1.set_ylim(0.0, 1.0)
        ax1.grid(True, linestyle='--', alpha=0.5)
        ax1.axhline(y=self.cv_scores.mean(), color='r', linestyle='--', label=f'Mean: {self.cv_scores.mean():.4f}')
        ax1.legend()

        # Right subplot: Training accuracy curve
        ax2.plot(sample_counts, train_acc, marker='o', label='Train Accuracy', linewidth=2, markersize=8)
        ax2.plot(sample_counts, test_acc, marker='o', label='Test Accuracy', linewidth=2, markersize=8)
        ax2.set_title('Training Accuracy Curve', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Training Samples', fontsize=12)
        ax2.set_ylabel('Accuracy', fontsize=12)
        ax2.grid(True, linestyle='--', alpha=0.5)
        ax2.legend()
        ax2.set_ylim(0.0, 1.0)

        plt.tight_layout()
        plt.savefig('combined_training_curves.png', dpi=300, bbox_inches='tight')
        print("Combined training curves plot saved to combined_training_curves.png")
        plt.close()

        return sample_counts, train_acc, test_acc

    def evaluate(self, X, y, y_original):
        """Evaluate the classifier with detailed metrics"""
        print("\nEvaluating classifier on test set...")
        
        # Standardize features and apply PCA
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        
        # Make predictions
        predictions = self.classifier.predict(X_pca)
        
        # Calculate metrics
        accuracy = accuracy_score(y, predictions)
        f1_weighted = f1_score(y, predictions, average='weighted')
        f1_macro = f1_score(y, predictions, average='macro')
        
        print(f"Test accuracy: {accuracy:.4f}")
        print(f"Weighted F1-score: {f1_weighted:.4f}")
        print(f"Macro F1-score: {f1_macro:.4f}")
        
        # Print detailed classification report
        print("\nDetailed Classification Report:")
        print(classification_report(y, predictions, 
                                    target_names=[self.reverse_label_mapping[i] 
                                                 for i in sorted(self.reverse_label_mapping.keys())]))
        
        # Confusion matrix
        cm = confusion_matrix(y, predictions)
        return accuracy, cm, predictions
    
    def plot_confusion_matrix(self, cm):
        """Plot confusion matrix"""
        labels = sorted(self.reverse_label_mapping.keys())
        label_names = [self.reverse_label_mapping[i] for i in labels]
        
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                    xticklabels=label_names, yticklabels=label_names)
        plt.title('Confusion Matrix - SVD+SVM Spectrogram Classifier')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig('confusion_matrix.png', dpi=300, bbox_inches='tight')
        print("Confusion matrix saved to confusion_matrix.png")
        plt.show()
    
    def predict(self, X):
        """Make predictions on new data"""
        X_scaled = self.scaler.transform(X)
        X_pca = self.pca.transform(X_scaled)
        predictions = self.classifier.predict(X_pca)
        confidences = self.classifier.decision_function(X_pca)
        return predictions, confidences


def main():
    # Set up paths
    base_dir = r"c:\Users\scarl\Downloads\Instruments"
    spectrogram_dir = os.path.join(base_dir, "Train_submission", "Train_Spectrograms")
    metadata_file = os.path.join(base_dir, "Metadata_Train.csv")
    
    # Check if paths exist
    if not os.path.exists(spectrogram_dir):
        print(f"Error: Spectrogram directory not found: {spectrogram_dir}")
        return
    
    if not os.path.exists(metadata_file):
        print(f"Error: Metadata file not found: {metadata_file}")
        return
    
    # Initialize classifier with advanced parameters
    print("="*70)
    print("ADVANCED SVD-based Instrument Spectrogram Classification")
    print("="*70)
    print("\nAdvanced Optimizations:")
    print("  ✓ Image size: 200x200 (optimized for detail)")
    print("  ✓ SVD features: 120 (comprehensive feature extraction)")
    print("  ✓ Advanced feature engineering (6 feature groups = 74 features)")
    print("  ✓ PCA denoising (80 principal components)")
    print("  ✓ Tuned SVM (C=50, gamma=0.0005)")
    print("  ✓ Balanced class weights")
    print("="*70)
    
    classifier = SpectrogramSVDClassifier(image_size=(200, 200), n_components=120)
    
    # Load data
    print("\n[1/5] LOADING DATA")
    print("-" * 70)
    X, y_encoded, y_original, filenames = classifier.load_data(spectrogram_dir, metadata_file)
    
    if len(X) == 0:
        print("Error: No data loaded successfully")
        return
    
    print(f"\nData shape: {X.shape}")
    print(f"Number of classes: {len(classifier.label_mapping)}")
    print(f"Classes: {classifier.label_mapping}")
    print(f"\nClass distribution:")
    class_counts = pd.Series(y_original).value_counts()
    for cls, count in class_counts.items():
        percentage = (count / len(y_original)) * 100
        print(f"  {cls}: {count} samples ({percentage:.1f}%)")
    
    # Split data with stratification
    print("\n[2/5] SPLITTING DATA")
    print("-" * 70)
    print("Splitting data into train (80%) and test (20%) sets (stratified)...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=0.2, random_state=42, stratify=y_encoded
    )
    
    print(f"Training set size: {len(X_train)} samples")
    print(f"Test set size: {len(X_test)} samples")
    print(f"Feature dimension: {X_train.shape[1]} features")
    
    # Train classifier
    print("\n[3/5] TRAINING")
    print("-" * 70)
    train_accuracy = classifier.train(X_train, y_train)
    
    # Plot combined CV and training accuracy curves
    print("\n[3.5/5] TRAINING ANALYSIS")
    print("-" * 70)
    classifier.plot_cv_and_training_curves(X_train, y_train, X_test, y_test)

    # Evaluate on test set
    print("\n[4/5] EVALUATION")
    print("-" * 70)
    test_accuracy, cm, predictions = classifier.evaluate(X_test, y_test, None)
    
    # Plot confusion matrix
    print("\n[5/5] VISUALIZATION")
    print("-" * 70)
    classifier.plot_confusion_matrix(cm)
    
    # Summary
    print("\n" + "="*70)
    print("FINAL SUMMARY")
    print("="*70)
    print(f"Image size: {classifier.image_size}")
    print(f"Initial features: {X_train.shape[1]} (after SVD extraction)")
    print(f"After PCA reduction: {classifier.pca.n_components_}")
    print(f"\nCross-validation accuracy: {classifier.cv_scores.mean():.4f} (+/- {classifier.cv_scores.std():.4f})")
    print(f"Training accuracy: {train_accuracy:.4f}")
    print(f"Test accuracy: {test_accuracy:.4f}")
    
    # Calculate improvement metrics
    overfit_ratio = train_accuracy - test_accuracy
    print(f"\nOverfitting gap: {overfit_ratio:.4f} (lower is better)")
    print(f"Model: SVM with PCA + advanced feature engineering")
    print("="*70)
    
    # Test on sample images
    print("\nSample Predictions:")
    print("-" * 70)
    for i in range(min(15, len(X_test))):
        pred_label_idx = predictions[i]
        pred_label = classifier.reverse_label_mapping[pred_label_idx]
        actual_label = classifier.reverse_label_mapping[y_test[i]]
        match = "✓" if y_test[i] == pred_label_idx else "✗"
        print(f"{match} Sample {i+1}: Predicted={pred_label:12s} | Actual={actual_label:12s}")


if __name__ == "__main__":
    main()
