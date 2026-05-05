import pandas as pd
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

# Load metadata
df = pd.read_csv('Metadata_Train.csv')

# Get some drum and violin samples
drum_samples = df[df['Class'] == 'Sound_Drum'].head(5)['FileName'].tolist()
violin_samples = df[df['Class'] == 'Sound_Violin'].head(5)['FileName'].tolist()

print('Examining raw spectrogram characteristics...')

# Analyze raw pixel distributions
drum_raw_stats = []
violin_raw_stats = []

for filename in drum_samples:
    image_path = f'Train_submission/Train_Spectrograms/{filename.replace(".wav", ".png")}'
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32)

        # Raw statistics
        stats = {
            'mean': np.mean(img_array),
            'std': np.std(img_array),
            'max': np.max(img_array),
            'min': np.min(img_array),
            'median': np.median(img_array),
            'q25': np.percentile(img_array, 25),
            'q75': np.percentile(img_array, 75),
            'q95': np.percentile(img_array, 95),
            'shape': img_array.shape,
            'nonzero_ratio': np.count_nonzero(img_array) / img_array.size
        }
        drum_raw_stats.append(stats)
    except Exception as e:
        print(f'Error processing drum {filename}: {e}')

for filename in violin_samples:
    image_path = f'Train_submission/Train_Spectrograms/{filename.replace(".wav", ".png")}'
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32)

        # Raw statistics
        stats = {
            'mean': np.mean(img_array),
            'std': np.std(img_array),
            'max': np.max(img_array),
            'min': np.min(img_array),
            'median': np.median(img_array),
            'q25': np.percentile(img_array, 25),
            'q75': np.percentile(img_array, 75),
            'q95': np.percentile(img_array, 95),
            'shape': img_array.shape,
            'nonzero_ratio': np.count_nonzero(img_array) / img_array.size
        }
        violin_raw_stats.append(stats)
    except Exception as e:
        print(f'Error processing violin {filename}: {e}')

# Compare statistics
print('\n=== RAW SPECTROGRAM STATISTICS ===')
print(f'{"Statistic":<15} {"Drum_Mean":<10} {"Violin_Mean":<12} {"Difference":<10}')
print('-' * 60)

drum_avg = {}
violin_avg = {}

for key in ['mean', 'std', 'max', 'min', 'median', 'q25', 'q75', 'q95', 'nonzero_ratio']:
    drum_values = [s[key] for s in drum_raw_stats]
    violin_values = [s[key] for s in violin_raw_stats]

    drum_avg[key] = np.mean(drum_values)
    violin_avg[key] = np.mean(violin_values)
    diff = abs(drum_avg[key] - violin_avg[key])

    print(f'{key:<15} {drum_avg[key]:<10.2f} {violin_avg[key]:<12.2f} {diff:<10.2f}')

print(f'\nDrum image shapes: {[s["shape"] for s in drum_raw_stats]}')
print(f'Violin image shapes: {[s["shape"] for s in violin_raw_stats]}')

# Check if images are identical
print('\n=== CHECKING FOR IDENTICAL IMAGES ===')
drum_arrays = []
violin_arrays = []

for filename in drum_samples[:2]:  # Just check first 2
    image_path = f'Train_submission/Train_Spectrograms/{filename.replace(".wav", ".png")}'
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        drum_arrays.append(img_array)
    except:
        pass

for filename in violin_samples[:2]:  # Just check first 2
    image_path = f'Train_submission/Train_Spectrograms/{filename.replace(".wav", ".png")}'
    try:
        img = Image.open(image_path).convert('L')
        img_array = np.array(img, dtype=np.float32)
        violin_arrays.append(img_array)
    except:
        pass

if len(drum_arrays) >= 2 and len(violin_arrays) >= 2:
    # Compare drum to drum
    drum_diff = np.mean(np.abs(drum_arrays[0] - drum_arrays[1]))
    print(f'Drum-to-drum difference: {drum_diff:.2f}')

    # Compare violin to violin
    violin_diff = np.mean(np.abs(violin_arrays[0] - violin_arrays[1]))
    print(f'Violin-to-violin difference: {violin_diff:.2f}')

    # Compare drum to violin
    cross_diff = np.mean(np.abs(drum_arrays[0] - violin_arrays[0]))
    print(f'Drum-to-violin difference: {cross_diff:.2f}')

    print(f'\nRatio (within-class / cross-class):')
    print(f'  Drums: {drum_diff/cross_diff:.3f}')
    print(f'  Violins: {violin_diff/cross_diff:.3f}')

print('\n=== CONCLUSION ===')
if abs(drum_avg['mean'] - violin_avg['mean']) < 1.0 and abs(drum_avg['std'] - violin_avg['std']) < 1.0:
    print('Spectrograms appear nearly identical between drums and violins!')
    print('This explains why the classifier cannot distinguish them.')
else:
    print('Spectrograms show some differences - the issue may be in feature extraction.')