import wfdb
import matplotlib.pyplot as plt
import numpy as np
from scipy import signal as sig
from collections import Counter


# ==========================================
# 1. Load ECG Record
# ==========================================

record = wfdb.rdrecord("100")

print("Sampling frequency:", record.fs)
print("Number of samples:", record.sig_len)
print("Number of channels:", record.n_sig)

signal = record.p_signal

print("Signal shape:", signal.shape)

# Select the first ECG channel
ecg = np.asarray(signal[:, 0], dtype=float)


# ==========================================
# 2. Load ECG Annotations
# ==========================================

annotation = wfdb.rdann("100", "atr")

print("\n--- ECG Annotations ---")
print("Number of annotations:", len(annotation.sample))

print("First 20 annotation symbols:")
print(annotation.symbol[:20])


# Count all annotation symbols
annotation_counts = Counter(annotation.symbol)

print("\n--- Annotation Distribution ---")

for symbol, count in annotation_counts.items():
    print(symbol, ":", count)


# ==========================================
# 3. Select Beat Annotations
# ==========================================

# Beat classes that we currently want to analyze
beat_symbols = {"N", "A", "V"}

# Create a mask for beat annotations only
beat_mask = np.array([
    symbol in beat_symbols
    for symbol in annotation.symbol
])

# Get sample positions of the beats
beat_samples = annotation.sample[beat_mask]

# Get labels of the beats
beat_labels = np.array(annotation.symbol)[beat_mask]

print("\n--- Beat Annotations ---")
print("Number of beat annotations:", len(beat_samples))

print("Beat label distribution:")
print(Counter(beat_labels))


# ==========================================
# 4. Bandpass Filtering
# ==========================================

lowcut = 0.5
highcut = 40

b, a = sig.butter(
    3,
    [lowcut, highcut],
    btype="bandpass",
    fs=record.fs
)

filtered_ecg = sig.filtfilt(b, a, ecg)


# ==========================================
# 5. R-Peak Detection
# ==========================================

peaks, properties = sig.find_peaks(
    filtered_ecg,
    distance=int(0.5 * record.fs),
    prominence=0.5
)

print("\n--- R-Peak Detection ---")
print("Number of detected peaks:", len(peaks))


# ==========================================
# 6. RR Intervals and Heart Rate
# ==========================================

rr_intervals = np.diff(peaks) / record.fs
heart_rates = 60 / rr_intervals

print("\n--- Heart Rate ---")
print("Mean RR interval:", np.mean(rr_intervals), "seconds")
print("Mean heart rate:", np.mean(heart_rates), "BPM")


# ==========================================
# 7. RR Interval Statistics
# ==========================================

min_rr = np.min(rr_intervals)
max_rr = np.max(rr_intervals)
std_rr = np.std(rr_intervals)

print("\n--- RR Statistics ---")
print("Minimum RR interval:", min_rr, "seconds")
print("Maximum RR interval:", max_rr, "seconds")
print("Standard deviation of RR intervals:", std_rr, "seconds")


# ==========================================
# 8. HRV Analysis
# ==========================================

# SDNN
sdnn = np.std(rr_intervals, ddof=1)

# Differences between consecutive RR intervals
rr_diff = np.diff(rr_intervals)

# RMSSD
rmssd = np.sqrt(np.mean(rr_diff ** 2))

# Number of successive RR differences greater than 50 ms
nn50 = np.sum(np.abs(rr_diff) > 0.05)

# pNN50
pnn50 = (nn50 / len(rr_diff)) * 100

print("\n--- HRV Metrics ---")
print("SDNN:", sdnn * 1000, "ms")
print("RMSSD:", rmssd * 1000, "ms")
print("pNN50:", pnn50, "%")


# ==========================================
# 9. RR Interval Tachogram
# ==========================================

rr_time = peaks[1:] / record.fs

plt.figure(figsize=(12, 5))

plt.plot(
    rr_time,
    rr_intervals,
    ".-",
    label="RR Intervals"
)

plt.xlabel("Time (seconds)")
plt.ylabel("RR Interval (seconds)")
plt.title("RR Interval Tachogram")
plt.legend()
plt.grid()

plt.show()


# ==========================================
# 10. ECG with Detected R-Peaks
#     First 10 seconds
# ==========================================

seconds = 10

samples = int(seconds * record.fs)

time = np.arange(samples) / record.fs

# Select detected peaks inside first 10 seconds
first_10_sec = peaks < samples
peaks_10sec = peaks[first_10_sec]

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    filtered_ecg[:samples],
    label="Filtered ECG"
)

plt.plot(
    peaks_10sec / record.fs,
    filtered_ecg[peaks_10sec],
    "ro",
    markersize=5,
    label="Detected R-peaks"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("Detected R-Peaks - First 10 Seconds")
plt.legend()
plt.grid()

plt.show()


# ==========================================
# 11. Original vs Filtered ECG
#     First 10 seconds
# ==========================================

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    ecg[:samples],
    label="Original ECG"
)

plt.plot(
    time,
    filtered_ecg[:samples],
    label="Filtered ECG"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("Original vs Filtered ECG - First 10 Seconds")
plt.legend()
plt.grid()

plt.show()


# ==========================================
# 12. Reference Annotations on ECG
#     First 10 seconds
# ==========================================

# Select annotations inside first 10 seconds
annotation_10sec_mask = beat_samples < samples

beat_samples_10sec = beat_samples[annotation_10sec_mask]
beat_labels_10sec = beat_labels[annotation_10sec_mask]

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    filtered_ecg[:samples],
    label="Filtered ECG"
)

# Plot each annotation class separately
for label in ["N", "A", "V"]:

    class_mask = beat_labels_10sec == label

    class_samples = beat_samples_10sec[class_mask]

    if len(class_samples) > 0:

        plt.plot(
            class_samples / record.fs,
            filtered_ecg[class_samples],
            "o",
            markersize=6,
            label=f"Annotation {label}"
        )

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("Reference ECG Annotations - First 10 Seconds")
plt.legend()
plt.grid()

plt.show()


# ==========================================
# 13. Display One Example Beat from Each Class
# ==========================================

for label in ["N", "A", "V"]:

    # Find beats belonging to this class
    indices = np.where(beat_labels == label)[0]

    # Skip class if it does not exist
    if len(indices) == 0:
        continue

    # Select the first beat of this class
    idx = indices[0]

    center = beat_samples[idx]

    # 0.4 second window on each side
    window = int(0.4 * record.fs)

    start = max(0, center - window)
    end = min(len(filtered_ecg), center + window)

    beat_time = np.arange(start, end) / record.fs

    plt.figure(figsize=(8, 4))

    plt.plot(
        beat_time,
        filtered_ecg[start:end]
    )

    plt.axvline(
        center / record.fs,
        linestyle="--",
        label=f"Annotation {label}"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"Example ECG Beat - Class {label}")
    plt.legend()
    plt.grid()

    plt.show()

# ==========================================
# 14. Beat Segmentation
# ==========================================

# Window size around each reference annotation
window_before = int(0.2 * record.fs)
window_after = int(0.4 * record.fs)

# Store segmented beats
beats = []
labels = []

for sample, label in zip(beat_samples, beat_labels):

    # Define segment boundaries
    start = sample - window_before
    end = sample + window_after

    # Skip beats too close to the signal boundaries
    if start < 0 or end > len(filtered_ecg):
        continue

    # Extract ECG segment
    beat = filtered_ecg[start:end]

    # Store beat and label
    beats.append(beat)
    labels.append(label)


# Convert lists to NumPy arrays
beats = np.array(beats)
labels = np.array(labels)


# ==========================================
# 15. Segmented Dataset Information
# ==========================================

print("\n--- Beat Segmentation ---")

print("Number of segmented beats:", len(beats))
print("Beat segment shape:", beats.shape)

print("\nSegment length:", beats.shape[1], "samples")

print("\nSegmented beat distribution:")
print(Counter(labels))


# ==========================================
# 16. Plot Example Segmented Beats
# ==========================================

for label in ["N", "A", "V"]:

    # Find beats belonging to this class
    indices = np.where(labels == label)[0]

    # Skip if class does not exist
    if len(indices) == 0:
        continue

    # Select first segmented beat
    idx = indices[0]

    beat = beats[idx]

    beat_time = np.arange(len(beat)) / record.fs

    plt.figure(figsize=(8, 4))

    plt.plot(
        beat_time,
        beat
    )

    plt.axvline(
        window_before / record.fs,
        linestyle="--",
        label=f"Class {label}"
    )

    plt.xlabel("Time (seconds)")
    plt.ylabel("Amplitude (mV)")
    plt.title(f"Segmented ECG Beat - Class {label}")
    plt.legend()
    plt.grid()

    plt.show()

    # ==========================================
# 17. Save Segmented Beat Dataset
# ==========================================

np.savez(
    "record_100_beats.npz",
    beats=beats,
    labels=labels
)

print("\n--- Dataset Saved ---")
print("File: record_100_beats.npz")


# ==========================================
# 18. Build Dataset from Multiple Records
# ==========================================

# Records that will be used
record_names = [
    "100",
    "101",
    "103",
    "105",
    "106",
    "107",
    "109",
    "111",
    "112",
    "113"
]

# Lists for the combined dataset
all_beats = []
all_labels = []
all_record_ids = []


# Process each record
for record_name in record_names:

    print(f"\nProcessing record {record_name}...")

    # Load ECG record
    record_multi = wfdb.rdrecord(
        record_name,
        pn_dir="mitdb"
    )

    # Select first ECG channel
    ecg_multi = np.asarray(
        record_multi.p_signal[:, 0],
        dtype=float
    )

    # Load annotations
    annotation_multi = wfdb.rdann(
        record_name,
        "atr",
        pn_dir="mitdb"
    )

    # Keep only N, A, and V beats
    beat_mask_multi = np.array([
        symbol in {"N", "A", "V"}
        for symbol in annotation_multi.symbol
    ])

    beat_samples_multi = annotation_multi.sample[beat_mask_multi]

    beat_labels_multi = np.array(
        annotation_multi.symbol
    )[beat_mask_multi]


    # ======================================
    # Filtering
    # ======================================

    b_multi, a_multi = sig.butter(
        3,
        [0.5, 40],
        btype="bandpass",
        fs=record_multi.fs
    )

    filtered_multi = sig.filtfilt(
        b_multi,
        a_multi,
        ecg_multi
    )


    # ======================================
    # Beat Segmentation
    # ======================================

    window_before = int(0.2 * record_multi.fs)
    window_after = int(0.4 * record_multi.fs)

    record_beats = []
    record_labels = []

    for sample, label in zip(
        beat_samples_multi,
        beat_labels_multi
    ):

        start = sample - window_before
        end = sample + window_after

        # Skip boundary beats
        if start < 0 or end > len(filtered_multi):
            continue

        beat = filtered_multi[start:end]

        record_beats.append(beat)
        record_labels.append(label)


    # Convert to NumPy arrays
    record_beats = np.array(record_beats)
    record_labels = np.array(record_labels)


    # Store results
    all_beats.extend(record_beats)
    all_labels.extend(record_labels)

    all_record_ids.extend(
        [record_name] * len(record_beats)
    )


    # Print record information
    print(
        f"Record {record_name}: "
        f"{len(record_beats)} beats"
    )


# ==========================================
# Convert Combined Dataset to NumPy Arrays
# ==========================================

all_beats = np.array(all_beats)
all_labels = np.array(all_labels)
all_record_ids = np.array(all_record_ids)


# ==========================================
# Dataset Summary
# ==========================================

print("\n==========================================")
print("Combined Dataset Summary")
print("==========================================")

print(
    "Total number of beats:",
    len(all_beats)
)

print(
    "Beat matrix shape:",
    all_beats.shape
)

print(
    "Number of samples per beat:",
    all_beats.shape[1]
)

print(
    "\nClass distribution:"
)

print(Counter(all_labels))

print(
    "\nNumber of records:",
    len(np.unique(all_record_ids))
)

# ==========================================
# 19. Feature Extraction
# ==========================================

print("\n==========================================")
print("Feature Extraction")
print("==========================================")


# Lists to store extracted features
feature_list = []


# Extract features from every beat
for beat in all_beats:

    # Time-domain features
    mean_value = np.mean(beat)
    std_value = np.std(beat)
    min_value = np.min(beat)
    max_value = np.max(beat)
    range_value = max_value - min_value

    # RMS
    rms_value = np.sqrt(
        np.mean(beat ** 2)
    )

    # Signal energy
    energy_value = np.sum(beat ** 2)

    # Median
    median_value = np.median(beat)

    # Absolute mean
    abs_mean_value = np.mean(
        np.abs(beat)
    )

    # Peak-to-peak amplitude
    peak_to_peak = np.ptp(beat)


    # Store features
    feature_list.append([
        mean_value,
        std_value,
        min_value,
        max_value,
        range_value,
        rms_value,
        energy_value,
        median_value,
        abs_mean_value,
        peak_to_peak
    ])


# Convert features to NumPy array
features = np.array(feature_list)


# ==========================================
# 20. Feature Names
# ==========================================

feature_names = [
    "Mean",
    "Std",
    "Min",
    "Max",
    "Range",
    "RMS",
    "Energy",
    "Median",
    "Absolute_Mean",
    "Peak_to_Peak"
]


# ==========================================
# 21. Feature Dataset Information
# ==========================================

print("\nFeature matrix shape:")
print(features.shape)

print("\nNumber of features:")
print(features.shape[1])

print("\nFeature names:")
print(feature_names)


# ==========================================
# 22. Save Complete Dataset
# ==========================================

np.savez(
    "ecg_dataset.npz",
    beats=all_beats,
    labels=all_labels,
    record_ids=all_record_ids,
    features=features
)

print("\n==========================================")
print("Complete Dataset Saved")
print("==========================================")

print("File: ecg_dataset.npz")


# ==========================================
# 23. Train/Test Split by Record
# ==========================================

print("\n==========================================")
print("Train/Test Split")
print("==========================================")


# Get unique record IDs
unique_records = np.unique(all_record_ids)

print("Available records:")
print(unique_records)


# ==========================================
# Select test records manually
# ==========================================

test_records = np.array([
    "101",
    "105"
])


# All remaining records will be used for training
train_records = np.array([
    record_id
    for record_id in unique_records
    if record_id not in test_records
])


print("\nTraining records:")
print(train_records)

print("\nTesting records:")
print(test_records)


# ==========================================
# Create masks
# ==========================================

train_mask = np.isin(
    all_record_ids,
    train_records
)

test_mask = np.isin(
    all_record_ids,
    test_records
)


# ==========================================
# Create Training and Testing datasets
# ==========================================

X_train = features[train_mask]
X_test = features[test_mask]

y_train = all_labels[train_mask]
y_test = all_labels[test_mask]


# ==========================================
# Dataset Shapes
# ==========================================

print("\n--- Dataset Shapes ---")

print("X_train:", X_train.shape)
print("X_test :", X_test.shape)

print("y_train:", y_train.shape)
print("y_test :", y_test.shape)


# ==========================================
# Class Distribution
# ==========================================

print("\n--- Training Class Distribution ---")
print(Counter(y_train))

print("\n--- Testing Class Distribution ---")
print(Counter(y_test))


# ==========================================
# 24. Class Distribution per Record
# ==========================================

print("\n==========================================")
print("Class Distribution per Record")
print("==========================================")


for record_id in unique_records:

    record_mask = all_record_ids == record_id

    record_labels = all_labels[record_mask]

    print(
        f"\nRecord {record_id}:"
    )

    print(
        Counter(record_labels)
    )


# ==========================================
# 25. Feature Scaling
# ==========================================

from sklearn.preprocessing import StandardScaler


print("\n==========================================")
print("Feature Scaling")
print("==========================================")


# Create scaler
scaler = StandardScaler()


# Fit scaler ONLY on training data
X_train_scaled = scaler.fit_transform(X_train)


# Apply the same scaler to test data
X_test_scaled = scaler.transform(X_test)


print("Original training shape:")
print(X_train.shape)

print("\nScaled training shape:")
print(X_train_scaled.shape)

print("\nOriginal testing shape:")
print(X_test.shape)

print("\nScaled testing shape:")
print(X_test_scaled.shape)


# ==========================================
# 26. Random Forest Classifier
# ==========================================

from sklearn.ensemble import RandomForestClassifier


print("\n==========================================")
print("Random Forest Classifier")
print("==========================================")


# Create Random Forest model
model = RandomForestClassifier(
    n_estimators=200,
    random_state=42,
    class_weight="balanced"
)


# Train the model
model.fit(
    X_train_scaled,
    y_train
)


print("Model training completed.")


# ==========================================
# 27. Prediction
# ==========================================

y_pred = model.predict(X_test_scaled)


print("\nPrediction completed.")

print("Number of predictions:")
print(len(y_pred))


# ==========================================
# 28. Model Evaluation
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)


print("\n==========================================")
print("Model Evaluation")
print("==========================================")


# ==========================================
# Accuracy
# ==========================================

accuracy = accuracy_score(
    y_test,
    y_pred
)

print("\nAccuracy:")
print(accuracy)


# ==========================================
# Classification Report
# ==========================================

print("\nClassification Report:")

print(
    classification_report(
        y_test,
        y_pred,
        labels=["N", "A", "V"],
        target_names=[
            "Normal (N)",
            "Atrial Premature (A)",
            "Ventricular Premature (V)"
        ],
        zero_division=0
    )
)


# ==========================================
# Confusion Matrix
# ==========================================

cm = confusion_matrix(
    y_test,
    y_pred,
    labels=["N", "A", "V"]
)

print("\nConfusion Matrix:")
print(cm)

# ==========================================
# 29. Confusion Matrix Visualization
# ==========================================

import matplotlib.pyplot as plt

plt.figure(figsize=(6, 5))

plt.imshow(cm)

plt.title("Random Forest Confusion Matrix")
plt.xlabel("Predicted Label")
plt.ylabel("True Label")

plt.xticks(
    [0, 1, 2],
    ["N", "A", "V"]
)

plt.yticks(
    [0, 1, 2],
    ["N", "A", "V"]
)

for i in range(cm.shape[0]):
    for j in range(cm.shape[1]):
        plt.text(
            j,
            i,
            cm[i, j],
            ha="center",
            va="center"
        )

plt.colorbar()

plt.tight_layout()
plt.show()

# ==========================================
# 30. SVM Classifier
# ==========================================

from sklearn.svm import SVC

print("\n==========================================")
print("SVM Classifier")
print("==========================================")

svm_model = SVC(
    kernel="rbf",
    class_weight="balanced",
    random_state=42
)

print("\nTraining SVM...")

svm_model.fit(
    X_train_scaled,
    y_train
)

print("SVM training completed.")

svm_pred = svm_model.predict(
    X_test_scaled
)

print("\nSVM prediction completed.")

print("Number of predictions:")
print(len(svm_pred))


# ==========================================
# 31. SVM Model Evaluation
# ==========================================

from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix
)

print("\n==========================================")
print("SVM Model Evaluation")
print("==========================================")

svm_accuracy = accuracy_score(
    y_test,
    svm_pred
)

print("\nSVM Accuracy:")
print(svm_accuracy)

print("\nSVM Classification Report:")

print(
    classification_report(
        y_test,
        svm_pred,
        labels=["N", "A", "V"],
        target_names=[
            "Normal (N)",
            "Atrial Premature (A)",
            "Ventricular Premature (V)"
        ],
        zero_division=0
    )
)

svm_cm = confusion_matrix(
    y_test,
    svm_pred,
    labels=["N", "A", "V"]
)

print("\nSVM Confusion Matrix:")
print(svm_cm)