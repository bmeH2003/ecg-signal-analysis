import wfdb
import matplotlib.pyplot as plt
import numpy as np

# Load ECG record
record = wfdb.rdrecord("100")
# Extract the first ECG channel
ecg_signal = record.p_signal[:,0]
# Sampling frequency
fs = record.fs

print("Sampling frequency:", fs)
print("Number of samples:", len(ecg_signal))
print("Signal duration:", len(ecg_signal) / fs, "seconds")

# Plot the first 10 seconds
time = np.arange(len(ecg_signal)) / fs

plt.figure(figsize=(12, 4))

plt.plot(
    time,
    ecg_signal
)

plt.xlim(0, 10)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("Raw ECG Signal - Record 100")

plt.grid()

plt.tight_layout()
plt.savefig("results/step1.png", dpi=300, bbox_inches="tight")
plt.close()

from scipy import signal as sig

# Define filter limits
low_cutoff = 0.5
high_cutoff = 40.0

# Create Butterworth band-pass filter
b, a = sig.butter(
    3,
    [low_cutoff, high_cutoff],
    btype="bandpass",
    fs=fs
)

# Apply the filter
filtered_ecg = sig.filtfilt(
    b,
    a,
    ecg_signal
)

# Plot original and filtered ECG
plt.figure(figsize=(12, 6))

plt.plot(
    time,
    ecg_signal,
    label="Original ECG"
)

plt.plot(
    time,
    filtered_ecg,
    label="Filtered ECG"
)

plt.xlim(0, 10)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("Original vs Filtered ECG")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("results/step2.png", dpi=300, bbox_inches="tight")
plt.close()

peaks, properties = sig.find_peaks(
    filtered_ecg,
    distance=int(0.25 * fs),
    prominence=0.5
)

print("\nNumber of detected R-peaks:", len(peaks))

# Plot filtered ECG with detected R-peaks
plt.figure(figsize=(12, 5))

plt.plot(
    time,
    filtered_ecg,
    label="Filtered ECG"
)

plt.plot(
    time[peaks],
    filtered_ecg[peaks],
    "x",
    label="Detected R-peaks"
)

plt.xlim(0, 10)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("R-Peak Detection")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("results/step3.png", dpi=300, bbox_inches="tight")
plt.close()

rr_intervals = np.diff(peaks) / fs

# Calculate average RR interval
mean_rr = np.mean(rr_intervals)

# Calculate average heart rate
heart_rate = 60 / mean_rr

print("\n==========================================")
print("Heart Rate Analysis")
print("==========================================")

print("Mean RR interval:", mean_rr, "seconds")
print("Average Heart Rate:", heart_rate, "BPM")

# Plot RR intervals
plt.figure(figsize=(12, 4))

plt.plot(
    rr_intervals
)

plt.xlabel("Beat Number")
plt.ylabel("RR Interval (seconds)")
plt.title("RR Intervals")

plt.grid()

plt.tight_layout()
plt.savefig("results/step4.png", dpi=300, bbox_inches="tight")
plt.close()

sdnn = np.std(rr_intervals, ddof=1)

# Calculate successive RR differences
rr_diff = np.diff(rr_intervals)

# Calculate RMSSD
rmssd = np.sqrt(np.mean(rr_diff ** 2))

print("\n==========================================")
print("Basic HRV Analysis")
print("==========================================")

print("SDNN:", sdnn, "seconds")
print("SDNN:", sdnn * 1000, "ms")

print("RMSSD:", rmssd, "seconds")
print("RMSSD:", rmssd * 1000, "ms")

# Load reference annotations
annotation = wfdb.rdann("100", "atr")

# Get annotation sample locations
annotation_samples = annotation.sample

# Convert annotation locations to seconds
annotation_times = annotation_samples / fs

print("\n==========================================")
print("Annotation Analysis")
print("==========================================")

print("Number of reference annotations:", len(annotation_samples))

# Plot ECG with detected R-peaks
plt.figure(figsize=(12, 5))

plt.plot(
    time,
    filtered_ecg,
    label="Filtered ECG"
)

plt.plot(
    time[peaks],
    filtered_ecg[peaks],
    "x",
    label="Detected R-peaks"
)

# Show only the first 10 seconds
plt.xlim(0, 10)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.title("Detected R-Peaks vs ECG Signal")

plt.legend()
plt.grid()

plt.tight_layout()
plt.savefig("results/step5.png", dpi=300, bbox_inches="tight")
plt.close()


# Convert reference annotations to NumPy array
reference_peaks = np.array(annotation_samples)

# Allowed difference: 100 milliseconds
tolerance = int(0.1 * fs)

# Count correctly detected peaks
correct_peaks = 0

for reference_peak in reference_peaks:

    difference = np.abs(peaks - reference_peak)

    if np.min(difference) <= tolerance:
        correct_peaks += 1

# Calculate detection rate
detection_rate = (
    correct_peaks / len(reference_peaks)
) * 100

print("\n==========================================")
print("R-Peak Detection Evaluation")
print("==========================================")

print("Reference peaks:", len(reference_peaks))
print("Detected peaks:", len(peaks))
print("Correctly detected peaks:", correct_peaks)
print("Detection rate:", round(detection_rate, 2), "%")