ECG Signal Analysis and R-Peak Detection

A Python-based ECG signal processing project using the MIT-BIH Arrhythmia Database.

This project demonstrates an ECG analysis pipeline including signal visualization, filtering, R-peak detection, heart rate estimation, RR interval analysis, basic HRV metrics, and comparison with reference annotations.

Project Overview

Electrocardiography (ECG) is widely used for monitoring and analyzing cardiac electrical activity.

This project processes Record 100 from the MIT-BIH Arrhythmia Database and applies several signal-processing techniques to extract cardiac information.

Objectives

- Visualize the raw ECG signal.
- Apply band-pass filtering to the ECG signal.
- Detect R-peaks.
- Estimate average heart rate.
- Analyze RR intervals.
- Calculate basic HRV metrics.
- Evaluate R-peak detection against reference annotations.

Analysis Pipeline

1. Raw ECG Signal

The ECG recording is loaded using WFDB, and the first ECG channel is visualized over the first 10 seconds.

2. ECG Signal Filtering

A third-order Butterworth band-pass filter is applied with cutoff frequencies of 0.5 Hz and 40 Hz.

3. R-Peak Detection

R-peaks are detected from the filtered ECG signal using scipy.signal.find_peaks.

4. RR Interval and Heart Rate Analysis

RR intervals are calculated from consecutive detected R-peaks.

The average heart rate is estimated using:

Heart Rate = 60 / Mean RR Interval

5. Basic HRV and Detection Evaluation

Two basic heart rate variability metrics are calculated:

- SDNN - Standard Deviation of NN intervals
- RMSSD - Root Mean Square of Successive Differences

The detected R-peaks are also compared with the reference annotations provided by the MIT-BIH database using a 100 ms tolerance window.

Dataset

This project uses the MIT-BIH Arrhythmia Database, specifically Record 100.

The original database files are not included in this GitHub repository. They should be obtained separately before running the analysis.

Technologies

- Python
- WFDB
- NumPy
- SciPy
- Matplotlib

Installation

Install the required Python packages:

pip install -r requirements.txt

Usage

After obtaining the required MIT-BIH Record 100 files, run:

python src/ecg_signal_analysis_final.py

The script generates five result figures inside the results directory:

- results/step1.png
- results/step2.png
- results/step3.png
- results/step4.png
- results/step5.png

Project Structure

ecg-signal-analysis/

README.md
requirements.txt
.gitignore

src/
ecg_signal_analysis_final.py

results/
step1.png
step2.png
step3.png
step4.png
step5.png

Skills Demonstrated

- ECG signal processing
- Biomedical signal analysis
- Digital filtering
- R-peak detection
- Heart rate estimation
- RR interval analysis
- Basic HRV analysis
- Python scientific computing
- Data visualization
- Signal detection evaluation

Author

Hawraa Bahaa

Biomedical Engineering