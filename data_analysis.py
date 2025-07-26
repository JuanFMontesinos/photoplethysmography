import polars
import os
import numpy as np
import shutil
import matplotlib.pyplot as plt
from scipy.signal import butter, filtfilt

import const as C

train_data = polars.read_csv(C.TRAIN_PATH) # N x 3005
labels = polars.read_csv(C.LABELS_PATH) # N x 1

labels_arr = labels.to_numpy()[:, 0] # N
log_labels_arr = labels_arr + np.abs(labels_arr.min()) + 1
log_labels_arr = np.log(log_labels_arr)
ts_arr = train_data.to_numpy()[:, :-5] # N x 3000
feats_df = train_data[C.feats_columns] # N x 5


def highpass_filter_ppg(ppg_signal, fs=C.SR, cutoff=0.5, order=4):
    """
    High‑pass filter for PPG at a fixed sampling rate of 100 Hz.
    """
    nyq = 0.5 * fs
    wn = cutoff / nyq
    b, a = butter(order, wn, btype="high")
    return filtfilt(b, a, ppg_signal)


ts_arr_filtered = np.stack(tuple(map(lambda x: highpass_filter_ppg(x, fs=C.SR), ts_arr)))

print("Feats describe")
print(feats_df.describe())
# ts decription max, min, mean, std, median etc...
print('Standard array:')
print(
    f" Max: {ts_arr.max()}, Min: {ts_arr.min()}, Mean: {ts_arr.mean()}, Std: {ts_arr.std()}"
)
print('Filtered array:')
print(
    f" Max: {ts_arr_filtered.max()}, Min: {ts_arr_filtered.min()}, Mean: {ts_arr_filtered.mean()}, Std: {ts_arr_filtered.std()}"
)
print("Labels describe")
print(labels.describe())

if os.path.exists("plots"):
    shutil.rmtree("plots")
os.makedirs("plots", exist_ok=True)

plt.hist(ts_arr.flatten(), bins=100)
plt.title("Histogram of Time Series Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig("plots/data_hist.png")

plt.figure()
plt.hist(ts_arr_filtered.flatten(), bins=100)
plt.title("Histogram of The filtered Time Series Data")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig("plots/filtered_data_hist.png")

plt.figure()
plt.hist(labels_arr.flatten(), bins=100)
plt.title("Histogram of labels")
plt.xlabel("Value")
plt.ylabel("Frequency")
plt.savefig("plots/labels_hist.png")

# Loghist
plt.figure()
plt.hist(log_labels_arr.flatten(), bins=100)
plt.title("Log Histogram of Time Series Data")
plt.xlabel("Log(Value + Min)")
plt.ylabel("Frequency")
plt.savefig("plots/labels_log_hist.png")
plt.figure()

x_axis = np.linspace(0, 30, ts_arr.shape[1], ts_arr.shape[1])
for i in np.random.randint(0, ts_arr.shape[0], 10):
    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, ts_arr[i])
    plt.title(f"Time Series Plot for Sample {i}")
    plt.xlabel("s")
    plt.ylabel("Value")
    plt.savefig(f"plots/ts_plot_{i}.png")
    plt.close()
    raw = ts_arr[i]
    filt = highpass_filter_ppg(raw, fs=C.SR)

    plt.figure(figsize=(10, 5))
    plt.plot(x_axis, filt)
    plt.title(f"PPG Highpass Sample {i} (fs={int(C.SR)} Hz)")
    plt.xlabel("Time (s)")
    plt.ylabel("Amplitude")
    plt.legend()
    plt.tight_layout()
    plt.savefig(f"plots/ts_plot_{i}_filtered.png")
    plt.close()


print()
