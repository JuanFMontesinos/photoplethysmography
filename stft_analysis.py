import torch
from torch import nn
import dataset
import matplotlib.pyplot as plt
import numpy as np
data = dataset.BaseDataset(False)
batch = [data[i] for i in range(10)]
ts, feats,gt = zip(*batch)
ts = torch.stack(ts)
feats = torch.stack(feats)



signal = ts[0]    
fs = 100.                    # sampling rate

# define a list of (n_fft, hop_length) pairs to try
param_list = [
    (128,  32),
    (256,  32),
    (256, 32),
    (512, 32),
    (512, 32),
    (1024, 24),
]


# prepare figure
n = len(param_list)
cols = 3
rows = int(np.ceil(n/cols))
fig, axes = plt.subplots(rows, cols, figsize=(15, 5*rows), sharex=False, sharey=False)
plt.figure(figsize=(10, 5))
plt.plot(np.linspace(0, 30, len(signal)), signal)
plt.title(f"PPG Highpass Sample  (fs={int(100)} Hz)")
plt.xlabel("Time (s)")
plt.ylabel("Amplitude")
plt.legend()
plt.tight_layout()
plt.savefig("ts_plot_filtered.png")
plt.close()

for idx, (n_fft, hop) in enumerate(param_list):
    # compute STFT with Hamming window
    window = torch.hamming_window(n_fft)        # length = n_fft
    spec = torch.stft(
        signal,
        n_fft=n_fft,
        hop_length=hop,
        window=window,
        return_complex=True
    )                                           # shape: [freq_bins, time_frames]
    mag = spec.abs().cpu().numpy()
    mag= mag[:126]
    print(f"n_fft={n_fft}, hop={hop}, shape={mag.shape}")
    # build axes
    freq_bins, time_frames = mag.shape
    freqs = np.linspace(0, fs/2, freq_bins)
    times = np.arange(time_frames) * hop / fs

    ax = axes.flatten()[idx]
    im = ax.pcolormesh(times, freqs, mag, shading='gouraud')
    ax.set_title(f'n_fft={n_fft}, hop={hop}')
    ax.set_xlabel('Time (s)')
    ax.set_ylabel('Frequency (Hz)')
    fig.colorbar(im, ax=ax, label='Magnitude')

# turn off any unused subplots
for j in range(idx+1, rows*cols):
    axes.flatten()[j].axis('off')

plt.tight_layout()
plt.savefig('stft_example.png')
