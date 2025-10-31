import numpy as np
import matplotlib.pyplot as plt
from pyccsn.signal import compute_wavelet_spectrogram
# from wavelet_spectrum import compute_wavelet_spectrogram  # ← change to your actual module name

def main():
    fs = 2000
    t, y = generate_test_signal(fs=fs)

    # --- Compute wavelet spectrogram ---
    f, t_spec, pxx = compute_wavelet_spectrogram(
        y,
        fs=fs,
        frange=(1, 100),
        mode="power",
        baseline=None,
        baseline_mode="none",
        decim=10
    )

    # --- Plot ---
    plt.figure(figsize=(10, 4))
    # plt.pcolormesh(t_spec, f, Pdb, 
    # plt.pcolormesh(t_spec, f, 10 * np.log10(pxx + 1e-10), 
    plt.pcolormesh(t_spec, f, pxx,
                   shading="gouraud", cmap="jet", vmin=None, vmax=None)
    plt.title("Wavelet Spectrogram (Morlet, 6 cycles)")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 100)
    plt.xticks(np.arange(0, 11, 1))
    plt.colorbar(label="Power (dB)")
    plt.axvspan(3, 4, color="red", alpha=0.3, label="Expected 20 Hz Burst")
    plt.axvspan(3.5, 3.8, color="green", alpha=0.3, label="Expected 45 Hz Burst")
    plt.axvspan(4, 4.1, color="blue", alpha=0.3, label="Expected 85 Hz Burst")
    plt.legend()
    plt.tight_layout()
    plt.show()
    
    # Diagnostic snippet (pseudo)
    # t = np.arange(0, 10, 1/fs)
    # x = np.zeros_like(t); t0 = 3.0
    # x[int(t0*fs)] = 1.0   # impulse
    # f, tt, Y = compute_wavelet_spectrogram(x, fs, freqs=np.array([20.0]),
    #                                     n_cycles=6.0, mode="amplitude", pad="reflect")
    # # Peak time ≈ t0
    # print(tt[np.argmax(Y[0])])

def generate_test_signal(fs=2000, duration=10):
    """
    Generate a noisy signal containing a transient 20 Hz burst.
    """
    t = np.arange(0, duration, 1/fs)
    y = 0.2 * np.random.randn(len(t))  # baseline noise

    idx = (t >= 3) & (t < 4)
    y[idx] += 2.0 * np.sin(2 * np.pi * 20 * t[idx])  # 20 Hz oscillation during 3–4 s
    
    idx = (t >= 3.5) & (t < 3.8)
    y[idx] += 1 * np.sin(2 * np.pi * 45 * t[idx])  # 20 Hz oscillation during 3.5-3.8 s
    
    idx = (t >= 4) & (t < 4.1)
    y[idx] += 1 * np.sin(2 * np.pi * 85 * t[idx])  # 85 Hz oscillation during 4-4.1 s

    return t, y

if __name__ == "__main__":
    main()
