from pyccsn import signal
import numpy as np
import matplotlib.pyplot as plt

def main():
    fs = 2000
    t, y = generate_test_signal(fs=fs)

    f, t_spec, pxx = signal.compute_spectrogram(y, fs=fs, t=t, wbin_t=1, mbin_t=0.1)

    plt.figure(figsize=(10, 4))
    plt.pcolormesh(t_spec, f, 10 * np.log10(pxx + 1e-10), shading='gouraud')
    plt.title("Spectrogram")
    plt.xlabel("Time (s)")
    plt.ylabel("Frequency (Hz)")
    plt.ylim(0, 100)
    plt.colorbar(label="Power (dB)")
    plt.axvspan(3, 4, color='red', alpha=0.3, label="Expected 20 Hz Burst")
    plt.legend()
    plt.tight_layout()
    plt.show()

def generate_test_signal(fs=2000, duration=10):
    t = np.arange(0, duration, 1/fs)
    y = 0.2 * np.random.randn(len(t))  # baseline noise

    idx = (t >= 3) & (t < 4)
    y[idx] += 2.0 * np.sin(2 * np.pi * 20 * t[idx])
    
    return t, y

if __name__ == "__main__":
    main()