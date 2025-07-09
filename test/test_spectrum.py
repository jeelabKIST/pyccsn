from pyccsn import signal
import numpy as np
import matplotlib.pyplot as plt


def main():
    fs = 2000
    t, y = generate_test_signal(fs=fs)
    # plt.figure()
    # plt.plot(t, y, 'k')
    # plt.show()
    
    f, px = signal.compute_spectrum(y, fs=fs)
    
    plt.figure()
    plt.plot(f, px)
    plt.xlim([10, 50])
    plt.title('Power Spectrum')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Power (dB)')
    plt.show()
    

def generate_test_signal(fs=2000, f1=20, a1=1, f2=31, a2=0.5, s=0.1):
    t = np.arange(0, 10, 1/fs)
    y = a1 * np.sin(2*np.pi*f1*t) + a2*np.cos(2*np.pi*f2*t) + s*np.random.randn(len(t))
    return t, y


if __name__ == "__main__":
    main()