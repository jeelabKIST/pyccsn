from pyccsn import signal
import numpy as np
import matplotlib.pyplot as plt


def main():
    fs = 2000
    t, x, y = generate_test_signal(fs=fs, tlag=0.2)
    tlag, cc = signal.get_correlation(x, y, srate=2000, max_lag=20, norm=True)
    
    plt.figure()
    plt.plot(tlag, cc)
    plt.xlim([-2, 2])
    plt.xlabel('Time Lag (s)')
    plt.ylabel('Cross-Correlation')
    plt.title('Cross-Correlation between x and y')
    plt.show()


def generate_test_signal(fs=2000, tlag=1):
    t = np.arange(0, 10, 1/fs)
    x = np.random.randn(len(t))*2
    # x = np.sin(2 * np.pi * 5 * t) + 0.5 * np.random.randn(len(t))
    nlag = int(tlag * fs)
    y = np.roll(x, nlag)
    
    return t, x, y


if __name__ == "__main__":
    main()
    
    