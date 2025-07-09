# pyccsn

**pyccsn** (Python Computational Cognitive Systems Neuroscience) is a modular Python package for analyzing neural signals.  

---

## 🚀 Installation

Install the package in editable (development) mode:

```bash
pip install -e .
```


## Example Usage
```python
from pyccsn import signal
import numpy as np

fs = 2000
t = np.arange(0, 10, 1/fs)
y = np.sin(2 * np.pi * 20 * t) + 0.1 * np.random.randn(len(t))

f, px = signal.compute_spectrum(y, fs=fs)
```