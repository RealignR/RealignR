# RealignR

**A Self-Realigning Optimizer Core for Deep Learning**  
*Powered by GRAIL: Gradient Realignment for Adaptive Infinite Learning*

---

## 🚀 What is RealignR?

**RealignR** is the first optimizer that knows when it's off — and knows how to realign.

It doesn’t just train. It observes. It remembers. It corrects.
RealignR is designed to recover from collapse without restarting training.

---

## 🧠 Key Features

| Feature | Description |
|---------|-------------|
| **Observer Mode** | Captures stable resistance memory (`G_ij`) during early training (epoch 25) before collapse occurs |
| **Slope Monitoring** | Detects failure by tracking loss descent rate (slope) |
| **G Drift Detection** | Measures how far current resistance memory has drifted from control benchmarks |
| **CPR Triggering** | Fires memory reset only when slope + drift both signal misalignment |
| **Memory Recovery** | Reinitializes G with noise + fresh gradients, centered on prior alignment |

---

## 📦 Installation

Coming soon as a pip package.

For now:
```bash
git clone https://github.com/yourname/realignr.git
cd realignr
pip install -e .
```

---

## 🧪 Quick Start (CIFAR100)

```bash
python arp_resume_CPRv7_3_dualsensor.py
```

---

## 🔁 Why This Matters

Traditional optimizers collapse and require manual reset or restarts. RealignR:
- Stores memory of good learning states
- Detects when it's gone off track
- Recovers without forgetting

This enables **infinite training**, **continual learning**, and **autonomous error correction.**

---

## 📐 Powered by GRAIL

> **Gradient Realignment for Adaptive Infinite Learning**

GRAIL is the engine behind RealignR. It combines gradient observation, adaptive memory, and self-correction into a continuous system.

---

## 📈 Coming Soon

- TensorBoard tracking for G drift
- Visualizer for CPR trigger events
- Multi-dataset support (ImageNet, text models)
- Layer-specific realignment logic

---




---

## License
MIT

