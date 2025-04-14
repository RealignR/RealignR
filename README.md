# 🧠 RealignR: A Lifelong Adaptive Optimization System

**RealignR** is a self-regulating machine learning training framework that fuses:
- 🧬 Physics-inspired conductance optimization (ARP)
- 🤖 GPT-based control steering
- 🔁 Lifelong learning via dataset switching with memory retention
- 🧠 CPR (Conduction Phase Reinforcement) for plateau recovery
- 📈 Real-time logging and metric-based introspection

## 🚀 System Overview

| Component            | Function                                                  |
|---------------------|-----------------------------------------------------------|
| **ARP Optimizer**   | Adaptive Resistance Principle — reinforcement-based updates via G₍ᵢⱼ₎ |
| **CPR**             | Detects optimization stalls and triggers adaptive resets |
| **GPT Controller**  | Dynamically adjusts alpha/mu using TensorBoard metrics and memory feedback |
| **Dataset Switcher**| Loads datasets based on `dataset_schedule.json` and recalls saved G₍ᵢⱼ₎ memory per domain |
| **Watcher Memory**  | Stores GPT responses and training history for contextual tuning |
| **TensorBoard Live**| Tracks accuracy, loss, G_mean, alpha, mu, CPR_trigger, etc. |

## 💾 Features

- **Multi-dataset stability**: Seamlessly switches between CIFAR100, SVHN, and CIFAR10 without loss of performance.
- **Self-tuning resistance**: Alpha and Mu are continuously adjusted by both internal logic and external GPT feedback.
- **Adaptive checkpoints**: Memory-aware conductance states saved per dataset.
- **CPR Recovery System**: Loss plateaus are auto-resolved via conductance resets and reactivation.
- **Infinity Mode**: Model trains indefinitely, smoothly adapting over time and tasks.

## 🧪 Results

- **CIFAR100**: 85%+ accuracy
- **SVHN**: 94.5%+ accuracy
- **CIFAR10**: 96%+ accuracy

Across all datasets, ARP retained performance through multiple switches while maintaining low variance and clean GradNorm.

## 📊 Metrics Tracked

- Accuracy / Loss
- G_mean (conductance average)
- alpha / mu (resistance memory tuning)
- CPR_trigger (stability intervention log)
- GradNorm (gradient pressure / instability signal)
- AccuracyVariance (performance consistency)

## 📂 Files Included

- `appr.py`: Main training loop with GPT + CPR + dataset switching
- `dataset_switcher_patch.py`: Modular G₍ᵢⱼ₎ memory system
- `watch_tensorboard_feedback_live.py`: GPT control feedback + TensorBoard integration
- `realignr_*.csv`: Live run logs, checkpoint metadata, and training scalars

## 🛠 Tech Stack

- Python 3.11
- PyTorch + AMP
- TensorBoard
- OpenAI GPT-4 API
- CUDA-enabled training (dual RTX 3090 Ti)

## 🌍 Use Cases

- Optimizer research / meta-learning
- Reinforcement-learning-like auto-tuners
- Continuous learning systems
- Online model adaptation for changing data streams

## 🔮 Future Directions

- Meta-optimizer: Fine-tune GPT based on RealignR's history
- Phase scheduler: Curriculum-based learning stages
- Live dashboard: Visual timeline + decision trails
- Human-in-the-loop optimizer coaching interface

---

Want to try it live or sponsor a buildout? [Message Ryan McKenna](mailto:you@example.com)



