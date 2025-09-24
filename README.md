# PGD Attack Demo - Educational Adversarial Machine Learning

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/license-MIT-green.svg)](LICENSE)

A comprehensive educational implementation of **PGD (Projected Gradient Descent)** adversarial attacks for understanding adversarial machine learning and defensive security research.

## 🎯 Overview

This project provides a complete implementation of PGD attacks with detailed analysis tools, visualization capabilities, and educational notebooks. It's designed for students, researchers, and security professionals to understand how adversarial examples work and how to defend against them.

### Key Features

- ✅ **Complete PGD Implementation** with L∞ and L2 norm constraints
- ✅ **FGSM Variants** (Basic, Iterative, Momentum) for comparison
- ✅ **Detailed Class Probability Analysis** showing confidence changes
- ✅ **Comprehensive Visualizations** with perturbation heatmaps
- ✅ **Multiple Model Support** (ResNet, VGG, DenseNet, etc.)
- ✅ **Educational Jupyter Notebooks** with step-by-step tutorials
- ✅ **Extensive Testing Suite** with validation checks
- ✅ **Logging and Metrics** for research analysis

## 📋 Prerequisites

- **Python 3.11+ (recommended)** or Python 3.10+
- **Anaconda or Miniconda** (recommended)
- **4GB+ free disk space** (for models and datasets)
- **Optional: NVIDIA GPU** with CUDA support for faster execution

### 🐍 Supported Python Versions:
- **Python 3.12** - Latest stable (use `environment-py312.yml`)
- **Python 3.11** - Recommended balance of features and stability (use `environment.yml`)
- **Python 3.10** - Minimum supported version

## 🚀 Quick Start

### Step 1: Clone and Setup Environment

```bash
# Clone or navigate to the project directory
cd pgd-attack

# Choose your Python version:

# For Python 3.11 (recommended)
conda env create -f environment.yml
conda activate pgd-demo

# OR for Python 3.12 (latest)
conda env create -f environment-py312.yml
conda activate pgd-demo-py312

# Verify installation
python -c "import torch; import torchvision; print('✅ Setup complete!')"
python --version  # Check Python version
```

### Step 2: Run Basic Tests

```bash
# Run unit tests
python tests/test_pgd.py

# Run detailed ResNet18 analysis with class probabilities
python tests/test_resnet18_detailed.py
```

### Step 3: Explore Notebooks

```bash
# Start Jupyter Lab
jupyter lab

# Or use classic notebook
jupyter notebook
```

Open the notebooks in order:
1. `01_pgd_introduction.ipynb` - Theory and concepts
2. `02_pgd_implementation.ipynb` - Step-by-step implementation
3. `03_pgd_experiments.ipynb` - Parameter analysis
4. `04_pgd_analysis.ipynb` - Advanced analysis

## 📁 Project Structure

```
pgd-attack/
│
├── environment.yml          # Conda environment specification
├── requirements.txt         # pip requirements
├── README.md               # This file
│
├── src/                    # Core implementation
│   ├── attacks/
│   │   ├── pgd.py         # PGD attack implementation
│   │   └── fgsm.py        # FGSM variants
│   ├── models/
│   │   └── load_models.py # Model loading utilities
│   └── utils/
│       └── visualization.py # Visualization tools
│
├── notebooks/              # Educational notebooks
│   ├── 01_pgd_introduction.ipynb
│   ├── 02_pgd_implementation.ipynb
│   ├── 03_pgd_experiments.ipynb
│   └── 04_pgd_analysis.ipynb
│
├── tests/                  # Test files
│   ├── test_pgd.py        # Unit tests
│   ├── test_resnet18.py   # Basic ResNet18 tests
│   └── test_resnet18_detailed.py # Detailed analysis
│
├── scripts/                # Command-line tools
│   ├── run_pgd_attack.py  # Main attack script
│   └── batch_experiments.py # Batch processing
│
├── data/                   # Datasets (auto-downloaded)
├── results/                # Output files
│   ├── figures/           # Visualizations
│   └── logs/              # Log files and metrics
│
└── models/                 # Model checkpoints
```

## 🔧 Detailed Setup Instructions

### Option A: Automated Setup with Conda

```bash
# Create environment
conda env create -f environment.yml
conda activate pgd-demo

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}')"
```

### Option B: Manual Setup

```bash
# Create environment manually (choose Python version)
conda create -n pgd-demo python=3.11  # or python=3.12
conda activate pgd-demo

# Install PyTorch (CPU version)
conda install pytorch torchvision cpuonly -c pytorch

# Or GPU version (if CUDA available)
conda install pytorch torchvision pytorch-cuda=12.1 -c pytorch -c nvidia

# Install other dependencies
conda install matplotlib numpy jupyter ipykernel tqdm pillow scikit-learn -c conda-forge
pip install ipywidgets
```

### Step 3: Verify Installation

```bash
python tests/test_pgd.py
```

Expected output:
```
Running PGD attack tests...
✓ Basic PGD attack test passed
✓ Targeted PGD attack test passed
✓ PGD with restarts test passed
✓ Attack evaluation test passed
✓ L2 norm constraint test passed
✓ FGSM attack test passed

✅ All tests passed successfully!
```

## 🎯 Usage Examples

### Basic PGD Attack

```python
from src.attacks.pgd import pgd_attack
from src.models.load_models import load_pretrained_model

# Load model
model, info = load_pretrained_model('resnet18', dataset='imagenet')

# Run attack
adv_images, perturbations = pgd_attack(
    model=model,
    images=your_images,
    labels=your_labels,
    epsilon=0.03,
    alpha=0.001,
    iterations=40
)
```

### Command-Line Usage

```bash
# Single image attack
python scripts/run_pgd_attack.py \
    --model resnet18 \
    --image path/to/image.jpg \
    --epsilon 0.03 \
    --iterations 40 \
    --save-results

# Batch experiments
python scripts/batch_experiments.py \
    --models resnet18 vgg16 \
    --epsilons 0.01 0.03 0.05 \
    --num-images 100
```

## 📊 Test Results Analysis

### Latest Test Results (ResNet18 on CIFAR-10)

**Attack Success Rate: 80% (4/5 samples)**

#### Sample Analysis:
```
Sample 1 (CIFAR-10: dog):
  Original Top-1: class_273 (12.71% confidence)
  Adversarial Top-1: class_310 (100.00% confidence)
  ✅ ATTACK SUCCESS: Prediction changed!
  Confidence change: +87.29%

Sample 2 (CIFAR-10: dog):
  Original Top-1: class_80 (31.07% confidence)
  Adversarial Top-1: class_80 (100.00% confidence)
  ❌ Attack failed: Prediction unchanged
  Confidence change: +68.93%
```

#### Technical Validation:
- ✅ **Epsilon Constraint**: Max L∞ = 0.030000 (exactly at ε=0.03)
- ✅ **Image Range**: All adversarial examples in [0, 1]
- ✅ **Mean L2 Perturbation**: 6.87
- ✅ **Shape Preservation**: All tensors correct dimensions

### Log Files and Detailed Results

All test runs generate detailed logs in `results/logs/`:
- **Detailed logs**: `pgd_test_YYYYMMDD_HHMMSS.log`
- **JSON results**: `detailed_results_YYYYMMDD_HHMMSS.json`
- **Visualizations**: `results/figures/detailed_analysis_*.png`

## 🎓 Educational Notebooks

### 1. Introduction to PGD (`01_pgd_introduction.ipynb`)
- Mathematical formulation with LaTeX
- Comparison with FGSM
- Epsilon ball visualization
- Interactive parameter exploration

### 2. Implementation Walkthrough (`02_pgd_implementation.ipynb`)
- Step-by-step PGD from scratch
- Detailed code explanations
- Visualization of iterative process
- Constraint validation

### 3. Experiments and Analysis (`03_pgd_experiments.ipynb`)
- Parameter sensitivity analysis
- Model robustness comparison
- Random restarts experiment
- Success rate measurements

### 4. Advanced Analysis (`04_pgd_analysis.ipynb`)
- Loss landscape visualization
- Perturbation distribution analysis
- Transferability experiments
- Defense mechanism testing

## 🛡️ Security and Ethics

This implementation is designed for **defensive security education** and research:

- ✅ **Educational Purpose**: Understanding adversarial ML for defense
- ✅ **Defensive Research**: Developing robust models and defenses
- ✅ **Security Analysis**: Evaluating model vulnerabilities
- ✅ **Ethical Guidelines**: Following responsible disclosure practices

**⚠️ Important**: This code should only be used for:
- Educational purposes in adversarial ML courses
- Defensive security research
- Robustness evaluation of your own models
- Academic research with proper ethics approval

## 🔬 Advanced Features

### Supported Models
- **ResNet Family**: ResNet-18, 34, 50, 101, 152
- **VGG Family**: VGG-11, 13, 16, 19 (with/without BatchNorm)
- **DenseNet Family**: DenseNet-121, 169, 201, 161
- **Others**: AlexNet, MobileNet, EfficientNet

### Attack Variants
- **PGD**: Basic and with random restarts
- **FGSM**: Basic, Iterative, Momentum variants
- **Constraints**: L∞ and L2 norm projections
- **Targeting**: Both targeted and untargeted attacks

### Analysis Tools
- Class probability tracking
- Confidence change analysis
- Entropy measurements
- Perturbation visualization
- Attack transferability testing

## 🚨 Troubleshooting

### Common Issues

#### 1. CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# Install CPU-only version if needed
conda install pytorch torchvision cpuonly -c pytorch
```

#### 2. Memory Issues
```bash
# Reduce batch size in scripts or notebooks
# Monitor memory usage
htop  # or Activity Monitor on macOS
```

#### 3. Import Errors
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or in Python/notebook
import sys
sys.path.append('.')
```

#### 4. Download Issues
```bash
# Clear cache and retry
rm -rf ~/.cache/torch/
python tests/test_resnet18.py
```

### Getting Help

1. Check the [troubleshooting section](#troubleshooting) above
2. Review the detailed logs in `results/logs/`
3. Run tests with verbose output: `python -v tests/test_pgd.py`
4. Open an issue with:
   - Error message
   - Environment info (`conda list`)
   - Steps to reproduce

## 📈 Performance Benchmarks

### Test Environment
- **OS**: macOS 14.6.0 / Linux / Windows
- **Python**: 3.11+ (recommended), 3.10+ (supported)
- **PyTorch**: 2.1.0+ (latest stable)
- **Device**: CPU and GPU supported

### 🚀 Performance with Modern Python:
- **Python 3.11**: ~25% faster than Python 3.9 for numerical computations
- **Python 3.12**: ~15% faster than Python 3.11, latest features
- **PyTorch 2.1+**: Improved performance and compatibility

### Attack Performance
- **PGD (40 iterations)**: ~35 seconds for 5 images
- **FGSM**: <1 second for 5 images
- **Memory Usage**: ~2GB RAM for ResNet18
- **Success Rate**: 60-90% depending on epsilon and model

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Add tests for new functionality
4. Update documentation
5. Submit a pull request

### Development Setup
```bash
git clone <your-fork>
cd pgd-attack
conda env create -f environment.yml
conda activate pgd-demo
pytest tests/  # Run all tests
```

## 📚 References and Further Reading

### Core Papers
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) (Madry et al., 2017)
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (Goodfellow et al., 2014)

### Additional Resources
- [Adversarial ML Reading List](https://nicholas.carlini.com/writing/2018/adversarial-machine-learning-reading-list.html)
- [CleverHans Library](https://github.com/cleverhans-lab/cleverhans)
- [Adversarial Robustness Toolbox](https://github.com/Trusted-AI/adversarial-robustness-toolbox)

## 📄 Citation

If you use this code for research, please cite:

```bibtex
@misc{pgd-attack-demo,
  title={PGD Attack Demo: Educational Adversarial Machine Learning},
  author={Your Name},
  year={2024},
  publisher={GitHub},
  url={https://github.com/your-username/pgd-attack}
}
```

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

## 🏁 Quick Validation

Run this to verify everything works:

```bash
conda activate pgd-demo
python tests/test_pgd.py
python tests/test_resnet18_detailed.py
```

Expected final output:
```
🎉 ALL TESTS COMPLETED SUCCESSFULLY!
- PGD attack successfully implemented
- Epsilon constraints properly enforced  
- Attack success rate: 80.0%
- All safety checks passed
✅ ResNet18 testing completed successfully!
```

**Happy Learning! 🚀**