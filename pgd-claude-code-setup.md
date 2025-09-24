# PGD Attack Demo - Complete Setup Guide with Claude Code

## 📋 Prerequisites

- Anaconda or Miniconda installed
- Claude Code CLI tool
- At least 4GB of free disk space (for PyTorch and model weights)
- (Optional) NVIDIA GPU with CUDA support for faster execution

## 🚀 Quick Start

```bash
# Clone or create project directory
mkdir pgd-attack-demo
cd pgd-attack-demo

# Run the automated setup with Claude Code
claude-code "Set up a conda environment for PGD adversarial attack demo with PyTorch and create the notebook structure"
```

## 📦 Step 1: Create Conda Environment

### Option A: Using Claude Code to Generate Environment File

```bash
claude-code "Create a conda environment.yml file for a PyTorch project with GPU support that includes torchvision, matplotlib, numpy, and jupyter"
```

### Option B: Manual Environment Setup

Create `environment.yml`:

```yaml
name: pgd-demo
channels:
  - pytorch
  - conda-forge
  - defaults
dependencies:
  - python=3.9
  - pytorch=2.0.1
  - torchvision=0.15.2
  - cudatoolkit=11.8  # Skip if CPU-only
  - matplotlib=3.7.1
  - numpy=1.24.3
  - jupyter=1.0.0
  - ipykernel=6.23.1
  - tqdm=4.65.0
  - pillow=9.5.0
  - scikit-learn=1.2.2
  - pip
  - pip:
    - ipywidgets==8.0.6
```

### Create and Activate Environment

```bash
# Create environment from file
conda env create -f environment.yml

# Activate the environment
conda activate pgd-demo

# Verify PyTorch installation
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

## 📁 Step 2: Project Structure Setup

### Using Claude Code:

```bash
claude-code "Create a project structure for PGD attack demo with directories for notebooks, scripts, data, results, and models"
```

### Expected Structure:

```
pgd-attack-demo/
│
├── environment.yml
├── README.md
├── requirements.txt
│
├── notebooks/
│   ├── 01_pgd_introduction.ipynb
│   ├── 02_pgd_implementation.ipynb
│   ├── 03_pgd_experiments.ipynb
│   └── 04_pgd_analysis.ipynb
│
├── src/
│   ├── __init__.py
│   ├── attacks/
│   │   ├── __init__.py
│   │   ├── fgsm.py
│   │   └── pgd.py
│   ├── models/
│   │   ├── __init__.py
│   │   └── load_models.py
│   ├── utils/
│   │   ├── __init__.py
│   │   ├── visualization.py
│   │   └── evaluation.py
│   └── data/
│       ├── __init__.py
│       └── dataloaders.py
│
├── scripts/
│   ├── run_pgd_attack.py
│   └── batch_experiments.py
│
├── data/
│   └── .gitkeep
│
├── results/
│   ├── figures/
│   └── logs/
│
└── models/
    └── .gitkeep
```

## 🔧 Step 3: Generate Core Implementation Files

### Generate PGD Attack Implementation

```bash
claude-code "Create src/attacks/pgd.py with a complete PGD attack implementation including:
- pgd_attack function with epsilon, alpha, iterations parameters
- Support for both targeted and untargeted attacks
- Random initialization option
- L-infinity norm projection
- Proper gradient computation and sign-based updates
Include detailed docstrings and type hints"
```

### Generate FGSM Implementation

```bash
claude-code "Create src/attacks/fgsm.py with FGSM attack implementation for comparison with PGD"
```

### Generate Visualization Utilities

```bash
claude-code "Create src/utils/visualization.py with functions to:
- Plot original vs adversarial images
- Show perturbation heatmaps
- Display attack success indicators
- Plot loss landscapes during iterations
- Create comparison grids for multiple attacks"
```

### Generate Model Loading Utilities

```bash
claude-code "Create src/models/load_models.py with functions to:
- Load pretrained ResNet, VGG, and DenseNet models
- Handle model normalization for ImageNet
- Set models to evaluation mode
- Support both CPU and GPU devices"
```

## 📔 Step 4: Create Jupyter Notebooks

### Notebook 1: Introduction to PGD

```bash
claude-code "Create notebooks/01_pgd_introduction.ipynb with:
- Markdown cells explaining PGD theory
- Mathematical formulation with LaTeX
- Comparison with FGSM
- Epsilon ball visualization
- Interactive widgets for parameter exploration
Structure it as a tutorial with clear sections"
```

### Notebook 2: PGD Implementation

```bash
claude-code "Create notebooks/02_pgd_implementation.ipynb with:
- Step-by-step PGD implementation from scratch
- Detailed comments explaining each step
- Visualization of the iterative process
- Comparison with the modular implementation in src/
- Tests on single images with verbose output"
```

### Notebook 3: Experiments

```bash
claude-code "Create notebooks/03_pgd_experiments.ipynb with:
- Parameter sensitivity analysis (epsilon, alpha, iterations)
- Comparison of different models' robustness
- Random restarts experiment
- Targeted vs untargeted attacks
- Success rate measurements
- Batch attack demonstrations"
```

### Notebook 4: Analysis

```bash
claude-code "Create notebooks/04_pgd_analysis.ipynb with:
- Pixel-level trajectory analysis
- Loss landscape visualization
- Perturbation distribution analysis
- Transferability experiments
- Defense mechanism testing
- Statistical analysis of results"
```

## 🏃 Step 5: Create Executable Scripts

### Main Attack Script

```bash
claude-code "Create scripts/run_pgd_attack.py as a command-line tool with:
- Argparse for parameters (model, epsilon, iterations, etc.)
- Support for single image or batch processing
- Save results to results/ directory
- Progress bars with tqdm
- Logging functionality
- Example usage in docstring"
```

### Batch Experiment Script

```bash
claude-code "Create scripts/batch_experiments.py to:
- Run experiments with different hyperparameters
- Generate comparison plots
- Save results in CSV format
- Create LaTeX tables for papers
- Support parallel processing"
```

## 📊 Step 6: Generate Sample Data and Tests

### Download Test Images

```bash
# Using Claude Code to create a download script
claude-code "Create scripts/download_sample_data.py to:
- Download CIFAR-10 test set samples
- Download ImageNet validation samples
- Organize in data/ directory
- Handle errors gracefully"

# Run the download script
python scripts/download_sample_data.py
```

### Create Unit Tests

```bash
claude-code "Create tests/test_pgd.py with pytest unit tests for:
- PGD attack function
- Epsilon ball projection
- Gradient computation
- Attack success metrics
- Edge cases (black/white images)"
```

## 🎯 Step 7: Run the Demo

### Start Jupyter Lab

```bash
# Make sure environment is activated
conda activate pgd-demo

# Install kernel for Jupyter
python -m ipykernel install --user --name pgd-demo --display-name "PGD Demo"

# Start Jupyter Lab
jupyter lab

# Or for classic notebook
jupyter notebook
```

### Run Command-Line Demo

```bash
# Single image attack
python scripts/run_pgd_attack.py \
    --model resnet18 \
    --image data/samples/cat.jpg \
    --epsilon 0.03 \
    --iterations 20 \
    --alpha 0.001 \
    --save-results

# Batch experiments
python scripts/batch_experiments.py \
    --models resnet18 vgg16 densenet121 \
    --epsilons 0.01 0.03 0.05 \
    --num-images 100 \
    --output results/experiment_1/
```

## 🧪 Step 8: Interactive Development with Claude Code

### Debugging and Refinement

```bash
# Debug PGD implementation
claude-code "Review src/attacks/pgd.py and identify any potential issues with gradient computation or projection"

# Optimize performance
claude-code "Optimize the PGD implementation for batch processing and GPU acceleration"

# Add new features
claude-code "Add support for L2 and L0 norm constraints to the PGD attack"
```

### Generate Documentation

```bash
# Create comprehensive README
claude-code "Create a detailed README.md with:
- Project overview
- Installation instructions  
- Usage examples
- API documentation
- Results and benchmarks
- Citation information"

# Generate API docs
claude-code "Create docs/API.md with complete documentation of all functions in src/"
```

## 🔍 Step 9: Troubleshooting

### Common Issues and Solutions

#### CUDA/GPU Issues
```bash
# Check CUDA availability
python -c "import torch; print(torch.cuda.is_available())"

# If false, install CPU-only PyTorch
conda install pytorch torchvision cpuonly -c pytorch
```

#### Memory Issues
```bash
# Reduce batch size in scripts
claude-code "Modify scripts to use smaller batch sizes and add memory management"
```

#### Import Errors
```bash
# Add project to Python path
export PYTHONPATH="${PYTHONPATH}:${PWD}"

# Or in notebook
import sys
sys.path.append('..')
```

## 📈 Step 10: Extend and Customize

### Add New Attack Methods

```bash
claude-code "Implement C&W (Carlini-Wagner) attack in src/attacks/cw.py"
```

### Add Defenses

```bash
claude-code "Create src/defenses/adversarial_training.py with:
- Adversarial training loop
- PGD-based training
- Evaluation metrics"
```

### Create Web Interface

```bash
claude-code "Create app.py using Gradio to:
- Upload images
- Select attack parameters
- Run PGD attack
- Display results
- Download adversarial examples"
```

## 🎓 Learning Path

1. **Start with Theory**: Read through `01_pgd_introduction.ipynb`
2. **Understand Implementation**: Work through `02_pgd_implementation.ipynb`
3. **Run Experiments**: Execute cells in `03_pgd_experiments.ipynb`
4. **Analyze Results**: Explore `04_pgd_analysis.ipynb`
5. **Try Command-Line**: Use `scripts/run_pgd_attack.py`
6. **Modify Code**: Experiment with parameters in `src/attacks/pgd.py`
7. **Build Your Own**: Create custom attacks and defenses

## 📚 Additional Resources

### Papers to Read
- [Towards Deep Learning Models Resistant to Adversarial Attacks](https://arxiv.org/abs/1706.06083) (Madry et al., 2017)
- [Explaining and Harnessing Adversarial Examples](https://arxiv.org/abs/1412.6572) (Goodfellow et al., 2014)

### Useful Commands

```bash
# Monitor GPU usage
watch -n 1 nvidia-smi

# Profile code performance
python -m cProfile scripts/run_pgd_attack.py

# Run tests
pytest tests/ -v

# Format code
black src/ scripts/

# Check code quality
pylint src/
```

## 🤝 Contributing

To contribute improvements:

```bash
# Create new feature
claude-code "Implement a new feature for [describe feature]"

# Generate tests
claude-code "Create comprehensive tests for the new feature"

# Update documentation
claude-code "Update documentation to reflect the new feature"
```

## 📄 License

This project is for educational purposes. Please cite the original PGD paper if using for research:

```bibtex
@article{madry2017towards,
  title={Towards deep learning models resistant to adversarial attacks},
  author={Madry, Aleksander and Makelov, Aleksandar and Schmidt, Ludwig and Tsipras, Dimitris and Vladu, Adrian},
  journal={arXiv preprint arXiv:1706.06083},
  year={2017}
}
```

---

**Happy Adversarial Learning! 🚀**