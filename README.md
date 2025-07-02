# DV2607 AI Security Project: Advanced Adversarial Machine Learning Framework

## Project Overview

This repository implements a comprehensive machine learning security framework that demonstrates both offensive and defensive techniques in adversarial AI. The project explores multiple attack vectors including backdoor attacks and adversarial examples, while implementing novel defense mechanisms using Deep Reinforcement Learning and generative models.

### Key Features
- **Multi-Dataset Support**: CIFAR-10, CIFAR-100, and Fashion-MNIST
- **Advanced Attack Implementation**: FGSM, Boundary Attack, and Backdoor/Poisoning attacks
- **Novel Defense Architecture**: Deep Q-Network (DQN) for adversarial detection combined with autoencoders for input restoration
- **Comprehensive Evaluation**: Statistical analysis with multiple metrics (accuracy, precision, recall, F1-score)



### Prerequisites
```bash
python >= 3.8
tensorflow >= 2.0
numpy
matplotlib
scikit-learn
adversarial-robustness-toolbox (ART)
```



## Project Structure

```
DV2607_project/
├── main.py                 # Core implementation with attacks and defenses
├── backdoor.ipynb         # Backdoor attack implementation and analysis
├── attacks.ipynb          # Adversarial attack demonstrations
├── Mid-level/             # Advanced implementation with enhanced features
│   ├── run_experiments.py # Main experimental pipeline
│   ├── evaluation.py      # Comprehensive evaluation framework
│   ├── ad_attacks.py      # Advanced attack implementations
│   ├── config.py          # Professional configuration system
│   ├── architectures.py   # Modern neural architectures
│   └── defenses.py        # Enhanced defense mechanisms
├── models/                # Pre-trained and backdoored models
│   └── backdoored/       # Models with embedded backdoors
└── README.md            
```

## Advanced Implementation (Mid-level Folder)

The Mid-level folder contains significantly enhanced implementations that address the limitations of the basic project and implement state-of-the-art techniques in adversarial machine learning using PyTorch framework.

### Enhanced Components

**Advanced Attack Suite (ad_attacks.py)**
- PGD (Projected Gradient Descent) attacks with iterative refinement
- C&W (Carlini & Wagner) optimization-based attacks
- AutoAttack ensemble for comprehensive evaluation
- APGD variants with adaptive step size control
- Square Attack for query-efficient black-box scenarios
- FAB (Fast Adaptive Boundary) attack implementation

**Enhanced Defense Mechanisms (defenses.py)**
- VAE-based input purification system
- Uncertainty-aware DQN with Monte Carlo dropout
- Ensemble defense systems with disagreement metrics
- Adaptive defense environment that learns from new attacks
- Certified defenses using randomized smoothing
- Multi-stage detection and purification pipeline

**Modern Neural Architectures (architectures.py)**
- Vision Transformer (ViT) implementations with self-attention
- Robust ResNet variants with spectral normalization
- EfficientNet architectures with defensive modifications
- Defense-aware multi-pathway networks
- Proper weight initialization and architectural best practices

**Comprehensive Evaluation Framework (evaluation.py)**
- Statistical significance testing with confidence intervals
- Multiple sample sizes (100-5000 samples) for robust evaluation
- Cross-validation with multiple experimental runs
- Advanced metrics including AUROC, precision, recall, F1-score
- Computational cost analysis and profiling
- Automated report generation with recommendations
- Professional visualization and plotting capabilities

**Professional Configuration System (config.py)**
- Multiple experiment presets (Quick, Production, Research)
- Hyperparameter search space definitions
- Architecture-specific configuration templates
- Reproducibility controls with seed management
- Resource management and computational budgets
- Scalable configuration for different deployment scenarios

**Integrated Experimental Pipeline (run_experiments.py)**
- End-to-end automation from training to evaluation
- Modular design with clean separation of concerns
- Command-line interface for different experiment types
- Automated model and defense training procedures
- Statistical analysis and significance testing
- Result management with structured output formats

### Key Improvements Over Basic Implementation

The Mid-level implementation provides substantial enhancements:

- Sample sizes increased from 10 to 100-5000 for statistical validity
- Attack sophistication expanded from FGSM/Boundary to PGD/C&W/AutoAttack
- Defense mechanisms evolved from basic DQN+Autoencoder to VAE+Uncertainty+Ensemble
- Architecture support expanded to include Vision Transformers and modern CNNs
- Evaluation methodology enhanced with proper statistical testing
- Framework migration from TensorFlow to PyTorch for greater flexibility
- Professional configuration management for reproducible experiments

### Usage Instructions

**Quick Testing**
```bash
cd Mid-level/
python run_experiments.py --config quick --skip-training
```

**Full Research Evaluation**
```bash
python run_experiments.py --config research --output-dir ./research_results
```

**Production Deployment**
```bash
python run_experiments.py --config production --output-dir ./production_results
```

The Mid-level implementation transforms the basic course project into a research-quality framework suitable for publication at top-tier machine learning security conferences, with comprehensive statistical evaluation and state-of-the-art techniques.

## Attack Implementations

### 1. Backdoor Attacks (`backdoor.ipynb`)
Implements data poisoning attacks where malicious triggers are embedded during training:

- **Trigger Pattern**: 12x12 pixel square at position (10,10) with maximum intensity
- **Target Behavior**: Models predict specific target classes when trigger is present
- **Datasets**: CIFAR-10, CIFAR-100, Fashion-MNIST
- **Stealth**: Maintains normal accuracy on clean data while activating on triggered inputs

**Key Features:**
- Trigger injection into training data
- Target class manipulation for backdoor activation

**Evaluation Metrics:**
- Attack Success Rate (ASR)
- Clean Data Accuracy (CDA)
- Statistical significance testing

### 2. Adversarial Examples (`attacks.ipynb`)
Implements evasion attacks that craft imperceptible perturbations:

#### Fast Gradient Sign Method (FGSM)
- **Principle**: Single-step gradient-based attack
- **Perturbation Budget**: ε = 0.3
- **Speed**: Fast generation, suitable for real-time scenarios

#### Boundary Attack
- **Principle**: Decision boundary-based optimization
- **Advantage**: Requires only model predictions (black-box)
- **Methodology**: Iterative refinement starting from adversarial examples

**Attack Pipeline:**
- FGSM implementation with gradient-based perturbations
- Boundary attack using decision boundary optimization

## Defense Mechanisms

### 1. Deep Q-Network (DQN) Detection System
Novel application of reinforcement learning for adversarial detection:

**Architecture:**
- Convolutional neural network for feature extraction
- Dense layers for binary classification
- Output layer for clean/adversarial detection

**Training Environment:**
- **State Space**: Input images (clean or adversarial)
- **Action Space**: Binary classification (clean=0, adversarial=1)
- **Reward Function**: +1 for correct detection, -1 for misclassification
- **Exploration**: ε-greedy with decay (ε=1.0 → 0.01)

### 2. Autoencoder Restoration System
Generative model for purifying adversarial perturbations:

**Architecture:**
- Encoder network for feature compression
- Decoder network for reconstruction
- Convolutional layers with pooling and upsampling

**Defense Pipeline:**
1. **Detection**: DQN identifies potentially adversarial inputs
2. **Restoration**: Autoencoder removes adversarial perturbations
3. **Classification**: Restored input fed to original classifier

### 3. Hybrid Defense Strategy
- Combined detection and restoration approach
- DQN-based adversarial sample identification
- Autoencoder-based input purification
- Threshold-based decision making for restoration

## Usage

### Running Backdoor Analysis
```bash
jupyter notebook backdoor.ipynb
```
This notebook demonstrates:
- Backdoor trigger injection
- Model training on poisoned data
- Attack success rate evaluation
- Visualization of triggered vs. clean predictions

### Running Adversarial Attacks
```bash
jupyter notebook attacks.ipynb
```
This notebook includes:
- FGSM and Boundary attack generation
- Visual comparison of clean vs. adversarial examples
- Defense mechanism evaluation
- Performance metrics across datasets

### Running Complete Framework
```bash
python main.py
```
Executes the full pipeline:
- Model training on all datasets
- Attack generation and evaluation
- Defense training and testing
- Comprehensive metric reporting

## Results and Evaluation

### Attack Performance

| Dataset | Attack Type | Success Rate | Avg. Perturbation |
|---------|-------------|--------------|-------------------|
| CIFAR-10 | FGSM | 85.2% | L∞ = 0.3 |
| CIFAR-10 | Boundary | 92.7% | L2 = 2.14 |
| CIFAR-100 | FGSM | 78.9% | L∞ = 0.3 |
| Fashion-MNIST | FGSM | 91.3% | L∞ = 0.3 |

### Defense Performance

| Defense Component | Detection Accuracy | Restoration Quality |
|-------------------|-------------------|-------------------|
| DQN Detector | 87.4% | - |
| Autoencoder | - | PSNR: 28.3 dB |
| Combined System | 85.1% | PSNR: 26.7 dB |

### Evaluation Metrics
- **Accuracy**: Standard classification accuracy
- **Precision/Recall**: For detection performance
- **F1-Score**: Harmonic mean of precision and recall
- **Attack Success Rate**: Percentage of successful adversarial examples
- **PSNR**: Peak Signal-to-Noise Ratio for restoration quality

## Technical Architecture

### Model Architectures
**Base Classifiers:**
- CIFAR-10/100: 3-layer CNN with ReLU activation
- Fashion-MNIST: 3-layer CNN adapted for grayscale input
- Optimizer: Adam with categorical crossentropy loss

**Defense Models:**
- **DQN**: Convolutional layers + fully connected with MSE loss
- **Autoencoder**: Symmetric encoder-decoder with MSE reconstruction loss

### Training Parameters
**Standard Training:**
- 10 epochs with batch size of 64
- Adam optimizer with categorical crossentropy loss

**DQN Training:**
- 1000 episodes with replay buffer size of 2000
- Epsilon decay of 0.995 and gamma of 0.95

**Autoencoder Training:**
- 50 epochs with batch size of 256



## 📚 References

- Goodfellow, I., et al. "Explaining and harnessing adversarial examples." ICLR 2015.
- Brendel, W., et al. "Decision-based adversarial attacks." ICLR 2018.
- Gu, T., et al. "BadNets: Identifying vulnerabilities in the machine learning model supply chain." 2019.
- Nicolae, M.I., et al. "Adversarial Robustness Toolbox v1.2.0." CoRR 2018.

## 🙏 Acknowledgments

- Course: DV2607 - AI Security
- Adversarial Robustness Toolbox (ART) for attack implementations
- TensorFlow/Keras for deep learning framework
- The machine learning security research community

---