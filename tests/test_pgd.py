"""
Unit tests for PGD attack implementation.
"""

import torch
import torch.nn as nn
import numpy as np
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from attacks.pgd import pgd_attack, pgd_attack_with_restarts, evaluate_attack
from attacks.fgsm import fgsm_attack
from models.load_models import load_pretrained_model


class SimpleModel(nn.Module):
    """Simple test model for unit tests."""
    def __init__(self, num_classes=10):
        super(SimpleModel, self).__init__()
        self.conv1 = nn.Conv2d(3, 16, 3, padding=1)
        self.conv2 = nn.Conv2d(16, 32, 3, padding=1)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(32, num_classes)
        self.relu = nn.ReLU()
    
    def forward(self, x):
        x = self.relu(self.conv1(x))
        x = self.relu(self.conv2(x))
        x = self.pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def test_pgd_basic():
    """Test basic PGD attack functionality."""
    # Create simple model and data
    model = SimpleModel()
    model.eval()
    
    batch_size = 4
    images = torch.randn(batch_size, 3, 32, 32)
    images = torch.clamp(images, 0, 1)  # Ensure valid image range
    labels = torch.randint(0, 10, (batch_size,))
    
    epsilon = 0.03
    alpha = 0.01
    iterations = 10
    
    # Run PGD attack
    adv_images, perturbations = pgd_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations,
        random_start=True
    )
    
    # Basic checks
    assert adv_images.shape == images.shape, "Adversarial images should have same shape as input"
    assert perturbations.shape == images.shape, "Perturbations should have same shape as input"
    
    # Check epsilon constraint
    max_perturbation = torch.abs(perturbations).max().item()
    assert max_perturbation <= epsilon + 1e-6, f"Max perturbation {max_perturbation} exceeds epsilon {epsilon}"
    
    # Check image range
    assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1), "Adversarial images should be in [0, 1] range"
    
    print("✓ Basic PGD attack test passed")


def test_pgd_targeted():
    """Test targeted PGD attack."""
    model = SimpleModel()
    model.eval()
    
    batch_size = 2
    images = torch.randn(batch_size, 3, 32, 32)
    images = torch.clamp(images, 0, 1)
    labels = torch.randint(0, 10, (batch_size,))
    target_labels = torch.randint(0, 10, (batch_size,))
    
    # Run targeted attack
    adv_images, perturbations = pgd_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=0.05,
        alpha=0.01,
        iterations=20,
        targeted=True,
        target_labels=target_labels
    )
    
    # Check shapes and constraints
    assert adv_images.shape == images.shape
    assert perturbations.shape == images.shape
    assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
    
    print("✓ Targeted PGD attack test passed")


def test_pgd_with_restarts():
    """Test PGD attack with random restarts."""
    model = SimpleModel()
    model.eval()
    
    images = torch.randn(2, 3, 32, 32)
    images = torch.clamp(images, 0, 1)
    labels = torch.randint(0, 10, (2,))
    
    # Run PGD with restarts
    adv_images, perturbations = pgd_attack_with_restarts(
        model=model,
        images=images,
        labels=labels,
        epsilon=0.03,
        alpha=0.01,
        iterations=10,
        restarts=3
    )
    
    # Check constraints
    assert adv_images.shape == images.shape
    assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
    max_pert = torch.abs(perturbations).max().item()
    assert max_pert <= 0.03 + 1e-6
    
    print("✓ PGD with restarts test passed")


def test_attack_evaluation():
    """Test attack evaluation metrics."""
    model = SimpleModel()
    model.eval()
    
    batch_size = 8
    images = torch.randn(batch_size, 3, 32, 32)
    images = torch.clamp(images, 0, 1)
    labels = torch.randint(0, 10, (batch_size,))
    
    # Create adversarial examples
    adv_images, _ = pgd_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=0.05,
        iterations=20
    )
    
    # Evaluate attack
    metrics = evaluate_attack(
        model=model,
        original_images=images,
        adversarial_images=adv_images,
        true_labels=labels
    )
    
    # Check metrics
    required_keys = ['accuracy_original', 'accuracy_adversarial', 'attack_success_rate', 
                     'mean_perturbation_l2', 'max_perturbation_linf']
    for key in required_keys:
        assert key in metrics, f"Missing metric: {key}"
        assert isinstance(metrics[key], (int, float)), f"Metric {key} should be numeric"
    
    print("✓ Attack evaluation test passed")


def test_l2_norm_constraint():
    """Test PGD attack with L2 norm constraint."""
    model = SimpleModel()
    model.eval()
    
    images = torch.randn(3, 3, 32, 32)
    images = torch.clamp(images, 0, 1)
    labels = torch.randint(0, 10, (3,))
    
    epsilon = 1.0
    
    # Run PGD with L2 constraint
    adv_images, perturbations = pgd_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        alpha=0.2,
        iterations=10,
        norm='2'
    )
    
    # Check L2 constraint
    l2_norms = perturbations.view(perturbations.shape[0], -1).norm(2, dim=1)
    assert torch.all(l2_norms <= epsilon + 1e-4), f"L2 norms {l2_norms} exceed epsilon {epsilon}"
    
    print("✓ L2 norm constraint test passed")


def test_fgsm_attack():
    """Test FGSM attack for comparison."""
    model = SimpleModel()
    model.eval()
    
    images = torch.randn(4, 3, 32, 32)
    images = torch.clamp(images, 0, 1)
    labels = torch.randint(0, 10, (4,))
    
    epsilon = 0.03
    
    # Run FGSM attack
    adv_images, perturbations = fgsm_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon
    )
    
    # Check constraints
    assert adv_images.shape == images.shape
    assert perturbations.shape == images.shape
    assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1)
    
    # FGSM should respect epsilon constraint exactly
    max_pert = torch.abs(perturbations).max().item()
    assert abs(max_pert - epsilon) < 1e-6, f"FGSM max perturbation {max_pert} should equal epsilon {epsilon}"
    
    print("✓ FGSM attack test passed")


if __name__ == "__main__":
    print("Running PGD attack tests...")
    
    test_pgd_basic()
    test_pgd_targeted() 
    test_pgd_with_restarts()
    test_attack_evaluation()
    test_l2_norm_constraint()
    test_fgsm_attack()
    
    print("\n✅ All tests passed successfully!")