"""
PGD (Projected Gradient Descent) Attack Implementation

This module implements the PGD adversarial attack for generating adversarial examples.
PGD is an iterative attack method that generates adversarial examples by repeatedly
applying FGSM and projecting the result back to the epsilon ball.

Reference:
    Madry et al., "Towards Deep Learning Models Resistant to Adversarial Attacks"
    https://arxiv.org/abs/1706.06083
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Union


def pgd_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.001,
    iterations: int = 40,
    random_start: bool = True,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
    norm: str = 'inf',
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform PGD (Projected Gradient Descent) attack on the input images.
    
    Args:
        model: The neural network model to attack
        images: Input images tensor of shape (batch_size, channels, height, width)
        labels: True labels for untargeted attack or source labels for targeted attack
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        alpha: Step size for each iteration
        iterations: Number of PGD iterations
        random_start: Whether to initialize with random noise within epsilon ball
        targeted: If True, perform targeted attack towards target_labels
        target_labels: Labels to target for targeted attack (required if targeted=True)
        norm: Norm constraint ('inf' for L-infinity, '2' for L2)
        device: Device to run the attack on (CPU or CUDA)
    
    Returns:
        Tuple of (adversarial_images, perturbations)
            - adversarial_images: Adversarial examples
            - perturbations: The perturbations added to create adversarial examples
    
    Raises:
        ValueError: If targeted is True but target_labels is None
    """
    if targeted and target_labels is None:
        raise ValueError("target_labels must be provided for targeted attack")
    
    if device is None:
        device = next(model.parameters()).device
    
    # Move tensors to device
    images = images.to(device)
    labels = labels.to(device)
    if target_labels is not None:
        target_labels = target_labels.to(device)
    
    # Store original images for projection
    original_images = images.clone().detach()
    
    # Initialize adversarial images
    adv_images = images.clone().detach()
    
    # Random initialization within epsilon ball
    if random_start:
        if norm == 'inf':
            random_noise = torch.empty_like(images).uniform_(-epsilon, epsilon)
        elif norm == '2':
            random_noise = torch.randn_like(images)
            random_noise = random_noise / random_noise.view(images.shape[0], -1).norm(2, dim=1).view(-1, 1, 1, 1)
            random_noise = random_noise * epsilon * torch.rand(images.shape[0], 1, 1, 1).to(device)
        else:
            raise ValueError(f"Unsupported norm: {norm}")
        
        adv_images = adv_images + random_noise
        adv_images = torch.clamp(adv_images, 0, 1)
    
    # Iterative attack
    for _ in range(iterations):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Calculate loss
        if targeted:
            # For targeted attack, minimize loss w.r.t. target labels
            loss = F.cross_entropy(outputs, target_labels)
            loss = -loss  # Minimize negative loss to maximize target class probability
        else:
            # For untargeted attack, maximize loss w.r.t. true labels
            loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get gradient sign for update
        grad_sign = adv_images.grad.sign()
        
        # Update adversarial images
        with torch.no_grad():
            if targeted:
                # For targeted attack, move in opposite direction of gradient
                adv_images = adv_images - alpha * grad_sign
            else:
                # For untargeted attack, move in direction of gradient
                adv_images = adv_images + alpha * grad_sign
            
            # Project back to epsilon ball
            if norm == 'inf':
                # L-infinity projection
                perturbation = torch.clamp(adv_images - original_images, -epsilon, epsilon)
            elif norm == '2':
                # L2 projection
                perturbation = adv_images - original_images
                perturbation_norms = perturbation.view(images.shape[0], -1).norm(2, dim=1)
                factor = epsilon / (perturbation_norms + 1e-10)
                factor = torch.min(factor, torch.ones_like(factor))
                perturbation = perturbation * factor.view(-1, 1, 1, 1)
            else:
                raise ValueError(f"Unsupported norm: {norm}")
            
            # Apply perturbation and clamp to valid image range
            adv_images = original_images + perturbation
            adv_images = torch.clamp(adv_images, 0, 1)
    
    # Calculate final perturbation
    perturbations = adv_images - original_images
    
    return adv_images.detach(), perturbations.detach()


def pgd_attack_with_restarts(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.001,
    iterations: int = 40,
    restarts: int = 10,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
    norm: str = 'inf',
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform PGD attack with random restarts.
    
    This function runs PGD attack multiple times with different random initializations
    and returns the adversarial examples that achieve the highest loss (for untargeted)
    or lowest loss (for targeted) attack.
    
    Args:
        model: The neural network model to attack
        images: Input images tensor
        labels: True labels for untargeted attack
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        iterations: Number of PGD iterations per restart
        restarts: Number of random restarts
        targeted: If True, perform targeted attack
        target_labels: Labels to target for targeted attack
        norm: Norm constraint ('inf' or '2')
        device: Device to run the attack on
    
    Returns:
        Tuple of (best_adversarial_images, best_perturbations)
    """
    if device is None:
        device = next(model.parameters()).device
    
    best_adv_images = None
    best_loss = None if not targeted else float('inf')
    best_perturbations = None
    
    for _ in range(restarts):
        # Run PGD attack with random start
        adv_images, perturbations = pgd_attack(
            model=model,
            images=images,
            labels=labels,
            epsilon=epsilon,
            alpha=alpha,
            iterations=iterations,
            random_start=True,
            targeted=targeted,
            target_labels=target_labels,
            norm=norm,
            device=device
        )
        
        # Evaluate attack effectiveness
        with torch.no_grad():
            outputs = model(adv_images)
            if targeted:
                loss = F.cross_entropy(outputs, target_labels, reduction='none')
                # For targeted attack, we want minimum loss
                if best_adv_images is None or loss.mean() < best_loss:
                    best_loss = loss.mean()
                    best_adv_images = adv_images
                    best_perturbations = perturbations
            else:
                loss = F.cross_entropy(outputs, labels, reduction='none')
                # For untargeted attack, we want maximum loss
                if best_adv_images is None or loss.mean() > best_loss:
                    best_loss = loss.mean()
                    best_adv_images = adv_images
                    best_perturbations = perturbations
    
    return best_adv_images, best_perturbations


def evaluate_attack(
    model: nn.Module,
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    true_labels: torch.Tensor,
    device: Optional[torch.device] = None
) -> dict:
    """
    Evaluate the effectiveness of adversarial attack.
    
    Args:
        model: The neural network model
        original_images: Original clean images
        adversarial_images: Adversarial examples
        true_labels: True labels of the images
        device: Device to run evaluation on
    
    Returns:
        Dictionary containing evaluation metrics:
            - accuracy_original: Accuracy on original images
            - accuracy_adversarial: Accuracy on adversarial images
            - attack_success_rate: Percentage of successful attacks
            - mean_perturbation: Mean L2 norm of perturbations
            - max_perturbation: Maximum L-infinity norm of perturbations
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    
    with torch.no_grad():
        # Move tensors to device
        original_images = original_images.to(device)
        adversarial_images = adversarial_images.to(device)
        true_labels = true_labels.to(device)
        
        # Get predictions for original images
        outputs_original = model(original_images)
        _, predicted_original = outputs_original.max(1)
        correct_original = (predicted_original == true_labels).float()
        
        # Get predictions for adversarial images
        outputs_adversarial = model(adversarial_images)
        _, predicted_adversarial = outputs_adversarial.max(1)
        correct_adversarial = (predicted_adversarial == true_labels).float()
        
        # Calculate metrics
        accuracy_original = correct_original.mean().item()
        accuracy_adversarial = correct_adversarial.mean().item()
        
        # Attack is successful if it causes misclassification on correctly classified samples
        attack_success = (correct_original == 1) & (correct_adversarial == 0)
        attack_success_rate = attack_success.float().mean().item()
        
        # Calculate perturbation statistics
        perturbations = adversarial_images - original_images
        mean_perturbation_l2 = perturbations.view(perturbations.shape[0], -1).norm(2, dim=1).mean().item()
        max_perturbation_linf = perturbations.abs().max().item()
    
    return {
        'accuracy_original': accuracy_original,
        'accuracy_adversarial': accuracy_adversarial,
        'attack_success_rate': attack_success_rate,
        'mean_perturbation_l2': mean_perturbation_l2,
        'max_perturbation_linf': max_perturbation_linf
    }