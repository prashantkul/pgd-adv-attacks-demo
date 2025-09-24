"""
FGSM (Fast Gradient Sign Method) Attack Implementation

This module implements the FGSM adversarial attack for generating adversarial examples.
FGSM is a single-step attack method that generates adversarial examples by taking
a step in the direction of the gradient sign.

Reference:
    Goodfellow et al., "Explaining and Harnessing Adversarial Examples"
    https://arxiv.org/abs/1412.6572
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple


def fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform FGSM (Fast Gradient Sign Method) attack on the input images.
    
    Args:
        model: The neural network model to attack
        images: Input images tensor of shape (batch_size, channels, height, width)
        labels: True labels for untargeted attack or source labels for targeted attack
        epsilon: Maximum perturbation magnitude (L-infinity norm)
        targeted: If True, perform targeted attack towards target_labels
        target_labels: Labels to target for targeted attack (required if targeted=True)
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
    
    # Set requires_grad for input images
    images.requires_grad = True
    
    # Forward pass
    outputs = model(images)
    
    # Calculate loss
    if targeted:
        # For targeted attack, minimize loss w.r.t. target labels
        loss = F.cross_entropy(outputs, target_labels)
    else:
        # For untargeted attack, maximize loss w.r.t. true labels
        loss = F.cross_entropy(outputs, labels)
    
    # Backward pass
    model.zero_grad()
    loss.backward()
    
    # Get gradient sign
    grad_sign = images.grad.sign()
    
    # Create adversarial images
    if targeted:
        # For targeted attack, move in opposite direction of gradient
        adversarial_images = images - epsilon * grad_sign
    else:
        # For untargeted attack, move in direction of gradient
        adversarial_images = images + epsilon * grad_sign
    
    # Clamp to valid image range [0, 1]
    adversarial_images = torch.clamp(adversarial_images, 0, 1)
    
    # Calculate perturbations
    perturbations = adversarial_images - images.detach()
    
    return adversarial_images.detach(), perturbations.detach()


def i_fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    iterations: int = 10,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform I-FGSM (Iterative Fast Gradient Sign Method) attack.
    
    I-FGSM is an iterative version of FGSM that applies smaller steps multiple times.
    This is similar to PGD without random initialization.
    
    Args:
        model: The neural network model to attack
        images: Input images tensor
        labels: True labels for untargeted attack
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        iterations: Number of iterations
        targeted: If True, perform targeted attack
        target_labels: Labels to target for targeted attack
        device: Device to run the attack on
    
    Returns:
        Tuple of (adversarial_images, perturbations)
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
    
    # Store original images
    original_images = images.clone().detach()
    
    # Initialize adversarial images
    adv_images = images.clone().detach()
    
    # Iterative attack
    for _ in range(iterations):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Calculate loss
        if targeted:
            loss = F.cross_entropy(outputs, target_labels)
        else:
            loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Get gradient sign
        grad_sign = adv_images.grad.sign()
        
        # Update adversarial images
        with torch.no_grad():
            if targeted:
                # For targeted attack, move in opposite direction
                adv_images = adv_images - alpha * grad_sign
            else:
                # For untargeted attack, move in gradient direction
                adv_images = adv_images + alpha * grad_sign
            
            # Project back to epsilon ball (L-infinity)
            perturbation = torch.clamp(adv_images - original_images, -epsilon, epsilon)
            adv_images = original_images + perturbation
            
            # Clamp to valid image range
            adv_images = torch.clamp(adv_images, 0, 1)
    
    # Calculate final perturbation
    perturbations = adv_images - original_images
    
    return adv_images.detach(), perturbations.detach()


def momentum_fgsm_attack(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    alpha: float = 0.01,
    iterations: int = 10,
    decay: float = 0.9,
    targeted: bool = False,
    target_labels: Optional[torch.Tensor] = None,
    device: Optional[torch.device] = None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Perform MI-FGSM (Momentum Iterative Fast Gradient Sign Method) attack.
    
    MI-FGSM uses momentum to stabilize update directions and escape poor local maxima.
    
    Args:
        model: The neural network model to attack
        images: Input images tensor
        labels: True labels for untargeted attack
        epsilon: Maximum perturbation magnitude
        alpha: Step size for each iteration
        iterations: Number of iterations
        decay: Momentum decay factor
        targeted: If True, perform targeted attack
        target_labels: Labels to target for targeted attack
        device: Device to run the attack on
    
    Returns:
        Tuple of (adversarial_images, perturbations)
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
    
    # Store original images
    original_images = images.clone().detach()
    
    # Initialize adversarial images and momentum
    adv_images = images.clone().detach()
    momentum = torch.zeros_like(images).to(device)
    
    # Iterative attack with momentum
    for _ in range(iterations):
        adv_images.requires_grad = True
        
        # Forward pass
        outputs = model(adv_images)
        
        # Calculate loss
        if targeted:
            loss = F.cross_entropy(outputs, target_labels)
        else:
            loss = F.cross_entropy(outputs, labels)
        
        # Backward pass
        model.zero_grad()
        loss.backward()
        
        # Update momentum
        grad = adv_images.grad.data
        grad_norm = grad / grad.view(grad.shape[0], -1).norm(1, dim=1).view(-1, 1, 1, 1)
        momentum = decay * momentum + grad_norm
        
        # Get momentum sign
        momentum_sign = momentum.sign()
        
        # Update adversarial images
        with torch.no_grad():
            if targeted:
                # For targeted attack, move in opposite direction
                adv_images = adv_images - alpha * momentum_sign
            else:
                # For untargeted attack, move in momentum direction
                adv_images = adv_images + alpha * momentum_sign
            
            # Project back to epsilon ball (L-infinity)
            perturbation = torch.clamp(adv_images - original_images, -epsilon, epsilon)
            adv_images = original_images + perturbation
            
            # Clamp to valid image range
            adv_images = torch.clamp(adv_images, 0, 1)
    
    # Calculate final perturbation
    perturbations = adv_images - original_images
    
    return adv_images.detach(), perturbations.detach()


def compare_fgsm_variants(
    model: nn.Module,
    images: torch.Tensor,
    labels: torch.Tensor,
    epsilon: float = 0.03,
    device: Optional[torch.device] = None
) -> dict:
    """
    Compare different FGSM variants on the same set of images.
    
    Args:
        model: The neural network model to attack
        images: Input images tensor
        labels: True labels
        epsilon: Maximum perturbation magnitude
        device: Device to run the attacks on
    
    Returns:
        Dictionary containing results for each variant:
            - fgsm: Results for basic FGSM
            - i_fgsm: Results for Iterative FGSM
            - mi_fgsm: Results for Momentum Iterative FGSM
    """
    if device is None:
        device = next(model.parameters()).device
    
    results = {}
    
    # Basic FGSM
    adv_fgsm, pert_fgsm = fgsm_attack(model, images, labels, epsilon, device=device)
    
    # I-FGSM
    adv_ifgsm, pert_ifgsm = i_fgsm_attack(
        model, images, labels, epsilon, alpha=epsilon/10, iterations=10, device=device
    )
    
    # MI-FGSM
    adv_mifgsm, pert_mifgsm = momentum_fgsm_attack(
        model, images, labels, epsilon, alpha=epsilon/10, iterations=10, device=device
    )
    
    # Evaluate each variant
    model.eval()
    with torch.no_grad():
        # Move tensors to device
        images = images.to(device)
        labels = labels.to(device)
        
        for name, adv_images, perturbations in [
            ('fgsm', adv_fgsm, pert_fgsm),
            ('i_fgsm', adv_ifgsm, pert_ifgsm),
            ('mi_fgsm', adv_mifgsm, pert_mifgsm)
        ]:
            # Get predictions
            outputs_original = model(images)
            outputs_adversarial = model(adv_images)
            
            _, pred_original = outputs_original.max(1)
            _, pred_adversarial = outputs_adversarial.max(1)
            
            # Calculate metrics
            correct_original = (pred_original == labels).float()
            correct_adversarial = (pred_adversarial == labels).float()
            
            accuracy_original = correct_original.mean().item()
            accuracy_adversarial = correct_adversarial.mean().item()
            
            attack_success = (correct_original == 1) & (correct_adversarial == 0)
            attack_success_rate = attack_success.float().mean().item()
            
            # Perturbation statistics
            l2_norm = perturbations.view(perturbations.shape[0], -1).norm(2, dim=1).mean().item()
            linf_norm = perturbations.abs().max().item()
            
            results[name] = {
                'accuracy_original': accuracy_original,
                'accuracy_adversarial': accuracy_adversarial,
                'attack_success_rate': attack_success_rate,
                'mean_l2_perturbation': l2_norm,
                'max_linf_perturbation': linf_norm,
                'adversarial_images': adv_images,
                'perturbations': perturbations
            }
    
    return results