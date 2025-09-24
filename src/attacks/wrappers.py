"""
Class wrappers for function-based attack implementations.
These provide a class-based interface that's compatible with the notebook examples.
"""

import torch
import torch.nn as nn
from .pgd import pgd_attack, pgd_attack_with_restarts
from .fgsm import fgsm_attack, i_fgsm_attack, momentum_fgsm_attack


class PGDAttack:
    """
    Class-based wrapper for PGD attack function.
    """
    
    def __init__(self, model, epsilon=0.03, alpha=0.001, steps=40, 
                 random_start=True, norm='inf', device=None):
        """
        Initialize PGD attack.
        
        Args:
            model: Target neural network model
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            steps: Number of PGD iterations
            random_start: Whether to use random initialization
            norm: Norm constraint ('inf' or '2')
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.norm = norm
        self.device = device if device else next(model.parameters()).device
    
    def __call__(self, images, labels, targeted=False, target_labels=None):
        """
        Generate adversarial examples.
        
        Args:
            images: Input images tensor
            labels: True labels for untargeted attack
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack
        
        Returns:
            Adversarial examples tensor
        """
        adv_images, _ = pgd_attack(
            model=self.model,
            images=images,
            labels=labels,
            epsilon=self.epsilon,
            alpha=self.alpha,
            iterations=self.steps,
            random_start=self.random_start,
            targeted=targeted,
            target_labels=target_labels,
            norm=self.norm,
            device=self.device
        )
        return adv_images
    
    def attack(self, images, labels, targeted=False, target_labels=None):
        """
        Alternative method name for generating adversarial examples.
        """
        return self.__call__(images, labels, targeted, target_labels)


class PGDAttackWithRestarts:
    """
    Class-based wrapper for PGD attack with random restarts.
    """
    
    def __init__(self, model, epsilon=0.03, alpha=0.001, steps=40, 
                 restarts=10, norm='inf', device=None):
        """
        Initialize PGD attack with restarts.
        
        Args:
            model: Target neural network model
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            steps: Number of PGD iterations per restart
            restarts: Number of random restarts
            norm: Norm constraint ('inf' or '2')
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.restarts = restarts
        self.norm = norm
        self.device = device if device else next(model.parameters()).device
    
    def __call__(self, images, labels, targeted=False, target_labels=None):
        """
        Generate adversarial examples with multiple restarts.
        
        Args:
            images: Input images tensor
            labels: True labels for untargeted attack
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack
        
        Returns:
            Best adversarial examples tensor
        """
        adv_images, _ = pgd_attack_with_restarts(
            model=self.model,
            images=images,
            labels=labels,
            epsilon=self.epsilon,
            alpha=self.alpha,
            iterations=self.steps,
            restarts=self.restarts,
            targeted=targeted,
            target_labels=target_labels,
            norm=self.norm,
            device=self.device
        )
        return adv_images
    
    def attack(self, images, labels, targeted=False, target_labels=None):
        """
        Alternative method name for generating adversarial examples.
        """
        return self.__call__(images, labels, targeted, target_labels)


class FGSM:
    """
    Class-based wrapper for FGSM attack function.
    """
    
    def __init__(self, model, epsilon=0.03, device=None):
        """
        Initialize FGSM attack.
        
        Args:
            model: Target neural network model
            epsilon: Maximum perturbation magnitude
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.device = device if device else next(model.parameters()).device
    
    def __call__(self, images, labels, targeted=False, target_labels=None):
        """
        Generate adversarial examples using FGSM.
        
        Args:
            images: Input images tensor
            labels: True labels for untargeted attack
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack
        
        Returns:
            Adversarial examples tensor
        """
        adv_images, _ = fgsm_attack(
            model=self.model,
            images=images,
            labels=labels,
            epsilon=self.epsilon,
            targeted=targeted,
            target_labels=target_labels,
            device=self.device
        )
        return adv_images
    
    def attack(self, images, labels, targeted=False, target_labels=None):
        """
        Alternative method name for generating adversarial examples.
        """
        return self.__call__(images, labels, targeted, target_labels)


class IterativeFGSM:
    """
    Class-based wrapper for Iterative FGSM attack.
    """
    
    def __init__(self, model, epsilon=0.03, alpha=0.01, steps=10, device=None):
        """
        Initialize I-FGSM attack.
        
        Args:
            model: Target neural network model
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            steps: Number of iterations
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.device = device if device else next(model.parameters()).device
    
    def __call__(self, images, labels, targeted=False, target_labels=None):
        """
        Generate adversarial examples using I-FGSM.
        
        Args:
            images: Input images tensor
            labels: True labels for untargeted attack
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack
        
        Returns:
            Adversarial examples tensor
        """
        adv_images, _ = i_fgsm_attack(
            model=self.model,
            images=images,
            labels=labels,
            epsilon=self.epsilon,
            alpha=self.alpha,
            iterations=self.steps,
            targeted=targeted,
            target_labels=target_labels,
            device=self.device
        )
        return adv_images
    
    def attack(self, images, labels, targeted=False, target_labels=None):
        """
        Alternative method name for generating adversarial examples.
        """
        return self.__call__(images, labels, targeted, target_labels)


class MomentumFGSM:
    """
    Class-based wrapper for Momentum Iterative FGSM attack.
    """
    
    def __init__(self, model, epsilon=0.03, alpha=0.01, steps=10, 
                 decay=0.9, device=None):
        """
        Initialize MI-FGSM attack.
        
        Args:
            model: Target neural network model
            epsilon: Maximum perturbation magnitude
            alpha: Step size for each iteration
            steps: Number of iterations
            decay: Momentum decay factor
            device: Device to run attack on
        """
        self.model = model
        self.epsilon = epsilon
        self.alpha = alpha
        self.steps = steps
        self.decay = decay
        self.device = device if device else next(model.parameters()).device
    
    def __call__(self, images, labels, targeted=False, target_labels=None):
        """
        Generate adversarial examples using MI-FGSM.
        
        Args:
            images: Input images tensor
            labels: True labels for untargeted attack
            targeted: Whether to perform targeted attack
            target_labels: Target labels for targeted attack
        
        Returns:
            Adversarial examples tensor
        """
        adv_images, _ = momentum_fgsm_attack(
            model=self.model,
            images=images,
            labels=labels,
            epsilon=self.epsilon,
            alpha=self.alpha,
            iterations=self.steps,
            decay=self.decay,
            targeted=targeted,
            target_labels=target_labels,
            device=self.device
        )
        return adv_images
    
    def attack(self, images, labels, targeted=False, target_labels=None):
        """
        Alternative method name for generating adversarial examples.
        """
        return self.__call__(images, labels, targeted, target_labels)