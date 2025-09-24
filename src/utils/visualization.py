"""
Visualization utilities for adversarial attacks.

This module provides functions to visualize adversarial examples, perturbations,
attack progress, and comparison results.
"""

import torch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from typing import List, Optional, Union, Tuple
import warnings
warnings.filterwarnings('ignore')


def plot_adversarial_examples(
    original_images: torch.Tensor,
    adversarial_images: torch.Tensor,
    perturbations: Optional[torch.Tensor] = None,
    true_labels: Optional[List[str]] = None,
    pred_original: Optional[List[str]] = None,
    pred_adversarial: Optional[List[str]] = None,
    epsilon: Optional[float] = None,
    num_samples: int = 5,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Plot original images, adversarial examples, and perturbations side by side.
    
    Args:
        original_images: Original clean images
        adversarial_images: Adversarial examples
        perturbations: Perturbation added to create adversarial examples
        true_labels: True labels of the images
        pred_original: Predictions on original images
        pred_adversarial: Predictions on adversarial images
        epsilon: Epsilon value used for the attack
        num_samples: Number of samples to display
        figsize: Figure size
    
    Returns:
        matplotlib figure object
    """
    num_samples = min(num_samples, original_images.shape[0])
    
    # Calculate perturbations if not provided
    if perturbations is None:
        perturbations = adversarial_images - original_images
    
    # Convert tensors to numpy arrays
    if torch.is_tensor(original_images):
        original_images = original_images.cpu().numpy()
    if torch.is_tensor(adversarial_images):
        adversarial_images = adversarial_images.cpu().numpy()
    if torch.is_tensor(perturbations):
        perturbations = perturbations.cpu().numpy()
    
    # Determine number of rows (3 if perturbations provided, 2 otherwise)
    num_rows = 3 if perturbations is not None else 2
    
    fig, axes = plt.subplots(num_rows, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = axes.reshape(-1, 1)
    
    for i in range(num_samples):
        # Original image
        img_orig = np.transpose(original_images[i], (1, 2, 0))
        img_orig = np.clip(img_orig, 0, 1)
        
        axes[0, i].imshow(img_orig)
        title = "Original"
        if true_labels and pred_original:
            title += f"\nTrue: {true_labels[i]}\nPred: {pred_original[i]}"
        axes[0, i].set_title(title, fontsize=10)
        axes[0, i].axis('off')
        
        # Adversarial image
        img_adv = np.transpose(adversarial_images[i], (1, 2, 0))
        img_adv = np.clip(img_adv, 0, 1)
        
        axes[1, i].imshow(img_adv)
        title = "Adversarial"
        if epsilon:
            title += f" (ε={epsilon})"
        if pred_adversarial:
            title += f"\nPred: {pred_adversarial[i]}"
            if pred_original and pred_original[i] != pred_adversarial[i]:
                axes[1, i].set_title(title, fontsize=10, color='red')
            else:
                axes[1, i].set_title(title, fontsize=10)
        else:
            axes[1, i].set_title(title, fontsize=10)
        axes[1, i].axis('off')
        
        # Perturbation (if available)
        if num_rows == 3:
            pert = np.transpose(perturbations[i], (1, 2, 0))
            # Normalize perturbation for visualization
            pert_norm = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
            
            axes[2, i].imshow(pert_norm)
            axes[2, i].set_title(f"Perturbation\nL∞={np.abs(pert).max():.3f}", fontsize=10)
            axes[2, i].axis('off')
    
    plt.tight_layout()
    return fig


def plot_perturbation_heatmap(
    perturbations: torch.Tensor,
    num_samples: int = 5,
    figsize: Tuple[int, int] = (15, 3)
) -> plt.Figure:
    """
    Plot perturbation heatmaps to visualize attack patterns.
    
    Args:
        perturbations: Perturbation tensors
        num_samples: Number of samples to display
        figsize: Figure size
    
    Returns:
        matplotlib figure object
    """
    num_samples = min(num_samples, perturbations.shape[0])
    
    if torch.is_tensor(perturbations):
        perturbations = perturbations.cpu().numpy()
    
    fig, axes = plt.subplots(1, num_samples, figsize=figsize)
    if num_samples == 1:
        axes = [axes]
    
    for i in range(num_samples):
        # Calculate perturbation magnitude across channels
        pert = perturbations[i]
        if len(pert.shape) == 3:  # If channels dimension exists
            pert_magnitude = np.linalg.norm(pert, axis=0)
        else:
            pert_magnitude = np.abs(pert)
        
        im = axes[i].imshow(pert_magnitude, cmap='hot', interpolation='nearest')
        axes[i].set_title(f'Sample {i+1}\nMax: {pert_magnitude.max():.3f}', fontsize=10)
        axes[i].axis('off')
        plt.colorbar(im, ax=axes[i], fraction=0.046, pad=0.04)
    
    plt.suptitle('Perturbation Magnitude Heatmaps', fontsize=12, y=1.05)
    plt.tight_layout()
    return fig


def plot_attack_comparison(
    original_image: torch.Tensor,
    attack_results: dict,
    true_label: Optional[str] = None,
    figsize: Tuple[int, int] = (15, 5)
) -> plt.Figure:
    """
    Compare multiple attack methods on the same image.
    
    Args:
        original_image: Original clean image
        attack_results: Dictionary with attack names as keys and
                       (adversarial_image, perturbation, prediction) as values
        true_label: True label of the image
        figsize: Figure size
    
    Returns:
        matplotlib figure object
    """
    num_attacks = len(attack_results)
    
    if torch.is_tensor(original_image):
        original_image = original_image.cpu().numpy()
    
    # Create figure with subplots
    fig = plt.figure(figsize=figsize)
    gs = gridspec.GridSpec(3, num_attacks + 1, height_ratios=[1, 1, 0.8])
    
    # Plot original image
    ax_orig = plt.subplot(gs[0, 0])
    img_orig = np.transpose(original_image, (1, 2, 0))
    img_orig = np.clip(img_orig, 0, 1)
    ax_orig.imshow(img_orig)
    title = "Original"
    if true_label:
        title += f"\nLabel: {true_label}"
    ax_orig.set_title(title, fontsize=10)
    ax_orig.axis('off')
    
    # Empty space for alignment
    ax_empty = plt.subplot(gs[1, 0])
    ax_empty.axis('off')
    
    # Plot attack results
    for idx, (attack_name, (adv_img, pert, pred)) in enumerate(attack_results.items(), 1):
        if torch.is_tensor(adv_img):
            adv_img = adv_img.cpu().numpy()
        if torch.is_tensor(pert):
            pert = pert.cpu().numpy()
        
        # Adversarial image
        ax_adv = plt.subplot(gs[0, idx])
        img_adv = np.transpose(adv_img, (1, 2, 0))
        img_adv = np.clip(img_adv, 0, 1)
        ax_adv.imshow(img_adv)
        title = f"{attack_name}\nPred: {pred}"
        if true_label and pred != true_label:
            ax_adv.set_title(title, fontsize=10, color='red')
        else:
            ax_adv.set_title(title, fontsize=10)
        ax_adv.axis('off')
        
        # Perturbation
        ax_pert = plt.subplot(gs[1, idx])
        pert_vis = np.transpose(pert, (1, 2, 0))
        pert_norm = (pert_vis - pert_vis.min()) / (pert_vis.max() - pert_vis.min() + 1e-8)
        ax_pert.imshow(pert_norm)
        ax_pert.set_title(f"Perturbation\nL∞={np.abs(pert).max():.3f}", fontsize=10)
        ax_pert.axis('off')
        
        # Statistics
        ax_stats = plt.subplot(gs[2, idx])
        ax_stats.axis('off')
        stats_text = f"L2: {np.linalg.norm(pert):.3f}\n"
        stats_text += f"L∞: {np.abs(pert).max():.3f}\n"
        stats_text += f"Mean: {np.abs(pert).mean():.4f}"
        ax_stats.text(0.5, 0.5, stats_text, ha='center', va='center', fontsize=9)
    
    plt.suptitle('Attack Methods Comparison', fontsize=14, y=0.98)
    plt.tight_layout()
    return fig


def plot_loss_landscape(
    losses: List[float],
    accuracies: Optional[List[float]] = None,
    epsilon_values: Optional[List[float]] = None,
    title: str = "Attack Progress",
    figsize: Tuple[int, int] = (12, 4)
) -> plt.Figure:
    """
    Plot loss and accuracy progression during iterative attacks.
    
    Args:
        losses: List of loss values during iterations
        accuracies: List of accuracy values during iterations
        epsilon_values: List of epsilon values (for epsilon sweep)
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib figure object
    """
    if accuracies is not None:
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=figsize)
    else:
        fig, ax1 = plt.subplots(1, 1, figsize=(figsize[0]//2, figsize[1]))
    
    # Plot losses
    if epsilon_values:
        ax1.plot(epsilon_values, losses, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Epsilon', fontsize=11)
    else:
        ax1.plot(losses, 'b-o', linewidth=2, markersize=6)
        ax1.set_xlabel('Iteration', fontsize=11)
    
    ax1.set_ylabel('Loss', fontsize=11)
    ax1.set_title('Loss Progression', fontsize=12)
    ax1.grid(True, alpha=0.3)
    
    # Plot accuracies if provided
    if accuracies is not None:
        if epsilon_values:
            ax2.plot(epsilon_values, accuracies, 'r-s', linewidth=2, markersize=6)
            ax2.set_xlabel('Epsilon', fontsize=11)
        else:
            ax2.plot(accuracies, 'r-s', linewidth=2, markersize=6)
            ax2.set_xlabel('Iteration', fontsize=11)
        
        ax2.set_ylabel('Accuracy (%)', fontsize=11)
        ax2.set_title('Accuracy Progression', fontsize=12)
        ax2.grid(True, alpha=0.3)
    
    plt.suptitle(title, fontsize=14, y=1.02)
    plt.tight_layout()
    return fig


def plot_epsilon_vs_accuracy(
    epsilon_values: List[float],
    clean_accuracies: List[float],
    adversarial_accuracies: List[float],
    attack_success_rates: Optional[List[float]] = None,
    title: str = "Epsilon vs Model Performance",
    figsize: Tuple[int, int] = (10, 6)
) -> plt.Figure:
    """
    Plot the relationship between epsilon and model accuracy.
    
    Args:
        epsilon_values: List of epsilon values
        clean_accuracies: Accuracies on clean images
        adversarial_accuracies: Accuracies on adversarial images
        attack_success_rates: Attack success rates
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    # Plot accuracies
    ax.plot(epsilon_values, clean_accuracies, 'g-o', label='Clean Accuracy', 
            linewidth=2, markersize=8)
    ax.plot(epsilon_values, adversarial_accuracies, 'r-s', label='Adversarial Accuracy', 
            linewidth=2, markersize=8)
    
    if attack_success_rates:
        ax.plot(epsilon_values, attack_success_rates, 'b-^', label='Attack Success Rate', 
                linewidth=2, markersize=8)
    
    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ax.set_ylabel('Accuracy / Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    # Add shaded region to highlight robustness
    ax.fill_between(epsilon_values, adversarial_accuracies, clean_accuracies, 
                     alpha=0.2, color='gray', label='Robustness Gap')
    
    plt.tight_layout()
    return fig


def create_attack_gif(
    images_sequence: List[torch.Tensor],
    output_path: str = "attack_progress.gif",
    duration: int = 200,
    labels: Optional[List[str]] = None
):
    """
    Create an animated GIF showing the progression of an iterative attack.
    
    Args:
        images_sequence: List of image tensors at each iteration
        output_path: Path to save the GIF
        duration: Duration of each frame in milliseconds
        labels: Optional labels for each frame
    """
    try:
        from PIL import Image
        import io
        
        frames = []
        for idx, img_tensor in enumerate(images_sequence):
            if torch.is_tensor(img_tensor):
                img_tensor = img_tensor.cpu().numpy()
            
            # Create figure for this frame
            fig, ax = plt.subplots(figsize=(5, 5))
            img = np.transpose(img_tensor, (1, 2, 0))
            img = np.clip(img, 0, 1)
            ax.imshow(img)
            
            title = f"Iteration {idx}"
            if labels and idx < len(labels):
                title += f"\nPrediction: {labels[idx]}"
            ax.set_title(title, fontsize=12)
            ax.axis('off')
            
            # Convert matplotlib figure to PIL Image
            buf = io.BytesIO()
            plt.savefig(buf, format='png', bbox_inches='tight')
            buf.seek(0)
            frames.append(Image.open(buf))
            plt.close()
        
        # Save as GIF
        frames[0].save(
            output_path,
            save_all=True,
            append_images=frames[1:],
            duration=duration,
            loop=0
        )
        print(f"GIF saved to {output_path}")
        
    except ImportError:
        print("PIL not installed. Cannot create GIF.")


def plot_model_comparison(
    model_names: List[str],
    accuracies: dict,
    epsilon_values: List[float],
    title: str = "Model Robustness Comparison",
    figsize: Tuple[int, int] = (12, 6)
) -> plt.Figure:
    """
    Compare robustness of different models against adversarial attacks.
    
    Args:
        model_names: List of model names
        accuracies: Dictionary mapping model names to accuracy lists
        epsilon_values: List of epsilon values
        title: Plot title
        figsize: Figure size
    
    Returns:
        matplotlib figure object
    """
    fig, ax = plt.subplots(figsize=figsize)
    
    colors = plt.cm.get_cmap('tab10', len(model_names))
    
    for idx, model_name in enumerate(model_names):
        if model_name in accuracies:
            ax.plot(epsilon_values, accuracies[model_name], 
                   label=model_name, linewidth=2, 
                   marker=['o', 's', '^', 'v', 'D', 'p', '*'][idx % 7],
                   markersize=8, color=colors(idx))
    
    ax.set_xlabel('Epsilon (ε)', fontsize=12)
    ax.set_ylabel('Adversarial Accuracy (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.legend(loc='best', fontsize=11)
    ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig