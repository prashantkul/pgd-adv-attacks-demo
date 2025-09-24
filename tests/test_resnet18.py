"""
Test PGD attack implementation on ResNet18 with real images.
"""

import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from attacks.pgd import pgd_attack, evaluate_attack
from attacks.fgsm import fgsm_attack, compare_fgsm_variants
from models.load_models import load_pretrained_model, get_class_names
from utils.visualization import plot_adversarial_examples, plot_attack_comparison


def test_resnet18_with_cifar10():
    """Test PGD attack on ResNet18 with CIFAR-10 images."""
    print("Testing PGD attack on ResNet18 with CIFAR-10...")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    # Load ResNet18 model
    print("Loading ResNet18...")
    model, model_info = load_pretrained_model('resnet18', dataset='imagenet', device=device)
    
    # Load CIFAR-10 test data (just a few samples for testing)
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize CIFAR-10 to ImageNet size
        transforms.ToTensor(),
    ])
    
    # Create a small test dataset
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Get a small batch for testing
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=8, shuffle=True)
    images, labels = next(iter(test_loader))
    
    # Move to device
    images = images.to(device)
    labels = labels.to(device)
    
    print(f"Test batch shape: {images.shape}")
    print(f"Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    
    # Test clean accuracy first
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        _, predicted = outputs.max(1)
        clean_accuracy = (predicted == labels).float().mean().item()
    
    print(f"Clean accuracy on test batch: {clean_accuracy:.2%}")
    
    # Parameters for PGD attack
    epsilon = 0.03
    alpha = 0.001
    iterations = 40
    
    print(f"\nRunning PGD attack with epsilon={epsilon}, alpha={alpha}, iterations={iterations}")
    
    # Run PGD attack
    adv_images, perturbations = pgd_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations,
        random_start=True,
        device=device
    )
    
    print(f"Adversarial images shape: {adv_images.shape}")
    print(f"Perturbation range: [{perturbations.min().item():.6f}, {perturbations.max().item():.6f}]")
    print(f"Max L∞ perturbation: {perturbations.abs().max().item():.6f}")
    
    # Evaluate attack
    metrics = evaluate_attack(
        model=model,
        original_images=images,
        adversarial_images=adv_images,
        true_labels=labels,
        device=device
    )
    
    print("\n=== Attack Results ===")
    print(f"Original accuracy: {metrics['accuracy_original']:.2%}")
    print(f"Adversarial accuracy: {metrics['accuracy_adversarial']:.2%}")
    print(f"Attack success rate: {metrics['attack_success_rate']:.2%}")
    print(f"Mean L2 perturbation: {metrics['mean_perturbation_l2']:.4f}")
    print(f"Max L∞ perturbation: {metrics['max_perturbation_linf']:.6f}")
    
    # Verify constraints
    assert metrics['max_perturbation_linf'] <= epsilon + 1e-6, "Epsilon constraint violated"
    assert torch.all(adv_images >= 0) and torch.all(adv_images <= 1), "Image range constraint violated"
    
    print("\n✅ PGD attack constraints verified!")
    
    return images, adv_images, perturbations, metrics


def test_attack_comparison():
    """Compare different attack methods."""
    print("\nTesting attack comparison...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load model
    model, _ = load_pretrained_model('resnet18', dataset='imagenet', device=device)
    
    # Create sample images
    images = torch.randn(4, 3, 224, 224).to(device)
    images = torch.clamp(images, 0, 1)
    labels = torch.randint(0, 1000, (4,)).to(device)
    
    epsilon = 0.05
    
    # Test FGSM variants
    fgsm_results = compare_fgsm_variants(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        device=device
    )
    
    print("\n=== FGSM Variants Comparison ===")
    for method, results in fgsm_results.items():
        print(f"{method.upper()}:")
        print(f"  Attack success rate: {results['attack_success_rate']:.2%}")
        print(f"  Mean L2 perturbation: {results['mean_l2_perturbation']:.4f}")
        print(f"  Max L∞ perturbation: {results['max_linf_perturbation']:.6f}")
    
    # Test PGD vs FGSM
    adv_pgd, pert_pgd = pgd_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        alpha=epsilon/10,
        iterations=20,
        device=device
    )
    
    adv_fgsm, pert_fgsm = fgsm_attack(
        model=model,
        images=images,
        labels=labels,
        epsilon=epsilon,
        device=device
    )
    
    # Compare effectiveness
    with torch.no_grad():
        outputs_pgd = model(adv_pgd)
        outputs_fgsm = model(adv_fgsm)
        
        _, pred_pgd = outputs_pgd.max(1)
        _, pred_fgsm = outputs_fgsm.max(1)
        
        pgd_success = (pred_pgd != labels).float().mean().item()
        fgsm_success = (pred_fgsm != labels).float().mean().item()
    
    print(f"\n=== PGD vs FGSM ===")
    print(f"PGD success rate: {pgd_success:.2%}")
    print(f"FGSM success rate: {fgsm_success:.2%}")
    
    print("✅ Attack comparison completed!")


def test_targeted_attack():
    """Test targeted attack functionality."""
    print("\nTesting targeted attack...")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model, _ = load_pretrained_model('resnet18', dataset='imagenet', device=device)
    
    # Create test images
    images = torch.randn(3, 3, 224, 224).to(device)
    images = torch.clamp(images, 0, 1)
    source_labels = torch.randint(0, 1000, (3,)).to(device)
    target_labels = torch.randint(0, 1000, (3,)).to(device)
    
    # Ensure target labels are different from source
    while torch.any(target_labels == source_labels):
        target_labels = torch.randint(0, 1000, (3,)).to(device)
    
    # Run targeted attack
    adv_images, perturbations = pgd_attack(
        model=model,
        images=images,
        labels=source_labels,
        epsilon=0.05,
        alpha=0.005,
        iterations=50,
        targeted=True,
        target_labels=target_labels,
        device=device
    )
    
    # Check if attack was successful
    with torch.no_grad():
        outputs = model(adv_images)
        _, predicted = outputs.max(1)
        success_rate = (predicted == target_labels).float().mean().item()
    
    print(f"Targeted attack success rate: {success_rate:.2%}")
    print(f"Max perturbation: {perturbations.abs().max().item():.6f}")
    
    print("✅ Targeted attack test completed!")


def save_visualization_results(images, adv_images, perturbations):
    """Save visualization results."""
    print("\nCreating visualizations...")
    
    # Convert to CPU and numpy for plotting
    images_np = images.cpu()
    adv_images_np = adv_images.cpu()
    perturbations_np = perturbations.cpu()
    
    # Create comparison plot
    fig = plot_adversarial_examples(
        original_images=images_np[:5],
        adversarial_images=adv_images_np[:5],
        perturbations=perturbations_np[:5],
        num_samples=5,
        figsize=(15, 8)
    )
    
    # Save plot
    os.makedirs('results/figures', exist_ok=True)
    fig.savefig('results/figures/pgd_attack_results.png', dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    print("✅ Visualization saved to results/figures/pgd_attack_results.png")


def main():
    """Run all tests."""
    print("=" * 60)
    print("PGD ATTACK TESTING ON RESNET18")
    print("=" * 60)
    
    try:
        # Test basic PGD attack
        images, adv_images, perturbations, metrics = test_resnet18_with_cifar10()
        
        # Save visualizations
        save_visualization_results(images, adv_images, perturbations)
        
        # Test attack comparisons
        test_attack_comparison()
        
        # Test targeted attacks
        test_targeted_attack()
        
        print("\n" + "=" * 60)
        print("🎉 ALL TESTS COMPLETED SUCCESSFULLY!")
        print("=" * 60)
        
        print(f"\nFinal Results Summary:")
        print(f"- PGD attack successfully implemented")
        print(f"- Epsilon constraints properly enforced")
        print(f"- Attack success rate: {metrics['attack_success_rate']:.1%}")
        print(f"- All safety checks passed")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ ResNet18 testing completed successfully!")
    else:
        print("\n❌ ResNet18 testing failed!")
        sys.exit(1)