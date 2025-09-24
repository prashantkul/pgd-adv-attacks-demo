"""
Enhanced test for PGD attack implementation on ResNet18 with detailed class probability analysis.
"""

import torch
import torch.nn.functional as F
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import numpy as np
import sys
import os
import json
import logging
from datetime import datetime

# Add src to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'src'))

from attacks.pgd import pgd_attack, evaluate_attack
from attacks.fgsm import fgsm_attack, compare_fgsm_variants
from models.load_models import load_pretrained_model, get_class_names
from utils.visualization import plot_adversarial_examples, plot_attack_comparison


def setup_logging():
    """Setup detailed logging."""
    log_dir = 'results/logs'
    os.makedirs(log_dir, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file = f"{log_dir}/pgd_test_{timestamp}.log"
    
    # Configure logging
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file),
            logging.StreamHandler()
        ]
    )
    
    return log_file


def get_top_predictions(model, images, top_k=5):
    """Get top-k predictions with probabilities."""
    model.eval()
    with torch.no_grad():
        outputs = model(images)
        probabilities = F.softmax(outputs, dim=1)
        top_probs, top_indices = torch.topk(probabilities, top_k, dim=1)
    
    return top_probs, top_indices, probabilities


def load_imagenet_classes():
    """Load ImageNet class names."""
    # Create a minimal class mapping for demonstration
    classes = {}
    for i in range(1000):
        classes[i] = f"class_{i}"
    
    # Add some common classes for better demonstration
    common_classes = {
        0: "tench", 1: "goldfish", 2: "great_white_shark", 3: "tiger_shark",
        4: "hammerhead", 5: "electric_ray", 6: "stingray", 7: "cock", 8: "hen",
        281: "tabby_cat", 285: "Egyptian_cat", 151: "Chihuahua",
        207: "golden_retriever", 281: "tabby_cat", 285: "Egyptian_cat"
    }
    
    classes.update(common_classes)
    return classes


def analyze_predictions(model, images, labels, attack_type="Original", top_k=5):
    """Analyze and log detailed predictions."""
    logging.info(f"\n=== {attack_type} Predictions Analysis ===")
    
    # Get predictions
    top_probs, top_indices, all_probs = get_top_predictions(model, images, top_k)
    
    # Load class names
    class_names = load_imagenet_classes()
    
    results = []
    
    for i in range(images.shape[0]):
        logging.info(f"\nSample {i+1}:")
        
        sample_result = {
            'sample_id': i+1,
            'true_label': labels[i].item() if labels is not None else 'Unknown',
            'predictions': []
        }
        
        # Log top-k predictions
        for j in range(top_k):
            class_idx = top_indices[i, j].item()
            prob = top_probs[i, j].item()
            class_name = class_names.get(class_idx, f"class_{class_idx}")
            
            pred_info = {
                'rank': j + 1,
                'class_idx': class_idx,
                'class_name': class_name,
                'probability': prob
            }
            
            sample_result['predictions'].append(pred_info)
            
            logging.info(f"  Top-{j+1}: {class_name} (ID: {class_idx}) - {prob:.4f} ({prob*100:.2f}%)")
        
        # Calculate entropy (measure of uncertainty)
        entropy = -torch.sum(all_probs[i] * torch.log(all_probs[i] + 1e-10)).item()
        sample_result['entropy'] = entropy
        logging.info(f"  Prediction Entropy: {entropy:.4f}")
        
        # Top prediction confidence
        max_prob = top_probs[i, 0].item()
        sample_result['confidence'] = max_prob
        logging.info(f"  Top-1 Confidence: {max_prob:.4f} ({max_prob*100:.2f}%)")
        
        results.append(sample_result)
    
    return results


def test_resnet18_with_detailed_analysis():
    """Test PGD attack on ResNet18 with detailed class probability analysis."""
    log_file = setup_logging()
    
    logging.info("=" * 80)
    logging.info("DETAILED PGD ATTACK ANALYSIS ON RESNET18")
    logging.info("=" * 80)
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f"Using device: {device}")
    
    # Load ResNet18 model
    logging.info("Loading ResNet18...")
    model, model_info = load_pretrained_model('resnet18', dataset='imagenet', device=device)
    logging.info(f"Model info: {model_info}")
    
    # Load CIFAR-10 test data
    transform = transforms.Compose([
        transforms.Resize(224),  # Resize CIFAR-10 to ImageNet size
        transforms.ToTensor(),
    ])
    
    test_dataset = torchvision.datasets.CIFAR10(
        root='./data', 
        train=False, 
        download=True, 
        transform=transform
    )
    
    # Get a small batch for detailed analysis
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=5, shuffle=True)
    images, cifar_labels = next(iter(test_loader))
    
    # Move to device
    images = images.to(device)
    cifar_labels = cifar_labels.to(device)
    
    logging.info(f"Test batch shape: {images.shape}")
    logging.info(f"Image range: [{images.min().item():.3f}, {images.max().item():.3f}]")
    logging.info(f"CIFAR-10 labels: {cifar_labels.tolist()}")
    
    # CIFAR-10 class names
    cifar_classes = ['airplane', 'automobile', 'bird', 'cat', 'deer', 
                     'dog', 'frog', 'horse', 'ship', 'truck']
    
    for i, label in enumerate(cifar_labels):
        logging.info(f"CIFAR-10 Sample {i+1}: {cifar_classes[label.item()]}")
    
    # Analyze original predictions
    logging.info("\n" + "="*60)
    logging.info("ANALYZING ORIGINAL IMAGES")
    logging.info("="*60)
    
    original_results = analyze_predictions(model, images, None, "Original Images", top_k=10)
    
    # PGD attack parameters
    epsilon = 0.03
    alpha = 0.001
    iterations = 40
    
    logging.info(f"\n" + "="*60)
    logging.info(f"RUNNING PGD ATTACK")
    logging.info(f"Parameters: epsilon={epsilon}, alpha={alpha}, iterations={iterations}")
    logging.info("="*60)
    
    # Run PGD attack
    adv_images, perturbations = pgd_attack(
        model=model,
        images=images,
        labels=torch.zeros_like(cifar_labels),  # Dummy labels for untargeted attack
        epsilon=epsilon,
        alpha=alpha,
        iterations=iterations,
        random_start=True,
        device=device
    )
    
    # Log perturbation statistics
    logging.info(f"\nPerturbation Statistics:")
    logging.info(f"  Shape: {perturbations.shape}")
    logging.info(f"  Range: [{perturbations.min().item():.6f}, {perturbations.max().item():.6f}]")
    logging.info(f"  Max L∞ norm: {perturbations.abs().max().item():.6f}")
    logging.info(f"  Mean L2 norm: {perturbations.view(perturbations.shape[0], -1).norm(2, dim=1).mean().item():.4f}")
    
    # Analyze adversarial predictions
    logging.info("\n" + "="*60)
    logging.info("ANALYZING ADVERSARIAL IMAGES")
    logging.info("="*60)
    
    adversarial_results = analyze_predictions(model, adv_images, None, "Adversarial Images", top_k=10)
    
    # Compare predictions
    logging.info("\n" + "="*60)
    logging.info("PREDICTION COMPARISON ANALYSIS")
    logging.info("="*60)
    
    attack_success_count = 0
    
    for i in range(len(original_results)):
        logging.info(f"\nSample {i+1} Comparison:")
        
        orig_top1 = original_results[i]['predictions'][0]
        adv_top1 = adversarial_results[i]['predictions'][0]
        
        logging.info(f"  Original Top-1: {orig_top1['class_name']} ({orig_top1['probability']:.4f})")
        logging.info(f"  Adversarial Top-1: {adv_top1['class_name']} ({adv_top1['probability']:.4f})")
        
        if orig_top1['class_idx'] != adv_top1['class_idx']:
            logging.info(f"  ✅ ATTACK SUCCESS: Prediction changed!")
            attack_success_count += 1
        else:
            logging.info(f"  ❌ Attack failed: Prediction unchanged")
        
        # Confidence change
        conf_change = adv_top1['probability'] - orig_top1['probability']
        logging.info(f"  Confidence change: {conf_change:+.4f}")
        
        # Entropy change
        entropy_change = adversarial_results[i]['entropy'] - original_results[i]['entropy']
        logging.info(f"  Entropy change: {entropy_change:+.4f}")
    
    attack_success_rate = attack_success_count / len(original_results)
    logging.info(f"\nOverall Attack Success Rate: {attack_success_rate:.2%} ({attack_success_count}/{len(original_results)})")
    
    # Evaluate attack with built-in metrics
    logging.info("\n" + "="*60)
    logging.info("TECHNICAL VALIDATION")
    logging.info("="*60)
    
    metrics = evaluate_attack(
        model=model,
        original_images=images,
        adversarial_images=adv_images,
        true_labels=torch.zeros_like(cifar_labels),  # Dummy labels
        device=device
    )
    
    logging.info(f"Technical Metrics:")
    logging.info(f"  Mean L2 perturbation: {metrics['mean_perturbation_l2']:.4f}")
    logging.info(f"  Max L∞ perturbation: {metrics['max_perturbation_linf']:.6f}")
    logging.info(f"  Epsilon constraint: {'✅ SATISFIED' if metrics['max_perturbation_linf'] <= epsilon + 1e-6 else '❌ VIOLATED'}")
    logging.info(f"  Image range: {'✅ VALID [0,1]' if torch.all(adv_images >= 0) and torch.all(adv_images <= 1) else '❌ INVALID'}")
    
    # Prepare model info for JSON serialization
    model_info_json = dict(model_info)
    model_info_json['device'] = str(model_info_json['device'])
    
    # Save detailed results
    results_data = {
        'timestamp': datetime.now().isoformat(),
        'device': str(device),
        'model_info': model_info_json,
        'attack_params': {
            'epsilon': epsilon,
            'alpha': alpha,
            'iterations': iterations
        },
        'original_predictions': original_results,
        'adversarial_predictions': adversarial_results,
        'attack_success_rate': attack_success_rate,
        'technical_metrics': metrics
    }
    
    results_file = f'results/logs/detailed_results_{datetime.now().strftime("%Y%m%d_%H%M%S")}.json'
    with open(results_file, 'w') as f:
        json.dump(results_data, f, indent=2)
    
    logging.info(f"\nDetailed results saved to: {results_file}")
    logging.info(f"Log file saved to: {log_file}")
    
    # Create visualization
    create_detailed_visualization(images, adv_images, perturbations, 
                                original_results, adversarial_results)
    
    logging.info("\n" + "="*80)
    logging.info("🎉 DETAILED ANALYSIS COMPLETED!")
    logging.info("="*80)
    
    return results_data


def create_detailed_visualization(images, adv_images, perturbations, 
                                orig_results, adv_results):
    """Create detailed visualization with class predictions."""
    
    logging.info("Creating detailed visualization...")
    
    num_samples = min(3, images.shape[0])
    fig, axes = plt.subplots(3, num_samples, figsize=(15, 12))
    
    for i in range(num_samples):
        # Original image
        img_orig = images[i].cpu().permute(1, 2, 0).numpy()
        img_orig = np.clip(img_orig, 0, 1)
        
        axes[0, i].imshow(img_orig)
        orig_pred = orig_results[i]['predictions'][0]
        axes[0, i].set_title(f"Original\n{orig_pred['class_name']}\n{orig_pred['probability']:.3f}", 
                           fontsize=10)
        axes[0, i].axis('off')
        
        # Adversarial image
        img_adv = adv_images[i].cpu().permute(1, 2, 0).numpy()
        img_adv = np.clip(img_adv, 0, 1)
        
        axes[1, i].imshow(img_adv)
        adv_pred = adv_results[i]['predictions'][0]
        color = 'red' if orig_pred['class_idx'] != adv_pred['class_idx'] else 'black'
        axes[1, i].set_title(f"Adversarial\n{adv_pred['class_name']}\n{adv_pred['probability']:.3f}", 
                           fontsize=10, color=color)
        axes[1, i].axis('off')
        
        # Perturbation
        pert = perturbations[i].cpu().permute(1, 2, 0).numpy()
        pert_norm = (pert - pert.min()) / (pert.max() - pert.min() + 1e-8)
        
        axes[2, i].imshow(pert_norm)
        axes[2, i].set_title(f"Perturbation\nL∞={np.abs(pert).max():.4f}", fontsize=10)
        axes[2, i].axis('off')
    
    plt.tight_layout()
    
    # Save visualization
    vis_file = f'results/figures/detailed_analysis_{datetime.now().strftime("%Y%m%d_%H%M%S")}.png'
    fig.savefig(vis_file, dpi=150, bbox_inches='tight')
    plt.close()
    
    logging.info(f"Detailed visualization saved to: {vis_file}")


def main():
    """Run detailed analysis."""
    try:
        results = test_resnet18_with_detailed_analysis()
        return True
    except Exception as e:
        logging.error(f"Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    success = main()
    if success:
        print("\n✅ Detailed ResNet18 analysis completed successfully!")
    else:
        print("\n❌ Detailed ResNet18 analysis failed!")
        sys.exit(1)