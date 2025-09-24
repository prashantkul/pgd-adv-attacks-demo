"""
Model loading utilities for adversarial attack experiments.

This module provides functions to load pretrained models from torchvision
and prepare them for adversarial attacks.
"""

import torch
import torch.nn as nn
import torchvision.models as models
import torchvision.transforms as transforms
from typing import Optional, Tuple, Dict, Any


# ImageNet normalization parameters
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# CIFAR-10 normalization parameters  
CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]


class NormalizedModel(nn.Module):
    """
    Wrapper to add normalization layer to a model.
    This ensures inputs are properly normalized for pretrained models.
    """
    
    def __init__(self, model: nn.Module, mean: list, std: list):
        """
        Args:
            model: The base model to wrap
            mean: Mean values for normalization
            std: Standard deviation values for normalization
        """
        super(NormalizedModel, self).__init__()
        self.model = model
        self.mean = torch.tensor(mean).view(1, 3, 1, 1)
        self.std = torch.tensor(std).view(1, 3, 1, 1)
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass with normalization.
        
        Args:
            x: Input tensor (expected to be in [0, 1] range)
        
        Returns:
            Model output
        """
        # Move normalization parameters to the same device as input
        if self.mean.device != x.device:
            self.mean = self.mean.to(x.device)
            self.std = self.std.to(x.device)
        
        # Normalize the input
        x_normalized = (x - self.mean) / self.std
        return self.model(x_normalized)


def load_pretrained_model(
    model_name: str = 'resnet18',
    dataset: str = 'imagenet',
    device: Optional[torch.device] = None,
    normalize: bool = True
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a pretrained model from torchvision.
    
    Args:
        model_name: Name of the model to load
        dataset: Dataset the model was trained on ('imagenet' or 'cifar10')
        device: Device to load the model on
        normalize: Whether to wrap model with normalization layer
    
    Returns:
        Tuple of (model, info_dict) where info_dict contains:
            - num_classes: Number of output classes
            - input_size: Expected input size
            - mean: Normalization mean
            - std: Normalization standard deviation
    
    Raises:
        ValueError: If model_name is not supported
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Model loading dictionary
    model_dict = {
        # ResNet family
        'resnet18': models.resnet18,
        'resnet34': models.resnet34,
        'resnet50': models.resnet50,
        'resnet101': models.resnet101,
        'resnet152': models.resnet152,
        
        # VGG family
        'vgg11': models.vgg11,
        'vgg13': models.vgg13,
        'vgg16': models.vgg16,
        'vgg19': models.vgg19,
        'vgg11_bn': models.vgg11_bn,
        'vgg13_bn': models.vgg13_bn,
        'vgg16_bn': models.vgg16_bn,
        'vgg19_bn': models.vgg19_bn,
        
        # DenseNet family
        'densenet121': models.densenet121,
        'densenet169': models.densenet169,
        'densenet201': models.densenet201,
        'densenet161': models.densenet161,
        
        # Other architectures
        'alexnet': models.alexnet,
        'squeezenet1_0': models.squeezenet1_0,
        'squeezenet1_1': models.squeezenet1_1,
        'inception_v3': models.inception_v3,
        'googlenet': models.googlenet,
        'mobilenet_v2': models.mobilenet_v2,
        'mobilenet_v3_small': models.mobilenet_v3_small,
        'mobilenet_v3_large': models.mobilenet_v3_large,
        'efficientnet_b0': models.efficientnet_b0,
        'efficientnet_b1': models.efficientnet_b1,
        'efficientnet_b2': models.efficientnet_b2,
        'efficientnet_b3': models.efficientnet_b3,
        'efficientnet_b4': models.efficientnet_b4,
    }
    
    if model_name not in model_dict:
        raise ValueError(f"Model {model_name} not supported. Available models: {list(model_dict.keys())}")
    
    # Load the model
    print(f"Loading {model_name} pretrained on {dataset}...")
    
    if dataset == 'imagenet':
        model = model_dict[model_name](pretrained=True)
        mean = IMAGENET_MEAN
        std = IMAGENET_STD
        num_classes = 1000
        input_size = (224, 224) if model_name != 'inception_v3' else (299, 299)
    elif dataset == 'cifar10':
        # For CIFAR-10, we need to modify the model architecture
        model = model_dict[model_name](pretrained=False)
        
        # Adjust the final layer for CIFAR-10 (10 classes)
        if 'resnet' in model_name:
            model.fc = nn.Linear(model.fc.in_features, 10)
        elif 'vgg' in model_name:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
        elif 'densenet' in model_name:
            model.classifier = nn.Linear(model.classifier.in_features, 10)
        elif 'mobilenet' in model_name:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
        elif 'efficientnet' in model_name:
            model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, 10)
        
        mean = CIFAR10_MEAN
        std = CIFAR10_STD
        num_classes = 10
        input_size = (32, 32)
    else:
        raise ValueError(f"Dataset {dataset} not supported. Use 'imagenet' or 'cifar10'.")
    
    # Wrap with normalization if requested
    if normalize:
        model = NormalizedModel(model, mean, std)
    
    # Move model to device and set to eval mode
    model = model.to(device)
    model.eval()
    
    # Prepare info dictionary
    info = {
        'num_classes': num_classes,
        'input_size': input_size,
        'mean': mean,
        'std': std,
        'device': device,
        'model_name': model_name,
        'dataset': dataset
    }
    
    print(f"Model loaded successfully on {device}")
    return model, info


def load_multiple_models(
    model_names: list,
    dataset: str = 'imagenet',
    device: Optional[torch.device] = None,
    normalize: bool = True
) -> Dict[str, Tuple[nn.Module, Dict[str, Any]]]:
    """
    Load multiple pretrained models.
    
    Args:
        model_names: List of model names to load
        dataset: Dataset the models were trained on
        device: Device to load the models on
        normalize: Whether to wrap models with normalization layer
    
    Returns:
        Dictionary mapping model names to (model, info) tuples
    """
    models_dict = {}
    
    for name in model_names:
        try:
            model, info = load_pretrained_model(name, dataset, device, normalize)
            models_dict[name] = (model, info)
        except Exception as e:
            print(f"Failed to load {name}: {e}")
    
    return models_dict


def get_data_transforms(dataset: str = 'imagenet', augment: bool = False) -> transforms.Compose:
    """
    Get data transformation pipeline for a specific dataset.
    
    Args:
        dataset: Dataset name ('imagenet' or 'cifar10')
        augment: Whether to include data augmentation
    
    Returns:
        Composition of transforms
    """
    if dataset == 'imagenet':
        if augment:
            transform = transforms.Compose([
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
            ])
    elif dataset == 'cifar10':
        if augment:
            transform = transforms.Compose([
                transforms.RandomCrop(32, padding=4),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
            ])
        else:
            transform = transforms.Compose([
                transforms.ToTensor(),
            ])
    else:
        raise ValueError(f"Dataset {dataset} not supported")
    
    return transform


def load_checkpoint(
    checkpoint_path: str,
    model_name: str = 'resnet18',
    dataset: str = 'imagenet',
    device: Optional[torch.device] = None
) -> Tuple[nn.Module, Dict[str, Any]]:
    """
    Load a model from a checkpoint file.
    
    Args:
        checkpoint_path: Path to the checkpoint file
        model_name: Name of the model architecture
        dataset: Dataset the model was trained on
        device: Device to load the model on
    
    Returns:
        Tuple of (model, checkpoint_dict)
    """
    if device is None:
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load the base model architecture
    model, info = load_pretrained_model(model_name, dataset, device, normalize=False)
    
    # Load the checkpoint
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    # Load state dict
    if 'state_dict' in checkpoint:
        model.load_state_dict(checkpoint['state_dict'])
    elif 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()
    
    return model, checkpoint


def test_model(model: nn.Module, dataloader: torch.utils.data.DataLoader, 
                device: Optional[torch.device] = None) -> float:
    """
    Test a model's accuracy on a dataloader.
    
    Args:
        model: The model to test
        dataloader: DataLoader for the test set
        device: Device to run the test on
    
    Returns:
        Accuracy as a float between 0 and 1
    """
    if device is None:
        device = next(model.parameters()).device
    
    model.eval()
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    accuracy = correct / total
    return accuracy


# ImageNet class names (first 10 for demo)
IMAGENET_CLASSES = [
    'tench', 'goldfish', 'great white shark', 'tiger shark', 'hammerhead',
    'electric ray', 'stingray', 'cock', 'hen', 'unknown_bird'
]

# CIFAR-10 class names
CIFAR10_CLASSES = [
    'airplane', 'automobile', 'bird', 'cat', 'deer',
    'dog', 'frog', 'horse', 'ship', 'truck'
]


def get_class_names(dataset: str = 'imagenet') -> list:
    """
    Get class names for a dataset.
    
    Args:
        dataset: Dataset name
    
    Returns:
        List of class names
    """
    if dataset == 'imagenet':
        # Load full ImageNet class names if available
        try:
            import json
            with open('imagenet_classes.json', 'r') as f:
                return json.load(f)
        except:
            return IMAGENET_CLASSES  # Return sample classes
    elif dataset == 'cifar10':
        return CIFAR10_CLASSES
    else:
        raise ValueError(f"Dataset {dataset} not supported")


# Convenient wrapper functions for backward compatibility
def load_resnet18(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """
    Load ResNet18 model.
    
    Args:
        device: Device to load the model on
        dataset: Dataset the model was trained on ('imagenet' or 'cifar10')
        normalize: Whether to wrap model with normalization layer
    
    Returns:
        ResNet18 model
    """
    model, _ = load_pretrained_model('resnet18', dataset, device, normalize)
    return model


def load_resnet34(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """Load ResNet34 model."""
    model, _ = load_pretrained_model('resnet34', dataset, device, normalize)
    return model


def load_resnet50(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """Load ResNet50 model."""
    model, _ = load_pretrained_model('resnet50', dataset, device, normalize)
    return model


def load_vgg16(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """
    Load VGG16 model.
    
    Args:
        device: Device to load the model on
        dataset: Dataset the model was trained on ('imagenet' or 'cifar10')
        normalize: Whether to wrap model with normalization layer
    
    Returns:
        VGG16 model
    """
    model, _ = load_pretrained_model('vgg16', dataset, device, normalize)
    return model


def load_vgg19(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """Load VGG19 model."""
    model, _ = load_pretrained_model('vgg19', dataset, device, normalize)
    return model


def load_densenet121(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """
    Load DenseNet121 model.
    
    Args:
        device: Device to load the model on
        dataset: Dataset the model was trained on ('imagenet' or 'cifar10')
        normalize: Whether to wrap model with normalization layer
    
    Returns:
        DenseNet121 model
    """
    model, _ = load_pretrained_model('densenet121', dataset, device, normalize)
    return model


def load_densenet169(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """Load DenseNet169 model."""
    model, _ = load_pretrained_model('densenet169', dataset, device, normalize)
    return model


def load_alexnet(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """Load AlexNet model."""
    model, _ = load_pretrained_model('alexnet', dataset, device, normalize)
    return model


def load_mobilenet_v2(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """Load MobileNet V2 model."""
    model, _ = load_pretrained_model('mobilenet_v2', dataset, device, normalize)
    return model


def load_efficientnet_b0(device: Optional[torch.device] = None, dataset: str = 'imagenet', normalize: bool = True) -> nn.Module:
    """Load EfficientNet B0 model."""
    model, _ = load_pretrained_model('efficientnet_b0', dataset, device, normalize)
    return model