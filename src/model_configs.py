from transformers import ViTForImageClassification, ResNetForImageClassification, EfficientNetForImageClassification


MEAN = [0.485, 0.456, 0.406]
STD = [0.229, 0.224, 0.225]

# ------ MODEL CONFIGURATIONS ------
MODEL_CONFIGS = {
    "vit-large-patch16-224": {
        "model_class": ViTForImageClassification,
        "model_name": "google/vit-large-patch16-224",
        "input_size": (224, 224),
        "mean": MEAN,
        "std": STD,
        "description": "ViT-L/16"
    },
    "vit-base-patch16-224": {
        "model_class": ViTForImageClassification,
        "model_name": "google/vit-base-patch16-224",
        "input_size": (224, 224),
        "mean": MEAN,
        "std": STD,
        "description": "ViT-B/16"
    },
    "resnet50": {
        "model_class": ResNetForImageClassification,
        "model_name": "microsoft/resnet-50",
        "input_size": (224, 224),
        "mean": [0.485, 0.456, 0.406],
        "std": [0.229, 0.224, 0.225],
        "description": "ResNet-50"
    },
    "resnet101": {
        "model_class": ResNetForImageClassification,
        "model_name": "microsoft/resnet-101",
        "input_size": (224, 224),
        "mean": MEAN,
        "std": STD,
        "description": "ResNet-101"
    },
    "resnet152": {
        "model_class": ResNetForImageClassification,
        "model_name": "microsoft/resnet-152",
        "input_size": (224, 224),
        "mean": MEAN,
        "std": STD,
        "description": "ResNet-152"
    },
    "efficientnet-b0": {
        "model_class": EfficientNetForImageClassification,
        "model_name": "google/efficientnet-b0",
        "input_size": (224, 224),
        "mean": MEAN,
        "std": STD,
        "description": "EfficientNet B0"
    },
    "efficientnet-b1": {
        "model_class": EfficientNetForImageClassification,
        "model_name": "google/efficientnet-b1",
        "input_size": (240, 240),
        "mean": MEAN,
        "std": STD,
        "description": "EfficientNet B1"
    },
    "efficientnet-b2": {
        "model_class": EfficientNetForImageClassification,
        "model_name": "google/efficientnet-b2",
        "input_size": (260, 260),
        "mean": MEAN,
        "std": STD,
        "description": "EfficientNet B2"
    },
    "efficientnet-b3": {
        "model_class": EfficientNetForImageClassification,
        "model_name": "google/efficientnet-b3",
        "input_size": (300, 300),
        "mean": MEAN,
        "std": STD,
        "description": "EfficientNet B3"
    },
    "efficientnet-b4": {
        "model_class": EfficientNetForImageClassification,
        "model_name": "google/efficientnet-b4",
        "input_size": (380, 380),
        "mean": MEAN,
        "std": STD,
        "description": "EfficientNet B4"
    }
}