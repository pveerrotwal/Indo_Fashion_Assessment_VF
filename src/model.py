import torch.nn as nn
import timm

from src.utils import count_parameters


def freeze_backbone(model: nn.Module) -> None:
    for name, param in model.named_parameters():
        if "classifier" not in name and "head" not in name and "fc" not in name:
            param.requires_grad = False


def unfreeze_backbone(model: nn.Module) -> None:
    for param in model.parameters():
        param.requires_grad = True


def _replace_classifier(model: nn.Module, num_classes: int) -> None:
    # timm returns different attribute names across versions, handle both
    if hasattr(model, "classifier") and isinstance(model.classifier, nn.Module):
        if isinstance(model.classifier, nn.Sequential):
            in_features = model.classifier[-1].in_features
        else:
            in_features = model.classifier.in_features
        model.classifier = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        return

    if hasattr(model, "head") and isinstance(model.head, nn.Module):
        in_features = model.head.in_features
        model.head = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        return

    if hasattr(model, "fc") and isinstance(model.fc, nn.Module):
        in_features = model.fc.in_features
        model.fc = nn.Sequential(nn.Dropout(0.3), nn.Linear(in_features, num_classes))
        return

    raise AttributeError("Unable to locate classifier head on model.")


def get_model(config) -> nn.Module:
    model = timm.create_model(config.MODEL_NAME, pretrained=config.PRETRAINED)
    _replace_classifier(model, config.NUM_CLASSES)
    freeze_backbone(model)
    count_parameters(model)
    return model
