import segmentation_models_pytorch as smp
import torch
import torch.nn as nn

# model factory for segmentation models
def create_model(architecture= "unet",encoder= "resnet34", encoder_weights= "imagenet",in_channels= 3, classes= 1):
    
    architecture_map = {"unet": smp.Unet, "unetplusplus": smp.UnetPlusPlus, "deeplabv3plus": smp.DeepLabV3Plus,}
    
    if architecture not in architecture_map: raise ValueError(f"Unknown architecture {architecture}.....")
    
    model = architecture_map[architecture](
        encoder_name=encoder, encoder_weights=encoder_weights,
        in_channels=in_channels, classes=classes, activation=None,
)
    
    # print the summary of the model
    total_params = 0
    for param in model.parameters():
        total_params += param.numel()

    trainable_params = 0
    for param in model.parameters():
        if param.requires_grad:
            trainable_params += param.numel()
    
    print(f' Model {architecture} with encoder {encoder}')
    print(f'Total params: {total_params:,}, Trainable params: {trainable_params:,}')
    
    return model


# class for Dice Loss
class DiceLoss(nn.Module):
    def __init__(self, smooth=1.0):
        super().__init__()
        self.smooth = smooth
    
    def forward(self, logits, targets):
        probabilities = torch.sigmoid(logits)
        probabilities_flat = probabilities.reshape(probabilities.size(0), -1)
        targets_flat = targets.reshape(targets.size(0), -1)
        
        probs_intersection = (probabilities_flat * targets_flat).sum(dim=1)
        probs_union = probabilities_flat.sum(dim=1) + targets_flat.sum(dim=1)
        
        dice = (2.0 * probs_intersection + self.smooth) / (probs_union + self.smooth)
        dice_loss = 1.0 - dice.mean()
        
        return dice_loss    

# class for bce and dice loss
class BCEDiceLoss(nn.Module):
    def __init__(self, bce_weight=0.5, dice_weight=0.5):
        super().__init__()
        self.bce_weight = bce_weight
        self.dice_weight = dice_weight
        self.bce = nn.BCEWithLogitsLoss()
        self.dice = DiceLoss()
    
    def forward(self, logits, targets):
        bce_loss = self.bce(logits, targets)
        dice_loss = self.dice(logits, targets)
        return self.bce_weight * bce_loss + self.dice_weight * dice_loss


# create loss function
def create_loss(loss_type= "bce_dice", **kwargs):
    if loss_type == "bce_dice":
        return BCEDiceLoss( bce_weight=kwargs.get("bce_weight", 0.5),dice_weight=kwargs.get("dice_weight", 0.5),)
    elif loss_type == "dice":
        return DiceLoss()
    elif loss_type == "bce":
        return nn.BCEWithLogitsLoss()
    else:
        raise ValueError(f"Unknown loss {loss_type}")