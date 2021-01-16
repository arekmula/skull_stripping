from segmentation_models.losses import dice_loss as sm_dice_loss


def dice_loss(y_true, y_pred):
    return sm_dice_loss(y_true, y_pred)
