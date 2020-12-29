from tensorflow.keras import backend as K


def dice_coef(y_true, y_pred, smooth=1):
    """
    Dice coefficient implementation
    https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2

    :param y_true: Ground truth labels
    :param y_pred: Predicted labels
    :param smooth:
    :return: Dice score
    """
    intersection = K.sum(y_true * y_pred, axis=[1, 2, 3])
    union = K.sum(y_true, axis=[1, 2, 3]) + K.sum(y_pred, axis=[1, 2, 3])
    dice = K.mean((2. * intersection + smooth) / (union + smooth), axis=0)
    return dice


def dice_coef_loss(y_true, y_pred):
    return 1-dice_coef(y_true, y_pred)
