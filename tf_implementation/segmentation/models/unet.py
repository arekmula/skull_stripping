from segmentation_models import Unet


def get_unet_model(backbone="efficientnetb0", classes=1, activation="sigmoid", encoder_weights="imagenet"):
    """
    Returns Unet model based on input variables

    :param backbone: Unet backbone
    :param classes: Number of classes
    :param activation: Activation function
    :param encoder_weights: Encoder weights
    :return model: Model based on input variables
    """
    model = Unet(backbone, classes=classes, activation=activation, encoder_weights=encoder_weights)

    return model
