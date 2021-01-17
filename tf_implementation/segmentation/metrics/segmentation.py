import segmentation_models as sm


def f_score(threshold=0.5):
    return sm.metrics.FScore(threshold=threshold)
