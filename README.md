# skull_stripping
The goal of the project was to prepare a skull stripping method on images from the T1 sequence of head MRI. The method should extract the whole area covered by the brain, skipping bones, other soft tissues, etc. 
![Example](https://www.researchgate.net/profile/Dario_Pompili/publication/309402865/figure/fig1/AS:420915604148224@1477365508110/Skull-stripping-steps-A-input-images-B-brain-contouring-and-C-removal-of.png)

## Dataset
The dataset was shared by a lecturer. It consisted of:
- 674 MRI labeled, train scans in [NIfTI-1 Data Format](https://nifti.nimh.nih.gov/nifti-1/)
- 97 MRI test scans in NIfTI-1 Data Format. 

## Project structure:
### [tf_implementation](https://github.com/arekmula/skull_stripping/tree/tensorflow_implementation/tf_implementation) - Implementation in Tensorflow 
- Requirements listed in `environment.yml`
- Segmentation model in `segmentation` folder
    *  `dataset` folder -> all functions related directly to dataset (Train and validation ImageGenerator etc.)
    *  `losses` folder -> Dice Loss
    *  `metrics` folder -> F1Score (Dice Score)
    *  `models` folder -> Unet model with efficientnetb0 backbone
    *  `utils` folder -> display utilities
- Training code -> `train.py`
- Training code for Google Colab -> `train.ipynb`

### pytorch - TODO


## Results
- TODO
