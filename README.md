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
- Generating predictions code -> `generate_predictions.py`
- Testing predictions code (Sending predictions to lecturer's server) -> `test_predictions.py`

### pytorch - TODO


## Model
The model trained by me is available [here](https://drive.google.com/drive/folders/1T52d_eHSe14PpG0ocJ2Iiglz9ndAxH04?usp=sharing). It's Unet based model with EfficientNetB0 backbone. 

### Training process:
- split scans to train and validation sets
- generate image and label slices from X axis of each scan and save them to separate `images` and `labels` directories
- create combined generator for images and labels for both train and validation set. Each image is preprocessed according to EfficientNetB0's rules and resized to target size -> 256,256
- compile model with loss=DiceLoss and metrics=DiceScore. Define callbacks to model: EarlyStopping, ModelCheckpoint and Reduce Learning Rate On Plateau.
- fit model using data from training generator and validation generator


## Results
- Evaluation set Dice Score: 0.9875
- Test set Dice Score: 0.9866
