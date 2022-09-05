# Tuberculosis detection using Chest X-RAY images in PyTorch
**AlexNet** CNN as the base model is used to classify CXR images as _**Normal / Tuberculosis_infected**_ through **Transfer learning**.

## Dataset
**Data Source:** _https://www.kaggle.com/datasets/tawsifurrahman/tuberculosis-tb-chest-xray-dataset_
- **Total File Size:** 700MB
- **Number of records:** 3500 Normal, 700 Tuberculosis (Separated into 2 _(3600 train and 600 validation)_ parts)
- Images are size of **(512 * 512 * 3)** and in **_.jpg_** format.   

<img src=https://user-images.githubusercontent.com/56191102/188506204-3aeb040b-7578-4577-8cb0-f9f5909d5e93.png alt="Normal" style="height: 150px; width:150px;"/> <img src=https://user-images.githubusercontent.com/56191102/188506363-15435994-6f6f-41e8-a1fe-d4ee3c7528e9.png alt="Normal" style="height: 150px; width:150px;"/>
 

## Network Architecture
 ```
 ===============================================================
 Layer (type)               Output Shape         Param 
 ===============================================================
            Conv2d-1         [18, 64, 127, 127]          23,296
              ReLU-2         [18, 64, 127, 127]               0
         MaxPool2d-3           [18, 64, 63, 63]               0
            Conv2d-4          [18, 192, 63, 63]         307,392
              ReLU-5          [18, 192, 63, 63]               0
         MaxPool2d-6          [18, 192, 31, 31]               0
            Conv2d-7          [18, 384, 31, 31]         663,936
              ReLU-8          [18, 384, 31, 31]               0
            Conv2d-9          [18, 256, 31, 31]         884,992
             ReLU-10          [18, 256, 31, 31]               0
           Conv2d-11          [18, 256, 31, 31]         590,080
             ReLU-12          [18, 256, 31, 31]               0
        MaxPool2d-13          [18, 256, 15, 15]               0
AdaptiveAvgPool2d-14          [18, 256, 15, 15]               0
          Dropout-15                [18, 57600]               0
           Linear-16                  [18, 510]      29,376,510
             ReLU-17                  [18, 510]               0
           Linear-18                    [18, 1]             511
================================================================
Total params: 31,846,717
Trainable params: 29,377,021
Non-trainable params: 2,469,696
```
- network uses pretrained weights and during the training, _CNN weights_ are freezed and training only affects the _Linear layers_.

## Training
- For solving the problem of imbalance dataset, **weighted classes** used alongside with **cross entropy loss**

## Accuracy 
```
Validation loss: 0.019457, Validation acc: 99.333333,
for class Normal:
 validation precision: 0.994024, validation recall: 0.998000 , validation F1: 0.996008
for class Tuberculosis:
 validation precision: 0.989796, validation recall: 0.970000 , validation F1: 0.979798
 ```
 Confusion matrix:
 <img src=https://user-images.githubusercontent.com/56191102/188505966-9353a90b-5aa2-4d24-ab27-2f1f2627b193.png alt="confusion matrix for validation data" style="height: 350px; "/>
 <img src=https://user-images.githubusercontent.com/56191102/188507451-877db3d8-25eb-4803-a7b9-ff716dbc597c.png alt="accuracy diagram" style="height: 350px; "/>
 <img src=https://user-images.githubusercontent.com/56191102/188507510-76010fdd-13ce-4173-af69-be1706a963cc.png alt="loss diagram" style="height: 350px; "/>
 


