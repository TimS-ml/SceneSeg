```
 .d888b, d8888b  88bd88b  d8888b d8888b .d888b, d8888b d888b8b  
 ?8b,   d8b_,dP  88P' ?8bd8P' `Pd8b_,dP ?8b,   d8b_,dPd8P' ?88  
   `?8b 88b     d88   88P88b    88b       `?8b 88b    88b  ,88b 
`?888P' `?888P'd88'   88b`?888P'`?888P'`?888P' `?888P'`?88P'`88b
                                                             )88
                                                            ,88P
                                                        `?8888P 
```

# Intruduction

Movie Scene Segmentation based on CVPR2020 (A Local-to-Global Approach to Multi-modal Movie Scene Segmentation)

[A Local-to-Global Approach to
Multi-modal Movie Scene Segmentation](https://anyirao.com/projects/SceneSeg.html)

[Code of Original SceneSeg](https://github.com/AnyiRao/SceneSeg)

[MovieNet](http://movienet.site/)

[MovieNet-tools](https://github.com/movienet/movienet-tools)

[MovieNet-tools-doc](http://docs.movienet.site/movie-toolbox/tools#get-started)


# Evaluation

We take two commonly used metrics:
1. mAP -- the mean Average Precision of scene transition predictions for each movie.
2. Miou -- for each ground-truth scene, we take the maximum intersection-over-union with the detected scenes, averaging them on the whole video. Then the same is done for detected scenes against ground-truth scenes, and the two quantities are again averaged. The intersection/union is evaluated by counting the frames.

Using pre-trained model without `Audio`
```
AP:     0.444
mAP:    0.450
Miou:   0.47563575360370447
Recall: 0.7438605361117098
```

Trained on 56 videos and `Place` feature only
```
AP:     0.401
mAP:    0.405
Miou:   0.46523629842773173
Recall: 0.5669334164159875
```


# Preparation and Usage
Please refer to [Install guide from original SceneSeg repo](https://github.com/AnyiRao/SceneSeg/blob/master/docs/INSTALL.md)


# More About that Paper
__Feature Extraction__
Feature extracting already been aggregrated in [MovieNet-tools](https://github.com/movienet/movienet-tools) (yehhhh), the source code is worth reading, it's not the focuse of this repo through
Here are how features are extracted:
- Place
  - ResNet50
- Cast
  - Faster-RCNN on CIM dataset to detect
  - ResNet50 on PIPA to extract
- Action
  - TSN on AVA dataset
- Audio
  - NaverNet on AVA-ActiveSpeaker dataset to separate speech
  - stft to get features repectively in a shot

__BNet (Boundary Net)__
Data: 2 * w_b shots => before and after boundary
Two parts
- B_d: d stands for 'difference'
  - 2 * convolution layers: before and after shot + inner product operation to calculate differences
- B_r: r stands for 'relationship' 
  - 1 * convolution layer + max pooling

__After BNet: Coarse Prediction at Segment Level__
Next step is to predicting a sequence binary
Use w_t shots each time to avoid memory leakage
- seq to seq model: Bi-LSTM
  - stride w_t / 2 shots
  - return a coarse score: probability of a shot boundary to be a scene boundary
- coarse prediction: 
  - binarizing coarse score (which is a list) with a threshold t 

__LGSS (Local-to-Global Scene Segmentation)__
Get Coarse Predictions separately from different features and sum them up (so disappointing)
