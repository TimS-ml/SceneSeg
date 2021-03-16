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

Movie Scene Segmentation based on CVPR2020 A Local-to-Global Approach to Multi-modal Movie Scene Segmentation

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
AP: 0.444
mAP: 0.450
Miou:  0.47563575360370447
Recall:  0.7438605361117098
```

64 videos and `Place` feature only
```
AP: 0.401
mAP: 0.405
Miou:  0.46523629842773173
Recall:  0.5669334164159875
```


# Preparation
Please refer to [Install guide from Original SceneSeg repo](https://github.com/AnyiRao/SceneSeg/blob/master/docs/INSTALL.md)


# Model Structure

