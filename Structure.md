# More about LGSS (Local-to-Global Scene Segmentation)
## BNet (Boundary Net)
Data: 2 * w_b shots => before and after boundary
Two parts
- B_d: d stands for 'difference'
  - 2 * convolution layers before and after shot
  - + inner product operation to calculate their differences
- B_r: r stands for 'relationship' 
  - 1 * convolution layer
  - + max pooling

## After BNet: Coarse Prediction at Segment Level
Next step is to predicting a sequence binary
use w_t shots each time to reduce memory leakage
- seq to seq model: Bi-LSTM
  - stride w_t / 2 shots
  - return a coarse score: probability of a shot boundary to be a scene boundary
- coarse prediction: 
  - binarizing coarse score (which is a list) with a threshold t 


# KWs and Notes
ap: average precision (calculated by sklearn)
    check 5.1 in paper
gt: ground truth
gt0: where gt == 0
gt1: where gt == 1


# Train and Test
save state dict every epoch


# lgss
- `gen_csv.py`
  - generate dirs

## src
- `get_data.py`

### [x] models
- `lgss.py`
|- LGSS (output is the sumup of features)
   |- LGSSone * 4
      |- BNet (place, cast, act)
      |  |- Cos (t-conv)
      |- LSTM
      |
      |- BNet_aud (aud) 
      |  |- audnet 
      |- LSTM

- `lgss_image.py`
  - resnet feature extractor (only img, no cast etc.)
  - BNet:
    - Cos + Res feat_extractor

```python
def forward(self, x):  # [batch_size, seq_len, shot_num, 3, 224, 224]
    feat = self.feat_extractor(x)
    # [batch_size, seq_len, shot_num, feat_dim] 
    context = feat.view(
        feat.shape[0]*feat.shape[1], 1, feat.shape[-2], feat.shape[-1])
    context = self.conv1(context)
    # batch_size*seq_len,sim_channel,1,feat_dim
    context = self.max3d(context)
    # batch_size*seq_len,1,1,feat_dim
    context = context.squeeze()
    sim = self.cos(feat)
    bound = torch.cat((context, sim), dim=1)
    return bound
```

### data
- `all.py`  - imported by `get_data.py`
  - `meta/split318.json` 
- How feature loaded
  - `_get_single_item`


## utilis
- read_json, read_pkl, read_txt_list, strcal

### `save_pred_seq`

### `cal_MIOU`

### `get_ap` and `get_mAP`
they are independent
```python
gts_raw, preds_raw
return average_precision_score(np.nan_to_num(gts), np.nan_to_num(preds))

```



# data - intermediate features
for ./scene318

## [1] `label318` 
- scene transit (1) or not (0) label for each shot
- `movie.txt`
  - '0044 0'

## [2] `meta` 
- [2.1] `scene_movie318.json`
  ```
  "tt0047396": {      # movie
      "0": {          # scene
          "shot": [   # shots
              "0001",
              "0002",
              "0003"
          ],
          "frame": [
              "49",   # frame start
              "3482"  # frame end
          ]
      },
      "1": {
          "shot": [
              "0004"
          ],
          "frame": [
              "3483",
              "3805"
          ]
      },
  
  ```
- [2.2] `split318.json`
  - 'train':[xxx, xxx], 'test': ...

## [3] `shot_movie318` 
- (mapping shot and frame for val purpose) shot and frame correspondence to _recover the time of each scene_
- they will be automatically handled by the processing codes e.g., the `data_pre` function in `src/data/all.py`
  - 0 48 12 24 36 ...
- by `shot_frm_path`
  - in `run.py`, MIOU and Recall will be used only when `shot_frm_path` is setted

## [4] intermediate features `place_feat, cast_feat, act_feat, aud_feat`
    From `src/data/all.py`
    id is from json list and func: `data_partition`
    ```python
    name = 'shot_{}.npy'.format(strcal(shotid, ind))
    path = osp.join(self.data_root, 'place_feat/{}'.format(imdbid), name)
    place_feat = np.load(path)
    place_feats.append(torch.from_numpy(place_feat).float())
    ```
    - torch.Size([2430, 2048])  # place
    - torch.Size([2430, 512])   # cast
    - torch.Size([2430, 512])   # act
    - torch.Size([2430, 512])   # aud


## Functions
### cfg and features => `data_pre_one` => `data_pre`
- data_pre_one
  - place_feat and label318 needed
  - `get_anno_dict`

- data_pre 
  - read from `split318.json`
  - `data_pre_one`
  - `data_partition`

### `data_partition`
- input: imdbidlist_json, annos_valid_dict is from `data_pre`

### `get_anno_dict`
- input: anno_fn = ./data/label318/xxx.txt
- `read_txt_list` from utilis



# pre - for preprocess
## ShotDetect 
- `shotdetect.py` and `shotdetect_p.py` 

**A**
```python
from shotdetect.video_manager import VideoManager
from shotdetect.shot_manager import ShotManager
from shotdetect.stats_manager import StatsManager

from shotdetect.detectors.content_detector_hsv_luv import ContentDetectorHSVLUV

from shotdetect.video_splitter import is_ffmpeg_available,split_video_ffmpeg
from shotdetect.keyf_img_saver import generate_images,generate_images_txt
```

**B**
```python
video_manager = VideoManager([video_path])
stats_manager = StatsManager()
# Construct our shotManager and pass it our StatsManager.
shot_manager = ShotManager(stats_manager)

# Add ContentDetector algorithm (each detector's constructor
# takes detector options, e.g. threshold).
shot_manager.add_detector(ContentDetectorHSVLUV())
base_timecode = video_manager.get_base_timecode()
```

**C**
```python
# Start video_manager.
video_manager.start()

# Perform shot detection on video_manager.
shot_manager.detect_shots(frame_source=video_manager)

# Obtain list of detected shots.
shot_list = shot_manager.get_shot_list(base_timecode)
# Each shot is a tuple of (start, end) FrameTimecodes.
```

### shotdetect
**Main**
- `video_manager.py`
```python
from shotdetect.platform import STRING_TYPE
import shotdetect.frame_timecode
from shotdetect.frame_timecode import FrameTimecode
```

- `stats_manager.py`
  - add cuts

- `shot_manager.py`
```python
from shotdetect.stats_manager import FrameMetricRegistered
```
  - For caching detection metrics and saving/loading to a stats file

**Sub**
- `frame_timecode.py`
  - for ShotDetect to store frame-accurate timestamps of each cut
  - unit tests for the FrameTimecode object can be found in `tests/test_timecode.py`.

- `platform.py`

- `shot_detector.py`  tamplate

#### [x] detectors (OpenCV)
https://docs.opencv.org/3.4/de/d25/imgproc_color_conversions.html

```python
# This convert image to grayscale
gray_image = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

# Useful for color and shade comparision
hsv_image = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

# Other conversions
effect_image = cv2.cvtColor(frame, cv2.COLOR_BGR2YCrCb)
effect_image = cv2.cvtColor(frame, cv2.COLOR_BGR2Luv)

# Calculate difference in image. Very useful for motion detection
gframe1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
gframe2 = cv2.cvtColor(frame2, cv2.COLOR_BGR2GRAY)
diff = cv2.subtract(gframe1, gframe2)  # Results in difference 
diff_sum = diff.sum()  # Sum of all pixel values.
```

- `content_detector_hsv_luv.py`
  - For content-aware shot detection
  - Detects fast cuts using changes in color and intensity between frames
    - i.e. 使用帧之间颜色和强度的变化检测快速剪切
  - Since the difference between frames is used, unlike the ThresholdDetector, **only fast cuts are detected** with this method.  To detect slow fades between content shots still using HSV information, use the DissolveDetector.
    - i.e. 由于使用了帧之间的差异，因此与ThresholdDetector不同，此方法仅检测到快速剪切. 要检测仍在使用HSV信息的内容拍摄之间的缓慢淡入，请使用DissolveDetector

- `motion_detector.py`
  - Detects motion events in shots containing a static background.
    - i.e. 检测包含静态背景的镜头中的运动事件
  - Uses background subtraction followed by noise removal (via morphological opening) to generate a frame score compared against the set threshold.
    - i.e. 使用背景减法，然后去除噪声，以生成与设定阈值相比的帧得分。

- `threadhold_detector.py`
  - Computes the average pixel value/intensity for all pixels in a frame.
    - i.e. 计算一帧中所有像素的平均像素值/强度
  - The value is computed by adding up the 8-bit R, G, and B values for each pixel, and dividing by the number of pixels multiplied by 3.
    - i.e. 通过将每个像素的8位R，G和B值相加，然后除以像素数乘以3，可以计算出该值

### utils(skip)
Mostly the same!!!
- `package.py`
- `utilis.py`

## place
- `extrac_feat.py`     # Extract place feature - lgss 
  - Extractor(ResNet50)
    - Key: Extractor.extract_features => feature and score


## audio
- `extrac_feat.py`     # Extract audio feature - src
  - using librosa.core.amplitude_to_db
  - 2 output dirs: 
    - wav  : mp4 to wav by shot
      - using ffmpeg, convert to `shot_id.wav`
    - stft : wav to stft using `librosa`
      - stft: short-term Fourier transform

