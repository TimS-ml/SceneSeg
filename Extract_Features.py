# ---
# jupyter:
#   jupytext:
#     formats: ipynb,py:percent
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.10.3
#   kernelspec:
#     display_name: Python 3
#     language: python
#     name: python3
# ---

# %%
# %xmode Verbose

# %%
import os

try:
    from google.colab import drive
    drive.mount("/content/drive")
    os.chdir('/content/drive/My Drive/Project/TakeHome/Eluvio/')
    print('Env: colab, run colab init')
    isColab = True
except:
    os.chdir('.')
    cwd = os.getcwd()
    print('Env: local')
    isColab = False

# %%
# install packages and refresh dirs
if isColab:
    # # !pip install pytorch-crf
    # # !pip install seqeval
    # # !pip install transformers
    # # !nvidia-smi
    # # !mkdir checkpoints
    # # !mkdir log
    # !ls

# %%
# # %run main.py


# %%
import torch
from utilis.package import *
from utilis import mkdir_ifmiss, read_pkl, write_pkl, to_numpy
from utilis.feat_utilis import *
# import multiprocessing
from mmcv import Config

# %% [markdown] id="q9y0KhBUE9F8"
# # Preview .pkl

# %%
data_root = './data/'
parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.num_workers = 4
args.extract_all_feat = True  # if False, only unpack `cast` and `action`


# %%
cfg = Config.fromfile('./config/mycfg_full.py')

# %%
for v in cfg.dirs.values():
    print(v)

# %%
# get all videos in folder
video_list = list_dir(cfg.dirs.source_path, 'pkl')
video_list = [video_id.split('.')[0] for video_id in video_list]
print(video_list)

# %% [markdown] id="aE49TQhnTsrM"
# # Preprocessing

# %% [markdown] id="vWNIf0QOE3vX"
# ## Unpack feat

# %% executionInfo={"elapsed": 9753, "status": "ok", "timestamp": 1614815915852, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 300} id="M1j6DEEs-cof"
# Extract features

# for video_id in tqdm(video_list):
#     # print(video_id)
#     mkdir_ifmiss(osp.join(cfg.dirs.feat_path, video_id))
#     extract_feat(cfg.dirs.source_path, cfg.dirs.feat_path, video_id, extract_all_feat=False)

# %% [markdown] id="caE7siPPUIsx"
# ## Place and audio

# %%
unpack_place_feat(cfg, video_list)
unpack_aud_feat(cfg, video_list)

# %% [markdown]
# ## Cast and action

# %% id="RzBehomebMyW"
# mkdir_ifmiss(osp.join(cfg.dirs.cast_feat_path))
# mkdir_ifmiss(osp.join(cfg.dirs.act_feat_path))

unpack_cast_feat(cfg, video_list)
unpack_act_feat(cfg, video_list)
