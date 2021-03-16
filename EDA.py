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
# %xmode Plain

# %% tags=[]
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
    # # !google-drive-ocamlfuse -cc
    # !ls

# %%
# # %run main.py

# %%
import argparse
# from utilis.package import *
# from utilis import mkdir_ifmiss, read_pkl, write_pkl, to_numpy
from utilis import preview_feat, preview_single_pkl, preview_single_npy
from utilis import list_dir
from mmcv import Config

# %% [markdown] id="q9y0KhBUE9F8"
# # Preview .pkl

# %%
data_root = './data/'
parser = argparse.ArgumentParser()
args = parser.parse_args(args=[])
args.num_workers = 10
args.extract_all_feat = False  # if False, only unpack `cast` and `action`


# %%
cfg = Config.fromfile('./config/train_all.py')

# %%
# get all videos in folder
# video_list = list_dir(cfg.dirs.source_path, 'pkl')
# video_list = [video_id.split('.')[0] for video_id in video_list]
# print(video_list)

# %% [markdown] id="nISE80ur-yv-"
# ## Preview Packed

# %%
preview_feat(cfg.dirs.source_path, 'tt2024544')

# %% [markdown] id="ialkHy-M-1Qt"
# ## Preview Single

# %% executionInfo={"elapsed": 41017, "status": "ok", "timestamp": 1614814718348, "user": {"displayName": "Tim Shen", "photoUrl": "https://lh3.googleusercontent.com/a-/AOh14GgNQrJ7jwDo36WwiK5d1fd2sl5mNxpQC8UsAvxjZw=s64", "userId": "18284420496267004524"}, "user_tz": 300} id="XmokrsPi9BC8"
# just cuirous
# for key, item in feat_pkl.items():
#     if key == 'cast' or key == 'action':
#         print(key, torch.sum(item[:10]), '\n')

# %% id="qCzr3HAgYjq7"
# # act and cast
# a = preview_single_pkl(data_root, 'act_feat/tt2024544', ret=True)
# b = preview_single_pkl(data_root, 'act_tt2024544', ret=True)

# c = preview_single_pkl(data_root, 'cast_feat/tt2024544', ret=True)
# d = preview_single_pkl(data_root, 'cast_tt2024544', ret=True)

# # aud and place
# preview_single_npy(data_root, 'aud_feat/tt2024544/shot_0000')
# preview_single_npy(data_root, 'aud_shot_0000')

# preview_single_npy(data_root, 'place_feat/tt2024544/shot_0000')
# preview_single_npy(data_root, 'place_shot_0000')

# %% [markdown]
# # lol

# %%
# feat_st0 = preview_single_npy(data_root, 'feat/tt2024544/audio', ret=True)
# feat_st1 = preview_single_npy(data_root, 'aud_feat/tt2024544/shot_0000', ret=True)
# feat_st2 = preview_single_npy(data_root, 'aud_feat/tt2024544/shot_0001', ret=True)

# feat_t1 = preview_single_npy('', '/home/Data/Senseg_back/aud_feat/tt2024544/shot_0000',  ret=True)
# feat_t2 = preview_single_npy('', '/home/Data/Senseg_back/aud_feat/tt2024544/shot_0001',  ret=True)

# print('-----------------------------\n')

# feat_su0 = preview_single_npy(data_root, 'feat/tt0063442/audio', ret=True)
# feat_st1 = preview_single_npy(data_root, 'aud_feat/tt0063442/shot_0000', ret=True)
# feat_st2 = preview_single_npy(data_root, 'aud_feat/tt0063442/shot_0001', ret=True)

# feat_u1 = preview_single_npy('', '/home/Data/Senseg_back/aud_feat/tt0063442/shot_0000',  ret=True)
# feat_u2 = preview_single_npy('', '/home/Data/Senseg_back/aud_feat/tt0063442/shot_0001',  ret=True)

# # preview_single_pkl(data_root, 'cast_feat/tt2024544')
# # preview_single_pkl(data_root, 'cast_tt2024544')
