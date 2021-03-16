from . import mkdir_ifmiss, read_pkl, write_pkl, to_numpy
from .package import *


def init_dir(cfg):
    '''
    make several dirs
    '''
    for v in cfg.dirs.values():
        if isinstance(v, str):
            mkdir_ifmiss(v)
    # for v in cfg.dirs.feat_save_dict.values():
    #     if isinstance(v, str):
    #         mkdir_ifmiss(v)


def has_empty_folder(path):
    '''
    return all empty folders in path
    '''
    empty_folder = []
    for root, dirs, shot_list in tqdm(os.walk(path)):
        try:
            os.path.join(root, shot_list[0])
        except:
            print('!!! Folder is empty', root)
            empty_folder.append(root)
    return empty_folder


def list_dir(path, t='pkl', basename=True):
    '''
    return list of specific files or subdir
    '''
    from glob import glob
    if t == 'dir':
        return glob(path + '*/')
    else:
        if basename:
            return list(map(osp.basename, glob(path + '*.' + t)))
        else:
            return glob(path + '*.' + t)


def preview_single_pkl(path, video_id, ret=False):
    '''
    preview single pkl features: path + video_id.pkl
    only print first 16 values
    '''
    feat_pkl = read_pkl(osp.join(path, video_id + '.pkl'))
    print(video_id)
    print(type(feat_pkl), len(feat_pkl))
    if isinstance(feat_pkl, dict):
        firstKey = next(iter(feat_pkl))
        print(firstKey, len(feat_pkl[firstKey]))
        print(feat_pkl[firstKey][:16], '\n')
    else:
        print(len(feat_pkl[0]))
        print(feat_pkl[0][:16], '\n')
    if ret:
        return feat_pkl


def preview_single_npy(path, video_id, ret=False):
    '''
    preview single npy features: path + video_id.npy
    only print first 16 values
    '''
    feat_npy = np.load(osp.join(path, video_id + '.npy'))
    print(video_id)
    print(type(feat_npy), feat_npy.shape)
    if len(feat_npy.shape) > 1:
        print(feat_npy[0][:16], '\n')
    else:
        print(feat_npy[:16], '\n')
    if ret:
        return feat_npy


def preview_feat(path, video_id):
    '''
    preview packed features: path + video_id.pkl
    only print first 16 values
    '''
    feat_pkl = read_pkl(osp.join(path, video_id + '.pkl'))
    for key, item in feat_pkl.items():
        if isinstance(item, str):
            print(key, item)
        else:
            print(key, item.shape)
            if len(list(item.shape)) == 1:
                print(item[:16], '\n')
            else:
                print(item[:4][:16], '\n')


def extract_feat(source, dest, video_id, extract_all_feat=True):
    '''
    extract packed features into separate files
    data_root/dest/video_id/{feature name}.xxx
    '''
    feat_pkl = read_pkl(osp.join(source, video_id + '.pkl'))
    for key, item in feat_pkl.items():
        if not isinstance(item, str):
            # print(key, item.shape)
            if key == 'cast' or key == 'action':
                # convert to *.npy
                item = to_numpy(item)
                save_fn = osp.join(dest, video_id,
                                   "{}.npy".format(key))
                np.save(save_fn, item)
            elif extract_all_feat == True:
                # convert to *.npy
                item = to_numpy(item)
                save_fn = osp.join(dest, video_id,
                                   "{}.npy".format(key))
                np.save(save_fn, item)
            else:
                continue


def unpack_place_feat(cfg, video_list):
    '''
    unpack place features into format:
    data_path/place_feat/video_id/shot_xxxx.npy
    also check pre/place/extract_feat.py
    video_list: list of move's imdb id or sth else
    '''
    for video_id in tqdm(video_list):
        # unpack single
        feat_npy_path = osp.join(cfg.dirs.feat_path, video_id, 'place.npy')
        try:
            # e.g. ./data/feat/video_id/place.npy
            feat_npy = np.load(feat_npy_path)
            mkdir_ifmiss(osp.join(cfg.dirs.place_feat_path, video_id))
            for shot_no, shot_feat in enumerate(feat_npy):
                # e.g. ./data/place_feat/video_id/shot_xxxx.npy
                save_fn = osp.join(cfg.dirs.place_feat_path, video_id,
                                   'shot_{0:04}.npy'.format(shot_no))
                np.save(save_fn, shot_feat)
        except:
            print('{} Extract Error !!!'. format(feat_npy_path))


def unpack_aud_feat(cfg, video_list):
    '''
    unpack place features into format:
    data_path/aud_feat/video_id/shot_xxxx.npy
    also check: pre/audio/extract_feat.py
    video_list: list of move's imdb id or sth else
    '''
    for video_id in tqdm(video_list):
        # unpack single
        feat_npy_path = osp.join(cfg.dirs.feat_path, video_id, 'audio.npy')
        try:
            # e.g. ./data/feat/video_id/audio.npy
            feat_npy = np.load(feat_npy_path)
            mkdir_ifmiss(osp.join(cfg.dirs.aud_feat_path, video_id))
            for shot_no, shot_feat in enumerate(feat_npy):
                # e.g. ./data/place_feat/video_id/shot_xxxx.npy
                save_fn = osp.join(cfg.dirs.aud_feat_path, video_id,
                                   'shot_{0:04}.npy'.format(shot_no))
                np.save(save_fn, shot_feat)
        except:
            print('{} Extract Error !!!'. format(feat_npy_path))


def unpack_cast_feat(cfg, video_list):
    '''
    unpack place features into format:
    data_path/cast_feat/video_id.pkl
    dict = {shot number:shot value} 
    pls check if dir exists first
    '''
    for video_id in tqdm(video_list):
        # e.g. ./data/feat/video_id/cast.npy
        feat_npy_path = osp.join(cfg.dirs.feat_path, video_id, 'cast.npy')
        # e.g. ./data/xxx_feat/video_id.pkl
        feat_pkl_dest = osp.join(cfg.dirs.cast_feat_path, video_id + '.pkl')
        # print(feat_npy_path, feat_pkl_dest)

        feat_npy = np.load(feat_npy_path)
        feat_dict = {}
        for shot_no, shot_feat in enumerate(feat_npy):
            feat_dict['{0:04}'.format(shot_no)] = [shot_feat]
        write_pkl(feat_pkl_dest, feat_dict)


def unpack_act_feat(cfg, video_list):
    '''
    unpack place features into format:
    data_path/act_feat/video_id.pkl
    dict = {shot number:shot value} 
    pls check if dir exists first
    '''
    for video_id in tqdm(video_list):
        # e.g. ./data/feat/video_id/action.npy
        feat_npy_path = osp.join(cfg.dirs.feat_path, video_id, 'action.npy')
        # e.g. ./data/xxx_feat/video_id.pkl
        feat_pkl_dest = osp.join(cfg.dirs.act_feat_path, video_id + '.pkl')
        # print(feat_npy_path, feat_pkl_dest)

        feat_npy = np.load(feat_npy_path)
        feat_dict = {}
        for shot_no, shot_feat in enumerate(feat_npy):
            feat_dict['{0:04}'.format(shot_no)] = shot_feat
        write_pkl(feat_pkl_dest, feat_dict)
