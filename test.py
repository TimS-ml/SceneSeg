from __future__ import print_function

from collections import OrderedDict
from mmcv import Config

import src.models as models
import torch
import torch.nn as nn
import torch.nn.functional as F

from src import get_data
from torch.utils.data import DataLoader
from utilis import (cal_MIOU, cal_Recall, cal_Recall_time, get_ap, get_mAP_seq,
                    load_checkpoint, mkdir_ifmiss, pred2scene, save_checkpoint,
                    save_pred_seq, scene2video, to_numpy, write_json)
from utilis.package import *


torch.manual_seed(2021)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False


parser = argparse.ArgumentParser(description='Runner')
parser.add_argument('config', help='config file path', default='./config/mycfg.py')
args = parser.parse_args()
# args = parser.parse_args(args=[])
# args.config = './config/mycfg.py'

cfg = Config.fromfile(args.config)

final_dict = {}
test_iter, val_iter = 0, 0


def test(cfg, model, test_loader, criterion, mode='test'):
    '''
    Returns:
        gts: Scene transition ground-truths
        preds: Predictions in probability
    '''
    global test_iter, val_iter
    model.eval()
    test_loss = 0
    correct1, correct0 = 0, 0
    gt1, gt0, all_gt = 0, 0, 0
    prob_raw, gts_raw = [], []
    preds, gts = [], []
    batch_num = 0

    with torch.no_grad():
        for data_place, data_cast, data_act, data_aud, target in test_loader:
            batch_num += 1
            data_place = data_place.cuda() if 'place' in cfg.dataset.mode or 'image' in cfg.dataset.mode else []
            data_cast = data_cast.cuda() if 'cast' in cfg.dataset.mode else []
            data_act = data_act.cuda() if 'act' in cfg.dataset.mode else []
            data_aud = data_aud.cuda() if 'aud' in cfg.dataset.mode else []
            target = target.view(-1).cuda()
            output = model(data_place, data_cast, data_act, data_aud)
            output = output.view(-1, 2)
            loss = criterion(output, target)

            test_loss += loss.item()
            output = F.softmax(output, dim=1)
            prob = output[:, 1]
            gts_raw.append(to_numpy(target))
            prob_raw.append(to_numpy(prob))

            gt = target.cpu().detach().numpy()
            prediction = np.nan_to_num(
                prob.squeeze().cpu().detach().numpy()) > 0.5
            idx1 = np.where(gt == 1)[0]
            idx0 = np.where(gt == 0)[0]
            gt1 += len(idx1)
            gt0 += len(idx0)
            all_gt += len(gt)
            correct1 += len(np.where(gt[idx1] == prediction[idx1])[0])
            correct0 += len(np.where(gt[idx0] == prediction[idx0])[0])

        for x in gts_raw:
            gts.extend(x.tolist())
        for x in prob_raw:
            preds.extend(x.tolist())

    test_loss /= batch_num
    ap = get_ap(gts_raw, prob_raw)
    mAP, mAP_list = get_mAP_seq(test_loader, gts_raw, prob_raw)
    print("AP: {:.3f}".format(ap))
    print('mAP: {:.3f}'.format(mAP))
    print('Average loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)'.format(
        test_loss, correct1 + correct0, 
        all_gt, 100. * (correct0 + correct1) / all_gt))
    print('Accuracy1: {}/{} ({:.0f}%), Accuracy0: {}/{} ({:.0f}%)'.format(
        correct1, gt1, 100. * correct1 / (gt1 + 1e-5), 
        correct0, gt0, 100. * correct0 / (gt0 + 1e-5)))

    if mode == "test_final":
        final_dict.update({
            "AP": ap,
            "mAP": mAP,
            "Accuracy": 100 * (correct0 + correct1) / all_gt,
            "Accuracy1": 100 * correct1 / (gt1 + 1e-5),
            "Accuracy0": 100 * correct0 / (gt0 + 1e-5),
        })
        return gts, preds


def run_test():
    '''
    Test pretrained model
    '''
    testSet = get_data(cfg, load='test')
    test_loader = DataLoader(testSet,
                            batch_size=cfg.batch_size,
                            shuffle=True,
                            **cfg.data_loader_kwargs)
    model = models.__dict__[cfg.model.name](cfg).cuda()
    model = nn.DataParallel(model)

    criterion = nn.CrossEntropyLoss(torch.Tensor(cfg.loss.weight).cuda())
    print("...data and model loaded")

    # Run Test
    print('...test with saved model')
    # load saved model for testing
    checkpoint = load_checkpoint(
        osp.join(cfg.logger.logs_dir, 'model_best.pth.tar'))

    # for those of you want to load part of the pre-trained model
    print('...loading state dict')

    # Let's try cast only
    state_dict = checkpoint['state_dict']
    new_state_dict = OrderedDict()

    # print(model)
    for k, v in state_dict.items():
        if 'place' in k and cfg.test_place:
            # print(k, v.size())
            new_state_dict[k] = v
        if 'cast' in k and cfg.test_cast:
            # print(k, v.size())
            new_state_dict[k] = v
        if 'act' in k and cfg.test_act:
            # print(k, v.size())
            new_state_dict[k] = v
        if 'aud' in k and cfg.test_aud:
            # print(k, v.size())
            new_state_dict[k] = v
        else:
            continue
    model.load_state_dict(new_state_dict)

    # run
    print('...start test')
    gts, preds = test(cfg, model, test_loader, criterion, mode='test_final')
    # get results
    print('...get results')
    save_pred_seq(cfg, test_loader, gts, preds)
    # calculate MIOU and Recall
    if cfg.shot_frm_path is not None:
        Miou = cal_MIOU(cfg, threshold=0.5)
        Recall = cal_Recall(cfg, threshold=0.5)
        Recall_time = cal_Recall_time(cfg, recall_time=3, threshold=0.5)
        final_dict.update({
            "Miou": Miou,
            "Recall": Recall,
            "Recall_time": Recall_time
        })
    else:
        print('...there is no correspondence file '
              'between shots and their frames')
    log_dict = {'cfg': cfg.__dict__['_cfg_dict'], 'final': final_dict}
    write_json(osp.join(cfg.logger.logs_dir, "log.json"), log_dict)


if __name__ == '__main__':
    if cfg.trainFlag:
        run_train()
    if cfg.testFlag:
        run_test()
