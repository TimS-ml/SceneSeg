experiment_name = "all_318"
experiment_description = "Tim's scene segmentation"

# Change this if state dict trained on multiple features 
# and you don't want to run test on all features 
test_place = 1
test_cast = 1
test_act = 1
test_aud = 1

# overall confg
# create a `shot_movie318` link in ./data in google drive if you don't have ...
data_root = './data'
shot_frm_path = data_root + "/shot_movie318"  # movieID.txt
shot_num = 4  # even
seq_len = 10  # even

# dataset dirs for preprocessing
dirs = dict(
    source_path='./data/packed/',
    feat_path='./data/feat/',
    place_feat_path='./data/place_feat/',
    aud_feat_path='./data/aud_feat/',
    cast_feat_path='./data/cast_feat/',
    act_feat_path='./data/act_feat/',
)

# dataset settings
dataset = dict(
    name='all',
    mode=['place', 'cast', 'act'],  
)

# model settings
model = dict(
    name='LGSS',  # Local to global scene seg
    sim_channel=512,  # dim of similarity vector
    place_feat_dim=2048,
    cast_feat_dim=512,
    act_feat_dim=512,
    aud_feat_dim=512,
    aud=dict(cos_channel=512),
    bidirectional=True,
    lstm_hidden_size=512,
    ratio=[0.5, 0.2, 0.2, 0])

# optimizer
optim = dict(name='Adam', setting=dict(lr=1e-2, weight_decay=5e-4))
stepper = dict(name='MultiStepLR', setting=dict(milestones=[15]))
loss = dict(weight=[0.5, 5])

# runtime settings
resume = None
trainFlag = 0
testFlag = 1
#  batch_size = 128
batch_size = 32
epochs = 30
logger = dict(log_interval=20,
              logs_dir="./run/{}".format(experiment_name))  # path updated
data_loader_kwargs = dict(num_workers=4, pin_memory=True, drop_last=True)
