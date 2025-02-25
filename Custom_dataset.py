# %% [markdown]
# # Custom Dataset

# %% [markdown]
# ## In this tutorial, we provide an example of adapting usb to custom dataset.

# %%
import numpy as np
import os
import os.path as osp
import glob
import re
from torchvision import transforms
from semilearn import get_data_loader, get_net_builder, get_algorithm, get_config, Trainer
from semilearn import split_ssl_data, BasicDataset
from semilearn.datasets.augmentation.randaugment import RandAugment
import copy
# %% [markdown]
# ## Specifiy configs and define the model

# %%

config = {
    'algorithm': 'fixmatch',
    'net': 'resnet50',
    'net_from_name': False,
    'use_pretrain': False, 
    'pretrain_path': './resnet50-19c8e357.pth',
    'save_dir': './saved_models/usb_cv/',
    'save_name': 'fixmatch_market1501',
    'resume': False,
    'load_path': './saved_models/usb_cv/fixmatch_market1501/latest_model.pth',
    'overwrite': True,
    'use_tensorboard': True,
    'use_wandb': False,

    # optimization configs
    'epoch': 200, #for all the configs in usb_cv is the same  
    'num_train_iter': 204800,  #for all the configs in usb_cv is the same  
    'num_eval_iter': 2048,     #for all the configs in usb_cv is the same  
    'num_log_iter': 256,       #for all the configs in usb_cv is the same  
    'num_warmup_iter': 5120,
    
    'optim': 'AdamW',          #for all the configs in usb_cv is the same  
    'lr': 5e-4,                
    'layer_decay': 0.95,        #???
    'batch_size': 8,           #for all the configs in usb_cv is the same  
    'eval_batch_size': 16,     #for all the configs in usb_cv is the same  

    # dataset configs           #sina should do this part
    'dataset': 'market1501',
    'num_labels': 751,
    'num_classes': 751,
    'img_size': (256, 128),
    'crop_ratio': 0.875,
    'data_dir': 'data',
    'train_sampler': 'RandomSampler',

    # algorithm specific configs
    'hard_label': True,
    'ema_m': 0.0,
    'momentum': 0.9,
    'weight_decay': 0.0005,
    'uratio': 1,
    'ulb_loss_ratio': 1.0,
    'T': 0.5,
    'p_cutoff': 0.95,
    'amp': False,
    'clip': 0.0,
    'use_cat': True,

    # device configs
    'world_size': 1,
    'num_workers': 4,
    'seed': 0,
    'rank': 0,
    'multiprocessing_distributed': False,
    'dist_url': 'tcp://127.0.0.1:10007',
    'dist_backend': 'gloo',
    'distributed': False,
    'gpu': 0
}

config = get_config(config)

# %%
# create model and specify algorithm
algorithm = get_algorithm(config,  get_net_builder(config.net, from_name=False), tb_log=None, logger=None)

# %% [markdown]
# ## Create dataset


# %%
def get_data(train=True):
    """
    get_data returns data (images) and targets (labels)
    shape of data: B, H, W, C
    shape of labels: B,
    """
    data_dir = os.path.join(config.data_dir, 'Market-1501-v15.09.15')
    train_dir = os.path.join(data_dir, 'bounding_box_train')
    test_dir = os.path.join(data_dir, 'bounding_box_test')
    train_images, train_labels = load_market1501_images(train_dir, relabel=True)
    test_images, test_labels = load_market1501_images(test_dir, relabel=False)
    return (train_images, train_labels) if train else (test_images, test_labels)


def load_market1501_images(directory, relabel=False):
    """
    Custom function to load Market1501 dataset images and labels.
    """
    images = []
    labels = []
    img_paths = glob.glob(osp.join(directory, '*.jpg'))
    pattern = re.compile(r'([-\d]+)_c(\d)')
    pid_container = set()
    for img_path in img_paths:
        pid, _ = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        pid_container.add(pid)
    pid2label = {pid: label for label, pid in enumerate(pid_container)}

    for img_path in img_paths:
        pid, camid = map(int, pattern.search(img_path).groups())
        if pid == -1: continue  # junk images are just ignored
        assert 0 <= pid <= 1501  # pid == 0 means background
        assert 1 <= camid <= 6
        camid -= 1  # index starts from 0
        if relabel: pid = pid2label[pid]
        labels.append(pid)
        images.append(img_path)

    unique_labels = np.unique(labels)  # Get unique labels
    print(f"unique labels {unique_labels}")
    label_map = {old_label: new_label for new_label, old_label in enumerate(unique_labels)}  # Map old -> new
    labels = np.array([label_map[y] for y in labels])
    print(labels.shape)
    return images, labels


train_transform = transforms.Compose([
    transforms.Resize((384, 128)),  # Resize to standard Market1501 size
    transforms.RandomHorizontalFlip(),
    transforms.RandomCrop(config.img_size, padding=(int(config.img_size[0] * (1 - config.crop_ratio)), int(config.img_size[1] * (1 - config.crop_ratio))), padding_mode='reflect'),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])


def get_dset(train, is_ulb=False,
             strong_transform=None, onehot=False):
    """
    get_dset returns class BasicDataset, containing the returns of get_data.

    Args
        is_ulb: If True, returned dataset generates a pair of weak and strong augmented images.
        strong_transform: list of strong_transform (augmentation) if use_strong_transform is True。
        onehot: If True, the label is not integer, but one-hot vector.
    """
    data, targets = get_data(train)
    return BasicDataset(algorithm, data, targets, config.num_classes, train_transform,
                        is_ulb, strong_transform, onehot)


def get_ssl_dset(num_labels, index=None, include_lb_to_ulb=True, strong_transform=None, onehot=False):
    """
    get_ssl_dset split training samples into labeled and unlabeled samples.
    The labeled data is balanced samples over classes.

    Args:
        num_labels: number of labeled data.
        index: If index of np.array is given, labeled data is not randomly sampled, but use index for sampling.
        include_lb_to_ulb: If True, consistency regularization is also computed for the labeled data.
        strong_transform: list of strong transform (RandAugment in FixMatch)
        onehot: If True, the target is converted into onehot vector.

    Returns:
        BasicDataset (for labeled data), BasicDataset (for unlabeld data)
    """
    data, targets = get_data()
    lb_data, lb_targets, ulb_data, ulb_targets = split_ssl_data(config, data, targets,
                                                                num_labels, config.num_classes,
                                                                index, include_lb_to_ulb)
    lb_dset = BasicDataset(algorithm, lb_data, lb_targets, config.num_classes,
                           train_transform, False, onehot)
    strong_transform = copy.deepcopy(train_transform)
    strong_transform.transforms.insert(0, RandAugment(3, 5))
    ulb_dset = BasicDataset(algorithm, ulb_data, ulb_targets, config.num_classes,
                            train_transform, True, strong_transform=strong_transform, onehot=onehot)
    return lb_dset, ulb_dset


lb_dset, ulb_dset = get_ssl_dset(config.num_labels,  index=None)

# %%
# replace with your own code
eval_dset = get_dset(train=False)

# %%
# define data loaders
train_lb_loader = get_data_loader(config, lb_dset, config.batch_size)
train_ulb_loader = get_data_loader(config, ulb_dset, int(config.batch_size * config.uratio))
eval_loader = get_data_loader(config, eval_dset, config.eval_batch_size)

# %% [markdown]
# ## Training and evaluation

# %%
# training and evaluation
trainer = Trainer(config, algorithm)
trainer.fit(train_lb_loader, train_ulb_loader, eval_loader)
trainer.evaluate(eval_loader)


