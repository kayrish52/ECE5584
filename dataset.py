import os
import os.path
import numpy as np
import random
import h5py
import torch
import cv2
import glob
import torch.utils.data as udata
from utils import data_augmentation


def normalize(data):
    return data/255.


def im2patch(img, win, stride=1):
    k = 0
    endc = img.shape[0]
    endw = img.shape[1]
    endh = img.shape[2]
    patch = img[:, 0:endw-win+0+1:stride, 0:endh-win+0+1:stride]
    total_pat_num = patch.shape[1] * patch.shape[2]
    y = np.zeros([endc, win*win, total_pat_num], np.float32)
    for i in range(win):
        for j in range(win):
            patch = img[:, i:endw-win+i+1:stride, j:endh-win+j+1:stride]
            y[:, k, :] = np.array(patch[:]).reshape(endc, total_pat_num)
            k = k + 1
    return y.reshape([endc, win, win, total_pat_num])


def prepare_data(data_path, patch_size, stride, aug_times=1):

    # train
    print('process training data')
    scales = [1]
    # scales = [1, 0.9, 0.8, 0.7]
    files = glob.glob(os.path.join(data_path, '*', '*.jpg'))

    train_files = []
    for i in range(25):
        idx = random.randrange(0, len(files))
        train_files.append(files.pop(idx))

    test_files = []
    for i in range(5):
        idx = random.randrange(0, len(files))
        test_files.append(files.pop(idx))

    train_files.sort()
    h5f = h5py.File('NWPU_train.h5', 'w')
    train_num = 0
    for i in range(len(train_files)):
        img = cv2.imread(train_files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            # Img = np.expand_dims(Img[:,:,:].copy(), 0)
            Img = np.transpose(Img, (2, 0, 1))
            Img = np.float32(normalize(Img))
            patches = im2patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (train_files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                h5f.create_dataset(str(train_num), data=data)
                train_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(str(train_num)+"_aug_%d" % (m+1), data=data_aug)
                    train_num += 1
    h5f.close()

    # val
    print('\nprocess validation data')
    # files.clear()
    # files = glob.glob(os.path.join(data_path, '*', '*.jpg'))
    test_files.sort()
    h5f = h5py.File('NWPU_val.h5', 'w')
    val_num = 0
    for i in range(len(test_files)):
        img = cv2.imread(test_files[i])
        h, w, c = img.shape
        for k in range(len(scales)):
            Img = cv2.resize(img, (int(h*scales[k]), int(w*scales[k])), interpolation=cv2.INTER_CUBIC)
            # Img = np.expand_dims(Img[:,:,:].copy(), 0)
            Img = np.transpose(Img, (2, 0, 1))
            Img = np.float32(normalize(Img))
            patches = im2patch(Img, win=patch_size, stride=stride)
            print("file: %s scale %.1f # samples: %d" % (test_files[i], scales[k], patches.shape[3]*aug_times))
            for n in range(patches.shape[3]):
                data = patches[:, :, :, n].copy()
                h5f.create_dataset(str(val_num), data=data)
                val_num += 1
                for m in range(aug_times-1):
                    data_aug = data_augmentation(data, np.random.randint(1, 8))
                    h5f.create_dataset(str(val_num)+"_aug_%d" % (m+1), data=data_aug)
                    val_num += 1
#    for i in range(len(test_files)):
#        print("file: %s" % test_files[i])
#        img = cv2.imread(test_files[i])
#        img = np.transpose(img, (2, 0, 1))
#        img = np.float32(normalize(img))
#        h5f.create_dataset(str(val_num), data=img)
#        val_num += 1
    h5f.close()
    print('training set, # samples %d\n' % train_num)
    print('val set, # samples %d\n' % val_num)


class Dataset(udata.Dataset):
    def __init__(self, train=True):
        super(Dataset, self).__init__()
        self.train = train
        if self.train:
            h5f = h5py.File('NWPU_train.h5', 'r')
        else:
            h5f = h5py.File('NWPU_val.h5', 'r')
        self.keys = list(h5f.keys())
        random.shuffle(self.keys)
        h5f.close()

    def __len__(self):
        return len(self.keys)

    def __getitem__(self, index):
        if self.train:
            h5f = h5py.File('NWPU_train.h5', 'r')
        else:
            h5f = h5py.File('NWPU_val.h5', 'r')
        key = self.keys[index]
        data = np.array(h5f[key])
        h5f.close()
        return torch.Tensor(data)
