import os
import torch.optim as optim
from torch.autograd import Variable
from torch.utils.data import DataLoader
from models import DnCNN
from dataset import prepare_data
from dataset import Dataset
from utils import *
from noise_generator import noise_generator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import easydict
from statistics import mean

dPath = "/media/kayrish52/DataStorage/NWPU"

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"

opt = easydict.EasyDict({
        "preprocess": False,
        "batchSize": 128,
        "num_of_layers": 4,
        "epochs": 5,
        "milestone": 30,
        "lr": 1e-3,
        "outf": "logs",
        "mode": "B",
        "noiseL": 20.0,
        "val_noiseL": 20.0
})

savePath = "test"

noiseMethod = 5


def main(opt, savePath, noiseMethod):
# def main():

    # Load dataset
    print('Loading dataset ...\n')
    dataset_train = Dataset(train=True)
    dataset_val = Dataset(train=False)
    loader_train = DataLoader(dataset=dataset_train, num_workers=8, batch_size=opt.batchSize, shuffle=True)
    print("# of training samples: %d\n" % int(len(dataset_train)))

    # Build model
    net = DnCNN(channels=3, num_of_layers=opt.num_of_layers)
    net.apply(weights_init_kaiming)
    criterion = nn.MSELoss(size_average=False)

    # Move to GPU
    device_ids = [0]
    model = nn.DataParallel(net, device_ids=device_ids).cuda()
    criterion.cuda()

    # Optimizer
    optimizer = optim.Adam(model.parameters(), lr=opt.lr)

    # training
    step = 0
    temp = []
    trainPSNR = []
    valPSNR = []
    valLoss = []

    noiseL_B = [0, 12.75]  # ignored when opt.mode=='S'
    for epoch in range(opt.epochs):
        if epoch < opt.milestone:
            current_lr = opt.lr
        else:
            current_lr = opt.lr / 10.

        # set learning rate
        for param_group in optimizer.param_groups:
            param_group["lr"] = current_lr
        print('learning rate %f' % current_lr)

        # train
        for i, data in enumerate(loader_train, 0):

            # training step
            temp1 = []
            trainPSNR1 = []
            noisePSNR1 = []
            model.train()
            model.zero_grad()
            optimizer.zero_grad()
            img_train = data

            # generate noise
            noise = noise_generator(img_train, opt.noiseL, noiseMethod)

            img_noise_train = img_train + noise

            img_train = Variable(img_train.cuda())
            img_noise_train = Variable(img_noise_train.cuda())
            noise1 = Variable(noise.cuda())
            out_train = model(img_noise_train)
            loss = criterion(out_train, noise1) / (img_noise_train.size()[0]*2)
            temp1.append(loss.item())

            # optimization step
            loss.backward()
            optimizer.step()

            # results
            model.eval()
            out_train = torch.clamp(img_noise_train-model(img_noise_train), 0., 1.)

            psnr_noise = batch_psnr(img_noise_train, img_train, 1.)
            noisePSNR1.append(psnr_noise.item())

            psnr_train = batch_psnr(out_train, img_train, 1.)
            trainPSNR1.append(psnr_train.item())
            print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" % (epoch+1, i+1, len(loader_train), loss.item(),
                                                                     psnr_train))

            step += 1
        # the end of each epoch

        model.eval()

        # validate
        psnr_val = 0
        lossVal = 0
        valPSNR1 = []
        valLoss1 = []
        for k in range(len(dataset_val)):
            img_val = torch.unsqueeze(dataset_val[k], 0)
            noise = noise_generator(img_val, opt.val_noiseL, noiseMethod)
            img_noise_val = img_val + noise
            img_val = Variable(img_val.cuda())
            img_noise_val = Variable(img_noise_val.cuda())
            noise1 = Variable(noise.cuda())
            out_val = model(img_noise_val)
            lossVal = criterion(out_val, noise1) / (img_noise_val.size()[0]*2)
            lossVal.backward()
            valLoss1.append(lossVal.item())
            out_val = torch.clamp(img_noise_val - model(img_noise_val), 0., 1.)
            psnr_val = batch_psnr(out_val, img_val, 1.)
            valPSNR1.append(psnr_val.item())

        print("\n[epoch %d], ValLoss: %.4f, PSNR_val: %.4f" % (epoch+1, mean(valLoss1), psnr_val))

        # writer.add_scalar('PSNR on validation data', psnr_val, epoch)
        # log the images
        # out_train = torch.clamp(img_noise_train-model(img_noise_train), 0., 1.)
        # Img = utils.make_grid(img_train.data, nrow=8, normalize=True, scale_each=True)
        # Imgn = utils.make_grid(imgn_train.data, nrow=8, normalize=True, scale_each=True)
        # Irecon = utils.make_grid(out_train.data, nrow=8, normalize=True, scale_each=True)
        # writer.add_image('clean image', Img, epoch)
        # writer.add_image('noisy image', Imgn, epoch)
        # writer.add_image('reconstructed image', Irecon, epoch)

        # save model
        torch.save(model.state_dict(), os.path.join(opt.outf, savePath, 'net.pth'))
        temp.append(mean(temp1))
        trainPSNR.append(mean(trainPSNR1))
        valPSNR.append(mean(valPSNR1))
        valLoss.append(mean(valLoss1))

    fig = plt.figure()
    # plt.suptitle('Performance of VDSR for Image Denoising\nCNN Depth: {}-Blocks\n\n'.format(opt.num_of_layers))
    plt.subplot(121)
    plt.plot(np.arange(0, opt.epochs), temp, label="Training Loss")
    plt.plot(np.arange(0, opt.epochs), valLoss, label="Validation Loss")
    plt.xlabel('Epoch')
    plt.ylabel('Loss Value')
    plt.title('Loss Value per Epoch')
    plt.legend()
    plt.subplot(122)
    plt.plot(np.arange(0, opt.epochs), trainPSNR, label="Training PSNR")
    plt.plot(np.arange(0, opt.epochs), valPSNR, label="Validation PSNR")
    plt.xlabel('Epoch')
    plt.ylabel('PSNR')
    plt.title('PSNR per Epoch')
    # plt.tight_layout(h_pad=10, w_pad=10, rect=[0, 0, 0.9, 0.9])
    plt.legend()
    # plt.subplots_adjust(top=0.8)
    plt.savefig(os.path.join(opt.outf, savePath, 'Training'))
#    plt.show()

    # plt.figure()
    # plt.plot(np.arange(0, opt.epochs), noisePSNR)
    # plt.show()


if __name__ == "__main__":
    # prepare_data(data_path=dPath, patch_size=40, stride=10, aug_times=1)
    # if opt.preprocess:
    #     if opt.mode == 'S':
    #         prepare_data(data_path='../dataset/NWPU', patch_size=40, stride=10, aug_times=1)
    #     if opt.mode == 'B':
    #         prepare_data(data_path='../dataset/NWPU', patch_size=50, stride=10, aug_times=2)
    main(opt, savePath, noiseMethod)
