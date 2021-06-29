import os
import os.path
import torch
import torch.nn as nn
from torch.autograd import Variable
import easydict
from models import DnCNN
from skimage import io
import matplotlib.pyplot as plt
import numpy as np
from testNoiseGenerator import noise_generator
from testNoiseGenerator import batch_psnr
from scipy.signal import wiener

os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0"


networkPath = "/home/kayrish52/PycharmProjects/5584HW4/logs/500Training/4BlkSmoothingPlusAdditive/net.pth"
# imgFile = "/home/kayrish52/PycharmProjects/5584HW4/testImages/Cameraman.png"
imgFile = "/home/kayrish52/PycharmProjects/5584HW4/testImages/Lena.png"
opt = easydict.EasyDict({
        "num_of_layers": 4,
        "noiseL": 20.0,
})


# def main(networkPath, img, opt):
def main():

    # Build Model and Move to GPU
    torch.cuda.empty_cache()
    model = nn.DataParallel(DnCNN(3, num_of_layers=4)).cuda()

    # Load the Model
    model.load_state_dict(torch.load(networkPath))

    # Print model's state_dict
    print("Model's state_dict:")
    for param_tensor in model.state_dict():
        print(param_tensor, "\t", model.state_dict()[param_tensor].size())

    img = np.array(io.imread(imgFile))/255
    sizeImg = img.shape

    imgN = img + noise_generator(img, opt.noiseL, 5)
    imgNPSNR = batch_psnr(imgN, img, 1.)

    # Apply Wiener Filter
    # kernel = gaussian_kernel(3)
    # kernel = gsgPSF()
    # imgWNR = wiener_filter(img, kernel, K=10)
    # imgWNR = unsupervised_wiener(img, kernel)
    # imgWNR = wiener(img, kernel, 1.0)
    imgWNR = wiener(imgN, (5, 5, 5))
    imgWNRPSNR = batch_psnr(imgWNR, img, 1.)

    imgTranspose = imgN.transpose(2, 1, 0)
    img_ = np.zeros((1, sizeImg[2], sizeImg[1], sizeImg[0]))
    img_[0, :, :, :] = imgTranspose

    output = model(Variable(torch.Tensor(img_)))
    imgRes_ = output[0, :, :, :]
    imgRes = imgRes_.cpu().data.numpy().transpose(2, 1, 0)
    imgCNN = imgN - imgRes
    imgCNNPSNR = batch_psnr(imgCNN, img, 1.)

    fig = plt.figure()
    plt.subplot(221)
    plt.imshow(img)
    plt.title("Original Image")
    plt.subplot(222)
    plt.imshow(imgN)
    plt.title("Noisy Plus Blurred Image\nPSNR = %4.2f" % imgNPSNR)
    plt.subplot(223)
    plt.imshow(imgWNR)
    plt.title("Wiener Deconvolved Image\nPSNR = %4.2f" % imgWNRPSNR)
    plt.subplot(224)
    plt.imshow(imgCNN)
    plt.title("VDSR Deconvolved Image\nPSNR = %4.2f" % imgCNNPSNR)
    plt.show()

    # plt.savefig("/home/kayrish52/PycharmProjects/5584HW4/logs/500Training/8BlkMotionPlusAdditive/output.png")


if __name__ == "__main__":
    main()


