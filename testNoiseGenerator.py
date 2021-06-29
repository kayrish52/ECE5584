import torch
import numpy as np
import cv2
from skimage.measure.simple_metrics import compare_psnr


def noise_generator(img, noiseL, noiseMethod):
    if noiseMethod == 0:
        noise = 0.01 * agn(img, noiseL)
    elif noiseMethod == 1:
        noise = agn(img, noiseL)
    elif noiseMethod == 2:
        noise = mng(img, 21, 11)
    elif noiseMethod == 3:
        noise = gsg(img)
    elif noiseMethod == 4:
        noise = agn(img, noiseL) + mng(img, 21, 11)
    elif noiseMethod == 5:
        noise = agn(img, noiseL) + gsg(img)
    else:
        noise = agn(img, noiseL) + gsg(img) + mng(img, 21, 11)

    return noise


def agn(img, noiseL):
    # additive gaussian noise generation
    return np.random.normal(0, noiseL, img.shape)/255


def mng(img, degree, angle):
    # create motion kernel
    m = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree

    # blur the image
    blurred_numpy = cv2.filter2D(img, -1, motion_blur_kernel)
    motion_noise = blurred_numpy - img

    return motion_noise


def gsg(img):
    # create motion kernel
    m = cv2.getGaussianKernel(5, 2)

    # blur the image
    blurred_numpy = cv2.filter2D(img, -1, m)
    gaussian_smoothing_noise = blurred_numpy - img

    return gaussian_smoothing_noise


def mngPSF():
    # Set parameters
    degree = 21
    angle = 11
    # create motion kernel
    m = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
    motion_blur_kernel = np.diag(np.ones(degree))
    motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (degree, degree))
    motion_blur_kernel = motion_blur_kernel / degree

    return motion_blur_kernel


def gsgPSF():
    m = np.zeros(shape=(5, 5, 3))

    # create motion kernel
    m1 = cv2.getGaussianKernel(5, 2)
    m2 = cv2.getGaussianKernel(5, 2)
    m3 = cv2.getGaussianKernel(5, 2)

    m[:, :, 0] = m1
    m[:, :, 1] = m2
    m[:, :, 2] = m3

    return m


def batch_psnr(img, imclean, data_range):
    Img = img.astype(np.float32)
    Iclean = imclean.astype(np.float32)
    PSNR = compare_psnr(Iclean[:, :, :], Img[:, :, :], data_range=data_range)
    return PSNR
