import torch
import cv2
import numpy as np


def noise_generator(img, noiseL, noiseMethod):
    if noiseMethod == 0:
        noise = 0.01*torch.FloatTensor(img[0, :, :, :].size()).normal_(mean=0, std=noiseL/255)
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
    additive_gaussian_noise = torch.zeros(img.size())
    for n in range(additive_gaussian_noise.size()[0]):
        additive_gaussian_noise[n, :, :, :] = torch.FloatTensor(img[0, :, :, :].size()).normal_(mean=0,
                                                                                                std=noiseL/255)

    return additive_gaussian_noise


def mng(img, degree, angle):
    # motion noise generation
    img_psf_numpy = torch.zeros(img.size()).numpy()
    for q in range(img.shape[0]):
        img_i = img[q, :, :, :]
        img_i_0_1 = img_i.transpose(0, 1)
        img_i_1_2 = img_i_0_1.transpose(1, 2)
        img_i_1_2_numpy = img_i_1_2.numpy()
        img_i_1_2_numpy_bgr = cv2.cvtColor(img_i_1_2_numpy, cv2.COLOR_RGB2BGR)

        # create motion kernel
        m = cv2.getRotationMatrix2D((degree / 2, degree / 2), angle, 1)
        motion_blur_kernel = np.diag(np.ones(degree))
        motion_blur_kernel = cv2.warpAffine(motion_blur_kernel, m, (degree, degree))
        motion_blur_kernel = motion_blur_kernel / degree

        # blur the image
        blurred_numpy = cv2.filter2D(img_i_1_2_numpy_bgr, -1, motion_blur_kernel)
        blurred_numpy_rgb = cv2.cvtColor(blurred_numpy, cv2.COLOR_BGR2RGB)
        blurred_numpy_rgb_tensor = torch.tensor(blurred_numpy_rgb)
        blurred_numpy_rgb_tensor_1_2 = blurred_numpy_rgb_tensor.transpose(1, 2)
        blurred_numpy_rgb_tensor_0_1 = blurred_numpy_rgb_tensor_1_2.transpose(0, 1)
        img_psf_numpy[q, :, :, :] = blurred_numpy_rgb_tensor_0_1.numpy()

    img_motion = torch.from_numpy(img_psf_numpy)
    motion_noise = img_motion - img

    return motion_noise


def gsg(img):
    # gaussian smoothing noise generation
    img_psf_numpy = torch.zeros(img.size()).numpy()
    for q in range(img.shape[0]):
        img_i = img[q, :, :, :]
        img_i_0_1 = img_i.transpose(0, 1)
        img_i_1_2 = img_i_0_1.transpose(1, 2)
        img_i_1_2_numpy = img_i_1_2.numpy()
        img_i_1_2_numpy_bgr = cv2.cvtColor(img_i_1_2_numpy, cv2.COLOR_RGB2BGR)

        # create motion kernel
        m = cv2.getGaussianKernel(5, 2)

        # blur the image
        blurred_numpy = cv2.filter2D(img_i_1_2_numpy_bgr, -1, m)
        blurred_numpy_rgb = cv2.cvtColor(blurred_numpy, cv2.COLOR_BGR2RGB)
        blurred_numpy_rgb_tensor = torch.tensor(blurred_numpy_rgb)
        blurred_numpy_rgb_tensor_1_2 = blurred_numpy_rgb_tensor.transpose(1, 2)
        blurred_numpy_rgb_tensor_0_1 = blurred_numpy_rgb_tensor_1_2.transpose(0, 1)
        img_psf_numpy[q, :, :, :] = blurred_numpy_rgb_tensor_0_1.numpy()

    img_smoothed = torch.from_numpy(img_psf_numpy)
    gaussian_smoothing_noise = img_smoothed - img

    return gaussian_smoothing_noise
