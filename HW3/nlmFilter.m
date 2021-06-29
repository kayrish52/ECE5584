function [nlmFiltImg,nlmPSNR] = nlmFilter(imgFile,noisyImg)
%function [nlmFiltImg,nlmPSNR] = nlmFilter(imgFile,noisyImg)
% - DEFINITION: This function applies a Non-Local Means Filter to the noisy
%               image. It will receive the original image as well as the
%               noisy image, and apply the build-in MATLAB NLM Filter. It
%               returns the filtered image and the signal to noise ratio of
%               the filtered image.
% - INPUTS:
% -- imgFile: The original input image that will have noise added for
%             purposes of HW3.
% -- noisyImg: The noisy image with motion or Gaussian white noise added
%              prior to filtering.
% - OUTPUTS:
% -- nlmFiltImg: The resulting filtered output image of the NLM Filter.
% -- nlmPSNR: The PSNR of the NLM Filtered image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read and display the image.
I = im2double(imread(imgFile));
figure(1);
subplot(1,3,1);
imshow(I);
title('Original Image');

% Simulate additive noise.
subplot(1,3,2);
imshow(noisyImg);
title(sprintf('Blur and Noise\nPSNR=%1.2f',psnr(I,noisyImg)));

% Apply the NLM Filter to the noisy Image.
[nlmFiltImg,estDos] = imnlmfilt(noisyImg);
nlmPSNR = psnr(I,nlmFiltImg);
subplot(1,3,3);
imshow(nlmFiltImg);
title(sprintf(['NLM Filtered Image\nEstimated DoS = ' ...
    '%.1f\nPSNR=%1.2f'],estDos,nlmPSNR));

end

