function [filteredImage] = wienerFilter(img,noiseImg,noiseParam,type,k,i)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [filteredImages] = wienerFilter(img,noiseImg,noiseParam,type,k)
% - DEFINITION: This function applies a Wiener Filter to the input noisy
%               image. It will receive the motion vector and the estimate
%               of the Gaussian noise parameter to determine the filtered
%               image.
% - INPUTS
% -- img: The input image that will have noise added for purposes of HW3.
% -- noiseParam: The noise parameters used for the blurred image. The
%                parameters include:
%                - len: The length of the motion vector for the motion 
%                       blurring.
%                - theta: The angle of the motion vector for the motion 
%                         blurring.
%                - noiseMean: The mean of the Gaussian noise for the 
%                             Gaussian blurring.
%                - noiseVar: The variance of the Gaussian noise for the 
%                            blurring.
%                - k: The ratio of the noise variance to the image 
%                     variance.
%                - i: Noise Variance selector.
%
% - OUTPUTS
% -- filteredImage: The resulting filtered output image.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Read and display the image.
I = im2double(imread(img));
figure(1);
subplot(2,2,1);
imshow(I);
title('Original Image');

% Simulate additive noise.
subplot(2,2,2);
imshow(noiseImg);
title(sprintf('Simulate Blur and Noise\nPSNR=%1.2f',psnr(I,noiseImg)));

% Define the Point-Spread Function (PSF).
if strcmp(type,'gaussian')
    PSF = fspecial('gaussian',noiseParam.hSize,noiseParam.sigma);
elseif strcmp(type,'motion')
    PSF = fspecial('motion',noiseParam.len,noiseParam.theta);
else
    PSF = fspecial('gaussian',3,0);
end

% Try restoration assuming no noise.
wnr2 = deconvwnr(noiseImg, PSF, k);
subplot(2,2,3);
imshow(wnr2);
title(sprintf('Restoration Using NSR = %1.4f\nPSNR=%1.2f',...
    k,psnr(I,wnr2)));

% Try restoration using a better estimate of the noise-to-signal-power 
% ratio.
estimatedNSR = noiseParam.noiseVar(i) / var(I(:));
% estimatedNSR = var(noiseImg(:)) / var(I(:));
wnr3 = deconvwnr(noiseImg, PSF, estimatedNSR);
subplot(2,2,4);
imshow(wnr3);
title(sprintf('Restoration Using Estimated NSR = %1.4f\nPSNR=%1.2f',...
    estimatedNSR,psnr(I,wnr3)));

% Capture the filtered image using the correct NSR.
filteredImage = wnr3;

return