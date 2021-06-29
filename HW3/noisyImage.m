function [noisyIm,noiseParam] = noisyImage(imFile)
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% function [noisyIm] = noisyImage(im)
% - INPUTS
% -- im: The input image that will have noise added for purposes of HW3.
%
% - OUTPUTS
% -- noisyIm: The resulting cell array of output images with varying 
%             degrees of additive noise.
% -- noiseParam: This is a struct containing all of the parameters used to
%                generate the noisy images. The parameters in noiseParam
%                include:
%                - len: The length of the motion vector.
%                - theta: The angle of the motion vector.
%                - hSize: The kernel size of the Gaussian Blurring filter.
%                - sigma: The variance of the Gaussian Blurring filter.
%                - noiseMean: The mean of additive Gaussian noise.
%                - noiseVar: The variance of additive Gaussian noise.

% Convert the image to grayscale.
im = imread(imFile);
im = im2double(im);

% Create a 'motion' blurring filter.
len = 20;
theta = 10;
PSF_motion = fspecial('motion', len, theta);
h_1 = imfilter(im, PSF_motion, 'conv', 'replicate');

% Create a 'Gaussian' blurring filter.
HSIZE = 5;
SIGMA = 20;
PSF_g = fspecial('gaussian',HSIZE,SIGMA);
h_2 = imfilter(im, PSF_g, 'conv', 'replicate');

% Initialize the noise parameters.
noiseMean = 0;
noiseVar = [0.0001 0.005 0.01];

% Aggregate the noise parameters.
noiseParam.len = len;
noiseParam.theta = theta;
noiseParam.hSize = HSIZE;
noiseParam.sigma = SIGMA;
noiseParam.noiseMean = noiseMean;
noiseParam.noiseVar = noiseVar;

% Create the figure. Loop over the additive noise and filtering
% parameters.
figure(1);
noisyIm = cell(2,3);
for k=1:3
    % Add noise to the motion blurred image. Calculate Peak SNR.
    camBlurNoise1 = imnoise(h_1, 'gaussian', noiseMean, noiseVar(k));
    noisyIm{1,k} = camBlurNoise1;
    psnr1 = psnr(im,camBlurNoise1);
    
    % Plot the motion blurred image.
    subplot(2,3,k);
    imshow(camBlurNoise1);
    title(sprintf('Motion Noise: %1.4f\nPSNR: %1.3f db',...
        noiseVar(k), psnr1));

    % Add noise to the Gaussian smoothed image. Calculate Peak SNR.
    camBlurNoise2 = imnoise(h_2, 'gaussian', noiseMean, noiseVar(k));
    noisyIm{2,k} = camBlurNoise2;
    psnr1 = psnr(im,camBlurNoise2);
    subplot(2,3,k+3);
    imshow(camBlurNoise2);
    title(sprintf('Gaussian Noise: %1.4f\nPSNR: %1.3f db',...
        noiseVar(k), psnr1));
end
sgtitle(sprintf('Noisy Images for %s',imFile))