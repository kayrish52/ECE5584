function [wFiltImages,blurredImages,bm3dFiltImages,nlmFiltImages] = ...
    runHW3(imageFile,imageSelector,k)
% function [filteredImage] = runHW3(imageFile)
% - INPUTS:
% -- imageFile: this is the original input image.
% -- imageSelector: this selects the appropriate blurred image.
% -- k: this is the user input estimate of NSR
%
% - OUTPUTS:
% -- wFiltImages: this is the cell array of Wiener filtered images.
% -- blurredImages: this is the cell array of the blurred images.
% -- bm3dFiltImages: this is the cell array of BM3D filtered images.
% -- nlmFiltImages: this is the cell array of the NLM filtered images.

% Initialize the output images.
wFiltImages = 1;
blurredImages = 1;
bm3dFiltImages = 1;
nlmFiltImages = 1;

% Generate and display noisy images.
[blurredImages,noiseParam] = noisyImage(imageFile);

% Wait 1 second then close the figure.
uiwait(gcf,5);
close;

% Filter the noisy images using wiener filtering.
wFiltImages = wienerFilter(imageFile,blurredImages{2,imageSelector},...
    noiseParam,'gaussian',k,imageSelector);

% Wait 1 second then close the figure.
uiwait(gcf,5);
close;

% Filter the noisy images using BM3D filtering.
bm3dFiltImages = BM3D(imageFile,blurredImages{2,imageSelector},k,'np',1);

% Wait 1 second then close the figure.
uiwait(gcf,5);
close;

% Filter the noisy images using Non-Local Means filtering.
nlmFiltImages = nlmFilter(imageFile,blurredImages{2,imageSelector});

% Wait 1 second then close the figure.
uiwait(gcf,5);
close;
end

