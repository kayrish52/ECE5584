function [imDemosaic] = mhcDemosaic(img)
% mhc_demosaic(f) is a function that will compute the demosaic of an image
% using the Malvar-He-Cutler method of demosaicing.
% - INPUTS: img - the input image to compute the MHC demosaic on.
% - OUTPUTS: imDemosaic - the computed MHC demosaic.
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Load the image.
img = imread(img);

% Determine the height and width of the image.
h = size(img,1);
w = size(img,2);

% Overwrite the original image with a padded image.
img = padImage(img);

% Capture the adjusted height and width of the image.
h = size(img,1);
w = size(img,2);

% Initialize the demosaic image ad the Bayer Pattern image.
imDemosaic = zeros(size(img));
imgBayer = zeros(size(img));
imgBayer2d = zeros(h,w);

% Set the Bayer Pattern for the image.
[pattern] = setPattern(h,w);

% This will use the 'pattern' to create an image based on the Bayer
% pattern.
for i = 1:3
    % Isolate the reds, greens, or blues.
    [row,col] = find(pattern == i);
    
    % Capture the corresponding reds, greens, or blues.
    for j = 1:length(row)
        imgBayer(row(j),col(j),i) = img(row(j),col(j),i);
        imgBayer2d(row(j),col(j)) = img(row(j),col(j),i);
    end
end

% Create a new image from the built-in Demosaic function.
img2Demosaic = demosaic(uint8(imgBayer2d),'rggb');

% Load the filters.
[GR,GB,RGR,RGB,RB,BGR,BGB,BR] = loadFilters();

% This applies the MHC Demosaic for the center of the image.
for i = 3:h-2
    for j = 3:w-2
        patch = img(i-2:i+2,j-2:j+2,1:3);
        switch pattern(i,j)
            case 1
                imDemosaic(i,j,1) = img(i,j,1);
                imDemosaic(i,j,2) = sum(patch(:,:,1).*GR{1,1} + ...
                                    patch(:,:,2).*GR{2,1} + ...
                                    patch(:,:,3).*GR{3,1},'all');
                imDemosaic(i,j,3) = sum(patch(:,:,1).*BR{1,1} + ...
                                    patch(:,:,2).*BR{2,1} + ...
                                    patch(:,:,3).*BR{3,1},'all');
            case 2
                if mod(i,2) == 1
                    imDemosaic(i,j,1) = sum(patch(:,:,1).*RGR{1,1} + ...
                                    patch(:,:,2).*RGR{2,1} + ...
                                    patch(:,:,3).*RGR{3,1},'all');
                    imDemosaic(i,j,2) = img(i,j,2);
                    imDemosaic(i,j,3) = sum(patch(:,:,1).*BGR{1,1} + ...
                                    patch(:,:,2).*BGR{2,1} + ...
                                    patch(:,:,3).*BGR{3,1},'all');
                else
                    imDemosaic(i,j,1) = sum(patch(:,:,1).*RGB{1,1} + ...
                                    patch(:,:,2).*RGB{2,1} + ...
                                    patch(:,:,3).*RGB{3,1},'all');
                    imDemosaic(i,j,2) = img(i,j,2);
                    imDemosaic(i,j,3) = sum(patch(:,:,1).*BGB{1,1} + ...
                                    patch(:,:,2).*BGB{2,1} + ...
                                    patch(:,:,3).*BGB{3,1},'all');
                end
            case 3
                imDemosaic(i,j,1) = sum(patch(:,:,1).*RB{1,1} + ...
                                    patch(:,:,2).*RB{2,1} + ...
                                    patch(:,:,3).*RB{3,1},'all');
                imDemosaic(i,j,2) = sum(patch(:,:,1).*GB{1,1} + ...
                                    patch(:,:,2).*GB{2,1} + ...
                                    patch(:,:,3).*GB{3,1},'all');
                imDemosaic(i,j,3) = img(i,j,3);
        end
    end
end

% Plot the results of the algorithm and the built-in algorithm.
figure;
subplot(2,2,1);
imshow(img./255)
title('Original Image')
subplot(2,2,2);
imshow(imgBayer./255)
title({'Bayer Pattern','Of the Image'})
subplot(2,2,3);
imshow(imDemosaic./255)
title({'MHC Reconstruction','Of the Image'})
subplot(2,2,4);
imshow(double(img2Demosaic)./255)
title({'Built-In Demosaic Function','Of the Image'})


end
