function [img2] = padImage(img)
% 'padImage' will pad the input image with mirrored values at the edges and
% the corners.
% - INPUTS: img - the image to be padded.
% - OUTPUTS: img2 - the original image after it is padded.

% Determine the height and width of the image.
h = size(img,1);
w = size(img,2);

% Initialize and pad the image appropriately for processing.
img2 = zeros(h+4,w+4,3);

% Copy the image to the center of a new block.
img2(3:h+2,3:w+2,:) = img;

% Pad the right and left edges with mirrored data.
img2(1,3:w+2,:) = img(1,:,:);
img2(2,3:w+2,:) = img(1,:,:);
img2(h+3,3:w+2,:) = img(h,:,:);
img2(h+4,3:w+2,:) = img(h,:,:);

% Pad the top and bottom edges with mirrored data.
img2(3:h+2,1,:) = img(:,1,:);
img2(3:h+2,2,:) = img(:,1,:);
img2(3:h+2,w+3,:) = img(:,w,:);
img2(3:h+2,w+4,:) = img(:,w,:);

% Rewrite the top left corner of the image.
img2(1,1,:) = img2(3,3,:);
img2(1,2,:) = img2(3,3,:); 
img2(2,1,:) = img2(3,3,:);
img2(2,2,:) = img2(3,3,:);

% Rewrite the top right corner of the image.
img2(1,end,:) = img2(3,end-2,:);
img2(2,end,:) = img2(3,end-2,:);
img2(1,end-1,:) = img2(3,end-2,:);
img2(2,end-1,:) = img2(3,end-2,:);

% Rewrite the bottom left corner of the image.
img2(end-1,1,:) = img2(end-2,3,:);
img2(end-1,2,:) = img2(end-2,3,:);
img2(end,1,:) = img2(end-2,3,:);
img2(end,2,:) = img2(end-2,3,:);

% Rewrite the bottom right corner of the image.
img2(end-1,end-1,:) = img2(end-2,end-2,:);
img2(end-1,end,:) = img2(end-2,end-2,:);
img2(end,end-1,:) = img2(end-2,end-2,:);
img2(end,end,:) = img2(end-2,end-2,:);

end

