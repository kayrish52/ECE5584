clear;

%% Question 1:
% Load the image.
im1 = imread('flag.jpg');
im2 = imread('GoldenGate.jpg');

% Convert images to grayscale.
im1=rgb2gray(im1);
im2=rgb2gray(im2);

% Generate noise scales and inject noise into images.
v=[2 4 8 12 16 20]; 
n_pts=[0.5 1 2 4 8 12]/100;
for k=1:6
    n_im1{k} = imnoise(im1, 'salt & pepper', n_pts(k));
    n_im2{k} = imnoise(im2, 'salt & pepper', n_pts(k));
    psnr1_1(k) = psnr(im1, n_im1{k});
    psnr1_2(k) = psnr(im2, n_im2{k});
%     figure(41);
%     subplot(2,3,k);
%     imshow(n_im1{k});
%     title(sprintf('psnr=%1.2f db', psnr1_1(k)));
%     figure(42);
%     subplot(2,3,k);
%     imshow(n_im2{k});
%     title(sprintf('psnr=%1.2f db', psnr1_2(k)));
end
% fprintf('\n psnr=['); 
% fprintf('%1.2fdb ', psnr1_1);
% fprintf(']\n');

% % Gaussian Filter: adjust sigma, kernel size and padding methods, in the 
% % meantime, explain the performance with different configurations.
% % h = fspecial('gaussian',hsize,sigma)
% hsize = [3.0 5.0 7.0];
% sigma = [0.5 1.0 2.0 3.0 5.0];
% 
% pos = 1;
% for kSize = hsize
%     for sig = sigma
%         h = fspecial('gaussian',kSize,sig);
%         
%         % Padding Method: Zero, Replicate and Circular Padding
% %         filtered_image = imfilter(n_im1{1},h);
% %         filtered_image = imfilter(n_im1{1},h, 'circular');
%         filtered_image = imfilter(n_im1{6},h, 'replicate');
%         psnr_g1 = psnr(im1, filtered_image);
% %         filtered_image = imfilter(n_im2{1},h);
% %         filtered_image = imfilter(n_im2{1},h, 'circular');
% %         filtered_image = imfilter(n_im2{2},h, 'replicate');
% %         psnr_g2 = psnr(im2, filtered_image);
%         figure (1);
%         subplot(3,5,pos)
%         imshow(filtered_image);
%         title(sprintf('psnr=%1.2f db', psnr_g1));
% %         title(sprintf('psnr=%1.2f db', psnr_g2));
%         fprintf('\n psnr=[');
%         fprintf('%1.2fdb ', psnr_g1);
% %         fprintf('%1.2fdb ', psnr_g2);
%         
%         pos = pos+1;
%     end
% end

hsize = [3.0 5.0 7.0];
pos = 1;
for kSize = hsize
    % Box Filter: Adjust Filter Size and Explain
    % B = imboxfilt(A,filterSize) filters image A with a 2-D box filter 
    % with size specified by filterSize.
    filtered_image=imboxfilt(n_im1{6},kSize);
    psnr_g1 = psnr(im1, filtered_image);
%     filtered_image=imboxfilt(n_im2{6},kSize);
%     psnr_g2 = psnr(im2, filtered_image);
    figure (2);
    subplot(1,3,pos)
    imshow(filtered_image);
    title(sprintf('psnr=%1.2f db', psnr_g1));
%     title(sprintf('psnr=%1.2f db', psnr_g2));
    fprintf('\n psnr=[');
    fprintf('%1.2fdb ', psnr_g1);
%     fprintf('%1.2fdb ', psnr_g2);

    pos = pos + 1;
end

