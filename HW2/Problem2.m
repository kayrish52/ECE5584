clear;
%% Question 2:
im1 = imread('flag.jpg');
im1 = rgb2gray(im1);
im2 = imread('GoldenGate.jpg');
im2 = rgb2gray(im2);

% blurred image generation
n_pts=[5 10 15 20 25 30];
for k=1:6
    h=fspecial('disk',n_pts(k));

    % Blur the Flag Image
    im_1{k}=imfilter(im1,h,'replicate');
    n_im1{k} = imnoise(im_1{k}, 'poisson');
    [psnr1(k), snr1(k)] = psnr(im1, n_im1{k});
%     figure(1);
%     subplot(2,3,k);
%     imshow(n_im1{k});
%     title(sprintf('psnr=%1.2f db', psnr1(k)));
    
    % Blur the Golden Gate Image
    im_2{k}=imfilter(im2,h,'replicate');
    n_im2{k} = imnoise(im_2{k}, 'poisson');
    [psnr2(k), snr2(k)] = psnr(im2, n_im2{k}); 
%     figure(2);
%     subplot(2,3,k);
%     imshow(n_im2{k});
%     title(sprintf('psnr=%1.2f db', psnr2(k)));
end
% fprintf('\n psnr=['); 
% fprintf('%1.2fdb ', psnr1);

% % Bilateral Filter: choose different heat kernels (degree of smoothing
% % and sigma)
% DoS = [200 800 1400]; % Degree of Smoothing
% sigma = [1 2 4 8];
% pos = 1;
% for dos = DoS
%     for sig = sigma
%         filtered_image = imbilatfilt(n_im1{6},dos,sig);
%         
%         psnr_g1 = psnr(im1, filtered_image);
%         figure (3);
%         subplot(3,4,pos)
%         imshow(filtered_image);
%         title(sprintf('psnr=%1.2f db', psnr_g1));
%         fprintf('\n psnr=[');
%         fprintf('%1.2fdb ', psnr_g1);
%         
%         filtered_image = imbilatfilt(n_im2{6},dos,sig);
%         psnr_g2 = psnr(im2, filtered_image);
%         figure (4);
%         subplot(3,4,pos)
%         imshow(filtered_image);
%         title(sprintf('psnr=%1.2f db', psnr_g2));
%         fprintf('\n psnr=[');
%         fprintf('%1.2fdb ', psnr_g2);
%         pos = pos + 1;
%     end
% end

% Guided Filtering: choose appropriate image as guide (original image,
% blurred image and irrelevent image)
img = 6;
% guided_image = im1;
% guided_image = im_1{img};
guided_image = n_im1{img};

filtered_image = imguidedfilter(n_im1{img},guided_image);
psnr_g = psnr(im1, filtered_image);
psnr_i = psnr(im1,n_im1{img});
figure (1);
subplot(1,3,1)
imshow(guided_image);title('Guiding Image')
subplot(1,3,2)
imshow(n_im1{img})
title(sprintf('psnr=%1.2f db', psnr_i));
subplot(1,3,3)
imshow(filtered_image);title(sprintf('psnr=%1.2f db', psnr_g));
fprintf('\n psnr=['); 
fprintf('%1.2fdb ', psnr_g);

% guided_image = im2;
% guided_image = im_2{img};
guided_image = n_im2{img};

filtered_image = imguidedfilter(n_im2{img},guided_image);
psnr_g = psnr(im2, filtered_image); 
psnr_i = psnr(im2,n_im2{img});
figure (2);
subplot(1,3,1)
imshow(guided_image);title('Guiding Image')
subplot(1,3,2)
imshow(n_im2{img})
title(sprintf('psnr=%1.2f db', psnr_i));
subplot(1,3,3)
imshow(filtered_image);title(sprintf('psnr=%1.2f db', psnr_g));
fprintf('\n psnr=['); 
fprintf('%1.2fdb ', psnr_g);

