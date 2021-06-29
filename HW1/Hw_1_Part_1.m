% This file will address ECE5584 Fall 2019 Homework 1, Problem 1.

% This clears the data and loads the appropriate dataset.
clear all

load('CIFAR-10/test_batch.mat')
labelNames = cell(10,1);
labelNames{1} = 'Airplane';
labelNames{2} = 'Automobile';
labelNames{3} = 'Bird';
labelNames{4} = 'Cat';
labelNames{5} = 'Deer';
labelNames{6} = 'Dog';
labelNames{7} = 'Frog';
labelNames{8} = 'Horse';
labelNames{9} = 'Ship';
labelNames{10} = 'Truck';

% This will call a function, parseIMS, to process the image data and
% convert it from 'uint8' RGB data to 'double' HSV data.
data = double(data);
data = data./255;
[ims,imsHSV] = parseIMS(data,labels);

% This will reshape all of the input images into a 1024x3 matrix for 
% random sampling and computing the nearest neighbors of each class.
rawHSVdata = cell(10,1000);
for i = 1:10
    for j = 1:1000
        rawHSVdata{i,j} = reshape(imsHSV{i,j},1024,3);
    end
end

% Initialize the centers of the HSV histograms.
nHBins = (0:8)./8;
hBinsCenter = (1:8)./9;
nSBins = (0:4)./4;
sBinsCenter = (1:4)./5;
nVBins = (0:2)./2;
vBinsCenter = (1:2)./3;

% This initializes the color histogram cell array for each image with
% respect to the calculated K-Means model.
xHSVColorHist = cell(10,1000);
hCount = cell(10,1000);
sCount = cell(10,1000);
vCount = cell(10,1000);
for i = 1:10
    for j = 1:1000
        [hCount{i,j},HCenter] = histcounts(rawHSVdata{i,j}(:,1),nHBins);
        [sCount{i,j},SCenter] = histcounts(rawHSVdata{i,j}(:,2),nSBins);
        [vCount{i,j},VCenter] = histcounts(rawHSVdata{i,j}(:,3),nVBins);
        xHSVColorHist{i,j} = [hCount{i,j},sCount{i,j},vCount{i,j}];
    end
end

plotHistograms;

% Reorganize the images and labels into a searchable format.
imgColorHist = cell(1000,1);
imgLib = cell(1000,1);
imgHSVLib = cell(1000,1);
labelList = [zeros(100,1);ones(100,1);2*ones(100,1);3*ones(100,1);...
    4*ones(100,1);5*ones(100,1);6*ones(100,1);7*ones(100,1);...
    8*ones(100,1);9*ones(100,1)];
count = 1;
for i = 1:10
    for j = 1:100
        imgColorHist{count} = xHSVColorHist{i,j};
        imgHSVLib{count} = imsHSV{i,j};
        imgLib{count} = ims{i,j};
        count = count + 1;
    end
end

% Calculate the distances between each pair of images.
distances = 99999*ones(1000,1000);
for i = 1:1000
    for j = 1:1000
        if i == j
            continue;
        else
            distances(i,j) = pdist2(imgColorHist{i},imgColorHist{j});
        end
    end
end


% This is a temporary stop point and load point.
save('Part1.mat','imgColorHist','imgLib','imgHSVLib','labelList',...
    'distances')
load('Part1.mat')

% Organize the distance vector.
distVec = [];
for i = 1:size(distances,2)
    distVec = [distVec;distances(:,i)];
end

% Capture the first image labels.
img1 = [];
for i = 1:1000
    img1 = [img1;i*ones(1000,1)];
end

% Capture the second image labels.
img2 = repmat((1:1000)',1000,1);

% Sort the distances. Capture the indexes. Sort the labels.
[distVec2,idx] = sort(distVec);
img1 = img1(idx);
img2 = img2(idx);
img1Labels = labelList(img1);
img2Labels = labelList(img2);
precision = zeros(10,1);

% Loop over each class for the top-10 retrievals.
for i = 1:10
    
    % Isolate the class labels for each class.
    clsIdx = find(img1Labels == (i-1));
    
    % Initialize the counter to zero.
    correct = 0;
    
    % Loop over the top-10 for the class under test. Plot and determine if
    % it correctly matched to it's class.
    for j = 1:10
        figure(1)
        
        subplot(1,2,1)
        imshow(imgLib{img1(clsIdx(j))})
        title(sprintf('Class: %s',labelNames{img1Labels(clsIdx(j))+1}))
        subplot(1,2,2)
        imshow(imgLib{img2(clsIdx(j))})
        title(sprintf('Class: %s',labelNames{img2Labels(clsIdx(j))+1}))
        sgtitle({sprintf('Top-10 Retrievals for %s',labelNames{i}),...
            sprintf('Rank: %i, Distance: %3.2f',j,distVec2(clsIdx(j)))})
        
        % Count the number ofcorrect matches.
        if img1Labels(clsIdx(j)) == img2Labels(clsIdx(j))
            correct = correct + 1;
        else
            continue;
        end
    end
    
    % Calculate the precision for the class.
    precision(i) = correct/10;
end
        




