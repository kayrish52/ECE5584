function [pattern] = setPattern(h,w)
% 'setPattern' is a function that will establish a symbolic Bayer Pattern 
% that will be used to compute the MHC demosaic of an image.
% - INPUTS: h - the height of an image in pixels.
%           w - the width of an image in pixels.
% - OUTPUTS: pattern - the Bayer Pattern for an image.

% Initialize the pattern to all zeros.
pattern = zeros(h,w);

% Initialize the row styles of the Bayer Pattern. This will be used to
% quickly populate the pattern.
rowStyle = zeros(2,w);

% Initialize a red row.
rowStyle(1,1:2:end) = 1;
rowStyle(1,2:2:end) = 2;

% Initialize a blue row.
rowStyle(2,1:2:end) = 2;
rowStyle(2,2:2:end) = 3;

% Loop over to set the row style for each case.
for i = 1:h
    if mod(i,2) == 1
        pattern(i,:) = rowStyle(1,:);
    else
        pattern(i,:) = rowStyle(2,:);
    end
end

end

