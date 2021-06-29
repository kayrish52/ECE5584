function [GR,GB,RGR,RGB,RB,BGR,BGB,BR] = loadFilters()
% 'loadFilters' will generate the filters that are applied to the box in
% the image to compute the MHE demosaic.
% - INPUTS: none
% - OUTPUTS: GR - the filter for green blocks at red locations
%            GB - the filter for green blocks at blue locations
%            RGR - the filter for red blocks at green locations on red rows
%            RGB - the filter for red blocks at green locations on blue
%                  rows
%            RB - the filter for red blocks at blue locations
%            BGR - the filter for blue blocks at green locations on red
%                  rows
%            BGB - the filter for blue blocks at green locations on blue
%                  rows
%            BR - the filter for blue blocks at red locations
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

% Initialize the filter for green blocks at red locations.
GR = cell(3,1);
GR{1,1} = [0,0,-1,0,0;0,0,0,0,0;-1,0,4,0,-1;0,0,0,0,0;0,0,-1,0,0]./8;
GR{2,1} = [0,0,0,0,0;0,0,2,0,0;0,2,0,2,0;0,0,2,0,0;0,0,0,0,0]./8;
GR{3,1} = [0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0]./8;

% Initialize the filter for green blocks at red locations.
GB = cell(3,1);
GB{1,1} = [0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0]./8;
GB{2,1} = [0,0,0,0,0;0,0,2,0,0;0,2,0,2,0;0,0,2,0,0;0,0,0,0,0]./8;
GB{3,1} = [0,0,-1,0,0;0,0,0,0,0;-1,0,4,0,-1;0,0,0,0,0;0,0,-1,0,0]./8;

% Initialize the filter for red blocks at green locations on blue rows.
RGR = cell(3,1);
RGR{1,1} = [0,0,0,0,0;0,0,0,0,0;0,4,0,4,0;0,0,0,0,0;0,0,0,0,0]./8;
RGR{2,1} = [0,0,0.5,0,0;0,-1,0,-1,0;-1,0,5,0,-1;0,-1,0,-1,0;...
            0,0,0.5,0,0]./8;
RGR{3,1} = [0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0]./8;


% Initialize the filter for red blocks at green locations on blue rows.
RGB = cell(3,1);
RGB{1,1} = [0,0,0,0,0;0,0,4,0,0;0,0,0,0,0;0,0,4,0,0;0,0,0,0,0]./8;
RGB{2,1} = [0,0,-1,0,0;0,-1,0,-1,0;0.5,0,5,0,0.5;0,-1,0,-1,0;...
            0,0,-1,0,0]./8;
RGB{3,1} = [0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0]./8;

% Initialize the filter for red blocks at blue locations.
RB = cell(3,1);
RB{1,1} = [0,0,0,0,0;0,2,0,2,0;0,0,0,0,0;0,2,0,2,0;0,0,0,0,0]./8;
RB{2,1} = [0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0]./8;
RB{3,1} = [0,0,-1.5,0,0;0,0,0,0,0;-1.5,0,6,0,-1.5;0,0,0,0,0;...
           0,0,-1.5,0,0]./8;

% Initialize the filter for blue blocks at green locations on red rows.
BGR = cell(3,1);
BGR{1,1} = [0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0]./8;
BGR{2,1} = [0,0,0.5,0,0;0,-1,0,-1,0;-1,0,5,0,-1;0,-1,0,-1,0;...
            0,0,0.5,0,0]./8;
BGR{3,1} = [0,0,0,0,0;0,0,0,0,0;0,4,0,4,0;0,0,0,0,0;0,0,0,0,0]./8;

% Initialize the filter for blue blocks at green locations on blue rows.
BGB = cell(3,1);
BGB{1,1} = [0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0]./8;
BGB{2,1} = [0,0,-1,0,0;0,-1,0,-1,0;0.5,0,5,0,0.5;0,-1,0,-1,0;...
            0,0,-1,0,0]./8;
BGB{3,1} = [0,0,0,0,0;0,0,4,0,0;0,0,0,0,0;0,0,4,0,0;0,0,0,0,0]./8;

% Initialize the filter for blue blocks at red locations.
BR = cell(3,1);
BR{1,1} = [0,0,-1.5,0,0;0,0,0,0,0;-1.5,0,6,0,-1.5;0,0,0,0,0;...
           0,0,-1.5,0,0]./8;
BR{2,1} = [0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0;0,0,0,0,0]./8;
BR{3,1} = [0,0,0,0,0;0,2,0,2,0;0,0,0,0,0;0,2,0,2,0;0,0,0,0,0]./8;
end

