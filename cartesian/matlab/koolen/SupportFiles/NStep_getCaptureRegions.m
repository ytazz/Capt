function captureRegions = NStep_getCaptureRegions(modelInfo, modelState, plotSettings, enableReactionMass, enableBoS)
%NSTEP_getCaptureRegions
%   Returns Cartesian coordinates to plot all capture regions
%   [CAPTUREREGIONS] = NStep_getCaptureRegions(MODELINFO, MODELSTATE,
%   PLOTSETTINGS, ENABLEREACTIONMASS, ENABLEBOS) returns the N-step capture
%   regions defined by the model variables MODELINFO, current model state 
%   MODELSTATE. PLOTSETTINGS describes the plotresolution and number of 
%   regions requested. Booleans ENABLEREACTIONMASS, 
%   ENABLEBOS set which model type should be selected for the 
%   calculations. CAPTUREREGIONS is a cell array that contains the number
%   of the region, and the x and y boundary points. 
%
%   This file is supplied as an addition to the draft paper:
%   "Analysis and Control of Legged Locomotion with Capture Points" 
%   - Part 2: Application to Three Simple Models -
%
%   For further information, contact:
%   Tomas de Boer, tomasdeboer@gmail.com, or    
%   Twan Koolen,   tkoolen@ihmc.us
%
%   Copyright 2010, Delft BioRobotics Laboratory
%   Delft University of Technology
%   $Revision: 1.0 $  $Date: February 2010 $



% 1) compute 1-step capture regions
oneStepCR = NStep_oneStepCaptureRegions(modelInfo, modelState, plotSettings, enableReactionMass, enableBoS);

% 2) compute relevant capture limits
captureLimits = NStep_captureLimits(modelInfo, plotSettings, false, enableBoS); % disable reaction mass: after the first step, we can't use it anymore
captureLimitsArray = [];
for i = 1 : length(captureLimits)
    captureLimitsArray = [captureLimitsArray; captureLimits{i}.limit];
end

% 3) use capture limits and 1-step capture region to compute N-step capture
% regions
NMax = plotSettings.NMax;
captureRegions = cell(NMax + 2, 1);

% first compute rough capture region using makeHalo
roughCaptureRegions = NStep_makeHalo([oneStepCR.boundary.x; oneStepCR.boundary.y], captureLimitsArray, plotSettings.pointsPerRadian);


for i = 1 : length(captureLimits)
    N = captureLimits{i}.N + 1;
    % then shave off the polygon elements outside the lMax circle  
    captureRegion = NStep_simplePolygonDiskOverlap(modelState.rAnkle, modelInfo.lMax, roughCaptureRegions{i}, plotSettings.pointsPerRadian);

    % pack
    captureRegions{i}.N = N;
    captureRegions{i}.boundary.x = captureRegion(1, :);
    captureRegions{i}.boundary.y = captureRegion(2, :);
end

end