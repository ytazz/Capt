function captureLimits = NStep_captureLimits(modelInfo, plotSettings, enableReactionMass, enableBoS)
%NSTEP_captureLimits
%   Returns the N-Step capture limits for a given model
%   [CAPTURELIMITS] = NStep_captureLimits(MODELINFO, PLOTSETTINGS,
%   ENABLEREACTIONMASS, ENABLEBOS) returns the N-step capture limits for
%   the model settings as in the MODELINFO struct. PLOTSETTINGS describes the
%   plotresolution and number of regions requested. ENABLEREACTIONMASS, 
%   ENABLEBOS booleans set which model type should be selected for the 
%   calculations. CAPTURELIMITS is a cell array of which each cell
%   represents a capture region. It contains the fields N of the number of 
%   the capture region and the limit; which is the absolute distance to the
%   0-step capture region in state space.
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

lMax = modelInfo.lMax;
NMax = plotSettings.NMax;
if enableBoS
    rBoS = modelInfo.maxBoSRadius;
else
    rBoS = 0;
end



% compute limits
expMinusDeltaTMin = exp(-modelInfo.dTmin);
captureLimits   = cell(NMax + 2, 1);
NArray          = [0 : NMax, Inf];

% N = 0
captureLimits{1}.N = 0;
captureLimits{1}.limit = rBoS;

% N finite
for i = 2 : length(NArray) - 1
    N                       = NArray(i);
    captureLimits{i}.N = N;
    captureLimits{i}.limit = (lMax + captureLimits{i - 1}.limit - rBoS) * expMinusDeltaTMin + rBoS;
end

% N infinite
captureLimits{end}.N = inf;
if enableBoS
    captureLimits{end}.limit = rBoS + lMax * expMinusDeltaTMin / (1 - expMinusDeltaTMin);
else
    captureLimits{end}.limit = lMax * expMinusDeltaTMin / (1 - expMinusDeltaTMin);
end

end