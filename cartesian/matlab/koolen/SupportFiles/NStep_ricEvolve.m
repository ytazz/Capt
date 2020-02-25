function ret = NStep_ricEvolve(rIC0, rCMP, dt)
%NSTEP_ricEvolve
%   RET = NStep_ricEvolve(RIC0, RCMP, DT)
%   Returns the Instantaneous Capture Point location RET [x,y] after time 
%   interval DT, given the initial Instanteous Capture Point location
%   RICO [x,y] and a constant Centroidal Moment Pivot location RCMP [x,y].
%
%   This file is supplied as an addition to the draft paper:
%   "Capturability-Based Analysis and Control of Legged Locomotion, Part 2:
%   Application to Three Simple Gait Models" -
%
%   For further information, contact:
%   Tomas de Boer, tomasdeboer@gmail.com, or    
%   Twan Koolen,   tkoolen@ihmc.us
%
%   Copyright 2010, Delft BioRobotics Laboratory
%   Delft University of Technology
%   $Revision: 1.0 $  $Date: February 2010 $

ret = (rIC0 - rCMP) * exp(dt)  + rCMP;

end