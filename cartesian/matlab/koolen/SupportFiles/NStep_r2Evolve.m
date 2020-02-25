function ret = NStep_r2Evolve(r20, rCMP, dt)
%NSTEP_r2Evolve
%   Returns the second index of the state vector as presented in
%   paper eq. (10)
%   RET = NStep_r2Evolve(R20, RCMP, DT) returns the second index of the 
%   state vector RET [x,y] as presented in paper eq. (10) after time 
%   interval DT. Given the initial second state R20 [x,y] and a constant 
%   Centroidal Moment Pivot location RCMP [x,y]. 
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

ret = (r20  - rCMP) * exp(-dt) + rCMP;

end