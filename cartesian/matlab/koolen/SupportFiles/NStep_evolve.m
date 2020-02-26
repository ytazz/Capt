function [rIC, r2, r, rd] = NStep_evolve(rIC0, r20, rCMP, dt)
%NSTEP_evolve
%   Simulates the dynamics of the 3D-LIPM   
%   [RIC, R2, R, RD] = NSTEP_EVOLVE(RIC0, R20, RCMP, DT)
%   Returns the new CoM position R [x,y], velocity RD [dx,dy], the 
%   instantaneous capture point location RIC [x,y], the second index R2
%   [x,y] of the state vector as presented in paper eq. (10) of the 3D-LIPM. 
%   Inputs are a time interval DT, initial CoM position RIC0 [x,y] and the
%   second index R20 [x,y] of the state vector, and a constant CMP 
%   position RCMP.
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

rIC = NStep_ricEvolve(rIC0, rCMP, dt);
r2 = NStep_r2Evolve(r20, rCMP, dt);

r = (rIC + r2) / 2;
rd = (rIC - r2) / 2;

end

