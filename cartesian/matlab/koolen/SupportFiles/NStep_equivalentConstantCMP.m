function ret = NStep_equivalentConstantCMP(CMP1, CMP2, dt1, dt2)
%NSTEP_equivalentConstantCMP
%   Computes the equivalent constant Centroidal Moment Pivot
%   RET = NStep_equivalentConstantCMP(CMP1, CMP2, DT1, DT2)
%   Computes the equivalent constant CMP for the scenario that the CMP is
%	kept at CMP1 [x,y] during DT1 and is then switched to CMP2 [x,y] and
%   held there for DT2.
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

w = (exp(dt2) - 1) / (exp(dt1 + dt2) - 1);

ret = (1 - w) * CMP1 + w * CMP2;

end