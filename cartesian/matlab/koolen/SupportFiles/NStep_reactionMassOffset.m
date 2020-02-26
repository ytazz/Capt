function rC = NStep_reactionMassOffset(modelState,modelInfo,Mode,t0,t_step,t1,t2)
%NSTEP_reactionMassOffset
%   Returns the magnitude of rC as presented in paper eq. (37)
%   RC = NStep_reactionMassOffset(MODELSTATE,MODELINFO,MODE,T0,T_STEP,T1,T2)
%   returns the magnitude of rC as presented in eq. (37) depending on 
%   the model state MODELSTATE, model paramters MODELINFO. MODE can be set
%   to 'vector' or 'scalar' depending on if the magnitude of RC or direction and 
%   magnitude is desired. TO is the time at which the reaction mass starts 
%   accelerating. TSTEP is the time at which the step occurs. T1 is the time
%   the reaction mass starts decelerating. T2 is the time when the reaction
%   mass has zero velocity again.
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

switch Mode
    case 'vector' % case: an absolute location
        tau = -modelState.tauWindmill;
        dir = [cos(modelState.windmillAngle); sin(modelState.windmillAngle)];
    case 'scalar' % case: a relative offset from rIC in any direction
        t1 = t0 + modelInfo.dTWindmill;
        t2 = t0 + 2*modelInfo.dTWindmill;
        tau = modelInfo.tauMax;
        dir = 1;
end

rC  = tau * dir * (exp(-t0) - 2 *exp(-t1) + exp(-t2) ) * exp(t_step);
    
end