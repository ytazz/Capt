function [x, y] = NStep_arcPoints(center, radius, angleSpan, pointsPerRadian)
%NSTEP_arcpoints
%   Returns Cartesian coordinates of points on an arc with radius r at
%   a specified resolution.
%   [X, Y] = NStep_arcPoints(CENTER, RADIUS, ANGLESPAN, PPOINTSPERRADIAN)
%   creates an arc with the center described by the 2D vector CENTER, with
%   radius RADIUS between the angles [rad] described in the 2D vector
%   ANGLESPAN. POINTSPERRADIAN determines the amount of points to create
%   per radian, which are returned in the columnvectors X,Y, representing
%   the (x,y) location of all points. 
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
  
nPoints = max(floor(diff(angleSpan) * pointsPerRadian), 2);
angles = linspace(angleSpan(1), angleSpan(2), nPoints);

if diff(angleSpan) == 2*pi;     % only use unique points
    angles = angles(1:(end-1));
end

x = center(1) + cos(angles) * radius;
y = center(2) + sin(angles) * radius;



end