function [BoSPoints, BoSMaxD, rCoPOpt] = NStep_BoSCreate(rFoot,Action,BoSAngle)
%NSTEP_BoSCreate
%   Returns Cartesian coordinates of points on a user defined convex
%   foot
%   [BOSPOINTS, BOSMAXD, RCOPOPT] = NStep_BoSCreate(RFOOT,ACTION,BOSANGLE)
%   return absolute coordinates of a convex foot at absolute location described
%   by the 2D vector RFOOT. The orientation of the foot is set by the
%   BOSANGLE. The coordinates of all points are returned in the 2D VEctor
%   BOSPOINTS, BOSMAXD is the maximum radius of the convex foot. RCOPOPT
%   returns the absolute location of the point that is located at the
%   maximum radius of the foot. 
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

% returns Cartesian coordinates of points on a user defined convex foot


if strcmp(Action,'Give') || strcmp(Action,'Create') || strcmp(Action,'Rotate')
    
    if strcmp(Action, 'Create')
        % Load ans scale any user defined BoS point cloud
    else
        % Use default foot
        BOS = load('NStep_BoSPoints.mat');
    end
    % Make a nice sized BoS with nice ankle location :)
    BoSPoints  = (BOS.BoS + repmat([0.2;-0.05],1,size(BOS.BoS,2))) * 0.3 ;

    % Use only points that span a convex hull
    Index       = convhull(BoSPoints(1,:),BoSPoints(2,:));
    BoSPoints   = BoSPoints(:,Index);
    BoSPoints(:,end) =[];

    % Derive angles of all points relative to horizontal line, CC rotation is positive
    BoSAngles      = cart2pol(BoSPoints(1,:),BoSPoints(2,:));
    [BoSAngles, i] = sort(BoSAngles);
    BoSPoints      = BoSPoints(:,i);
    
    % Find most distal point of BoS border from rFoot
    NormVecArray    = BoSPoints - repmat([0;0],1,size(BoSPoints,2));
    [BoSMaxD BoSMaxDIndex] = max(NStep_gridfun(@norm,num2cell(NormVecArray,1),2));

    % Make sure most distal BoS point is rotated towards  rIC
    BoSAngle = BoSAngle - BoSAngles(BoSMaxDIndex);
    
    if strcmp(Action,'Rotate')
        R = [cos(BoSAngle), -sin(BoSAngle);
            sin(BoSAngle),  cos(BoSAngle)];
     
        BoSPoints = R * BoSPoints;
    end
        
    % Make absolute coordinate
    BoSPoints      = BoSPoints + repmat(rFoot,1,size(BoSPoints,2));
    rCoPOpt        = BoSPoints(:,BoSMaxDIndex);
end