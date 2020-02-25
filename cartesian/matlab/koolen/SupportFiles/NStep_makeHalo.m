function halo = NStep_makeHalo(vertices, radii, pointsPerRadian)
%NSTEP_makeHalo
%   Returns multiple offset paths (halos) around a convex path 
%   HALO = NStep_makeHalo(VERTICES, RADII, POINTSPERRADIAN) creates mutiple
%   halos around a convex hull with vertices VERTICES [x1 x2 ...;
%   y1 y2 ...] at offset distances RADII. NUMBEROFPOINTSPERRADIAN
%   sets the number of points per radian of any created arc. HALO returns
%   the set of HALOS as a cell, with each cell specifying the HALO
%   with points [x1 x2 ...;  y1 y2 ...]
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

n = size(vertices, 2);
m = length(radii);

halo = cell(m, 1);
if (n == 0) || (m == 0) % halo is nothing
    halo = [];
elseif n == 1 % halo is a circle
    for j = 1 : m
        radius = radii(j);
        [x, y] = NStep_arcPoints(vertices, radius, [0, 2 * pi], pointsPerRadian);
        halo{j} = [x; y];
    end
else % non-trivial cases
    % scrap together bits of the halo
    for i = 2 : n - 1
        haloBits = haloBit(vertices(:, i - 1), vertices(:, i), vertices(:, i + 1), radii, pointsPerRadian);
        for j = 1 : m
            halo{j} = [halo{j} haloBits{j}];
        end
    end

    % don't forget the last two
    haloBitsN = haloBit(vertices(:, end - 1), vertices(:, end), vertices(:, 1), radii, pointsPerRadian);
    haloBitsNPlus1 = haloBit(vertices(:, end), vertices(:, 1), vertices(:, 2), radii, pointsPerRadian);
    
    for j = 1 : m
        halo{j} = [halo{j} haloBitsN{j} haloBitsNPlus1{j}];
    end
end

% just return a normal array instead of a cell array if m = 1
if m == 1
    halo = halo{1};
end

end % makeHalo

function ret = haloBit(prevVertex, thisVertex, nextVertex, radii, pointsPerRadian)
if norm(nextVertex - thisVertex) == 0
    error('nextVertex == thisVertex');
end
if norm(thisVertex - prevVertex) == 0
    error('thisVertex - prevVertex == 0');
end

m = size(radii);

% first compute angle between perpendicular vectors:
prevVector = thisVertex - prevVertex;
nextVector = nextVertex - thisVertex;

prevPerpVector = [prevVector(2); -prevVector(1)];
nextPerpVector = [nextVector(2); -nextVector(1)];

prevPerpLength = norm(prevPerpVector);
nextPerpLength = norm(nextPerpVector);

prevPerpVector = prevPerpVector / prevPerpLength;
nextPerpVector = nextPerpVector / nextPerpLength;

dotProduct = dot(prevPerpVector, nextPerpVector);
dotProduct = clipToMinMax(dotProduct, -1, 1); % deal with round off errors.

perpAngle = acos(dotProduct);
% works because perpendicular vectors have been normalized

% cases based on angle between perpendicular vector:
epsilon = 1e-10;
if abs(perpAngle) < epsilon % angle is zeroish
    % going to return only one point
    for i = 1 : m
        radius = radii(i);
        ret{i} = thisVertex + radius * prevPerpVector;
        % works because perpendicular vectors have been normalized.
        % Could also have used nextPerpVector, since they're about the
        % same.
    end

elseif perpAngle > 0 % angle is positive
    % going to return some points on an arc
    % compute angle between x-axis and perpendicular vectors

    prevPerpCrossProduct = prevPerpVector(2);
    prevPerpAngleSign = sign(prevPerpCrossProduct);
    if prevPerpAngleSign == 0
        prevPerpAngleSign = 1;
    end
    prevPerpDotProduct = prevPerpVector(1);
    prevPerpAngle = prevPerpAngleSign * acos(prevPerpDotProduct);
    nextPerpAngle = prevPerpAngle + perpAngle;

    angleSpan(1) = min(prevPerpAngle + 1 / pointsPerRadian, nextPerpAngle);
    %             angleSpan(1) = prevPerpAngle;
    angleSpan(2) = nextPerpAngle;

    % create arc
    [normalizedHaloPointsX, normalizedHaloPointsY] = NStep_arcPoints([0; 0], 1, angleSpan, pointsPerRadian);
    for i = 1 : m
        radius = radii(i);
        if radius == 0
            % no sense creating more points than necessary:
            haloPointsX = thisVertex(1);
            haloPointsY = thisVertex(2);
        else
            haloPointsX = thisVertex(1) + radius * normalizedHaloPointsX;
            haloPointsY = thisVertex(2) + radius * normalizedHaloPointsY;
        end
        ret{i} = [haloPointsX; haloPointsY];
    end

else % angle is negative
    error('Should not get here: input vertices should form a convex hull');
end
end % haloBit

% faster dot product (assumes no complex numbers, vector size 2, no checks)
function ret = dot(a, b)
ret = a(1) * b(1) + a(2) * b(2);
end

function ret = clipToMinMax(number, min, max)
if number > max
    ret = max;
elseif number < min
    ret = min;
else
    ret = number;
end
end
