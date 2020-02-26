function [overlap, intersections] = NStep_simplePolygonDiskOverlap(center, radius, polygon, pointsPerRadian)
%NSTEP_simplePolygonDiskOverlap
%   Clip a polygon to a circle, assumes one intersection per polygon 
%   edge and either zero or two intersections in total
%   [OVERLAP, INTERSECTIONS] = NStep_simplePolygonDiskOverlap(CENTER, RADIUS, 
%   POLYGON, POINTSPERRADIAN) determines intersections between an input
%   polygon POLYGON and a circle with center CENTER and radius RADIUS. if
%   intersections exist, these locations are returned in INTERSECTIONS.
%   OVERLAP is the resulting clipped polygon which will exist of part of the
%   input polygon and part of the circle that will close the cliped polygon 
%   (if intersections exist). POINTSPERRADIAN specifies the plotting resolution
%   of the arc of the circle-part of the clipped polygon. 
%   This is a simplified version of NStep_polygonDiskOverlap which assumes 
%   one intersection per polygon edge and either zero or two intersections 
%   in total.
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

n = size(polygon, 2);
radiusSquared = radius^2;

% find one index that's inside and one index that's outside
insideIndex = [];
outsideIndex = [];
for i = randperm(n) % random order will most likely be better than sequential order
    testVertex = polygon(:, i);
    inside = isInside(center, radiusSquared, testVertex);
    if (isempty(insideIndex)) && inside
        insideIndex = i;
    elseif (isempty(outsideIndex)) && (~inside)
        outsideIndex = i;
    end
    
    if ~isempty(insideIndex) && ~isempty(outsideIndex)
        break;
    end
end

% handle cases
if isempty(insideIndex) && isempty(outsideIndex)
    % no points, apparently
    overlap = zeros(2, 0);
    intersections = [];
elseif isempty(insideIndex)
    % all points outside
    
    % see if the circle is contained in the polygon: no intersections,
    % so we only need to test if one point of the circle is inside the
    % polygon
    circleInsidePolygon = inpolygon(center(1), center(2) + radius, polygon(1, :), polygon(2, :));

    if circleInsidePolygon
        [overlapX, overlapY] = NStep_arcPoints(center, radius, [0; 2 * pi], pointsPerRadian);
        overlap = [overlapX; overlapY];
    else
        overlap = zeros(2, 0);
    end
elseif isempty(outsideIndex)
    % all points inside
    overlap = polygon;
    intersections = [];
else
    % there should now be exactly two intersections (assumption)
    
    % binary search for vertices that specify line segments that intersect
    % with the circle
    leftInsideIndex = insideIndex;
    rightInsideIndex = insideIndex;
    leftOutsideIndex = outsideIndex;
    rightOutsideIndex = outsideIndex;
    
    while ~areAdjacentInCounterClockwiseOrder(rightInsideIndex, leftOutsideIndex, n)
        testIndex = getMidIndex(rightInsideIndex, leftOutsideIndex, n);
        testVertex = polygon(:, testIndex);
        if isInside(center, radiusSquared, testVertex)
            rightInsideIndex = testIndex;
        else
            leftOutsideIndex = testIndex;
        end
    end
    
    while ~areAdjacentInCounterClockwiseOrder(rightOutsideIndex, leftInsideIndex, n)
        testIndex = getMidIndex(rightOutsideIndex, leftInsideIndex, n);
        testVertex = polygon(:, testIndex);
        if isInside(center, radiusSquared, testVertex)
            leftInsideIndex = testIndex;
        else
            rightOutsideIndex = testIndex;
        end
    end
    
    % find exact intersections
    leftInsideVertex = polygon(:, leftInsideIndex);
    rightInsideVertex = polygon(:, rightInsideIndex);
    leftOutsideVertex = polygon(:, leftOutsideIndex);
    rightOutsideVertex = polygon(:, rightOutsideIndex);
    intersections(:, 1) = circleLineSegmentIntersection(center, radius, rightInsideVertex, leftOutsideVertex);
    intersections(:, 2) = circleLineSegmentIntersection(center, radius, rightOutsideVertex, leftInsideVertex);
    
    % compute arc part
    angleSpan(1) = atan2(intersections(2, 1) - center(2),...
        intersections(1, 1) - center(1));
    angleSpan(2) = atan2(intersections(2, 2) - center(2),...
        intersections(1, 2) - center(1));
    % solve epsilon problem
    epsilon = 1e-10;
    if abs(diff(angleSpan)) < epsilon
        angleSpan(2) = angleSpan(1);
    end
    % solve 'overflow' problem
    if diff(angleSpan) < 0
        angleSpan(2) = angleSpan(2) + 2 * pi;
    end

    
    [arcX, arcY] = NStep_arcPoints(center, radius, angleSpan, pointsPerRadian);
    arcPart = [arcX; arcY];
    
    % compute polygon part
    index1 = leftInsideIndex;
    index2 = rightInsideIndex;

    if index2 >= index1
        % normal case
        polygonPartIndices = index1 : index2;
    else
        % 'special' case (index overflow)
        polygonPartIndices = [index1 : n, 1 : index2];
    end
    polygonPart = polygon(:, polygonPartIndices);
    
    % pack it up
    overlap = [arcPart, polygonPart];
end
end


function ret = isInside(center, radiusSquared, point)
d = point - center;
if (d(1)^2 + d(2)^2) <= radiusSquared
    ret = true;
else
    ret = false;
end
end

function ret = getMidIndex(leftIndex, rightIndex, numberOfIndices)

if (leftIndex <= rightIndex)
    % simple case
    ret = floor(leftIndex + (rightIndex - leftIndex) / 2);
else
    % 'overflow' case
    ret = floor(leftIndex + ((rightIndex + numberOfIndices) - leftIndex) / 2);
    if ret > numberOfIndices
        ret = mod(ret, numberOfIndices);
    end
end
end

function ret = areAdjacentInCounterClockwiseOrder(index1, index2, numVertices)
indexToTest = index1 + 1;
if indexToTest > numVertices
    indexToTest = 1;
end
ret = (indexToTest == index2);
end

function ret = circleLineSegmentIntersection(center, radius, point1, point2)
a = dot(point2 - point1, point2 - point1);
b = 2 * dot(point2 - point1, point1 - center);
c = dot(point1 - center, point1 - center) - radius^2;

t1 = (-b - sqrt(b^2 - 4 * a * c)) / (2 * a);
t2 = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a);

if (t1 >= 0) && (t1 <= 1)
    t = t1;
else
    t = t2;
end

ret = point1 + (point2 - point1) * t;
end

% faster dot product (assumes no complex numbers, vector size 2, no checks)
function ret = dot(a, b)
ret = a(1) * b(1) + a(2) * b(2);
end
