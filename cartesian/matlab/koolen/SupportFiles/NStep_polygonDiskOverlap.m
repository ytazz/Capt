function [overlap, intersections] = NStep_polygonDiskOverlap(center, radius, polygon, pointsPerRadian)
%NSTEP_polygonDiskOverlap
%   Clip a polygon to a circle
%   [OVERLAP, INTERSECTIONS] = NStep_polygonDiskOverlap(CENTER, RADIUS, 
%   POLYGON, POINTSPERRADIAN) determines intersections between an input
%   polygon POLYGON and a circle with center CENTER and radius RADIUS. if
%   intersections exist, these locations are returned in INTERSECTIONS.
%   OVERLAP is the resulting clipped polygon which will exist of part of the
%   input polygon and part of the circle that will close the cliped polygon 
%   (if intersections exist). POINTSPERRADIAN specifies the plotting resolution
%   of the arc of the circle-part of the clipped polygon. 
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

% determine where the intersections are
intersectionEdgeIndices = [];
intersections = zeros(2, 0);
intersectionTypes = [];
for i = 1 : n
    index1 = i;
    if (i + 1) <= n
        index2 = i + 1;
    else
        index2 = 1;
    end

    vertex1 = polygon(:, index1);
    vertex2 = polygon(:, index2);
    
    intersectionParameters = circleLineSegmentIntersection(center, radius, vertex1, vertex2);
    
    % determine if we're really dealing with an intersection
    for j = 1 : length(intersectionParameters)
        t = intersectionParameters(j);
        if isreal(t) && (t >= 0) && (t <= 1)
            intersection = vertex1 + (vertex2 - vertex1) * t;
            
            % determine if unique
            epsilon = 1e-10;
            [isRepeatedIntersection, setMember] = ismember(intersection, intersections, epsilon);

            if isRepeatedIntersection
                intersection = setMember;
            end

            intersectionEdgeIndices = [intersectionEdgeIndices; i];
            intersections = [intersections, intersection];

            if isempty(intersectionTypes)
                % first time only
                if norm(vertex1 - center) <= radius
                    intersectionType = 'insideToOutside';
                else
                    intersectionType = 'outsideToInside';
                end
            else
                % just switch back and forth
                if strcmp(intersectionTypes(end, :), 'insideToOutside')
                    intersectionType = 'outsideToInside';
                else
                    intersectionType = 'insideToOutside';
                end
            end

            intersectionTypes = [intersectionTypes; intersectionType];
        end
    end
end


% now do something with those intersections
if (isempty(intersections))
    % the trivial cases
    % test one point:
    allVerticesInside = isInside(center, radius^2, polygon(:, 1));
    if (allVerticesInside)
        % all vertices inside
        overlap = polygon;
    else
        % all vertices outside
        
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
    end
    return;
else
    % the non-trivial case
    overlap = [];
    for i = 1 : size(intersections, 2)
        newPart = [];
        if strcmp(intersectionTypes(i, :), 'insideToOutside')
            % inside to outside intersection: 
            % need an arc to the next intersection
            if i == size(intersections, 2)
                j = 1;
            else
                j = i + 1;
            end
            
            angleSpan(1) = atan2(intersections(2, i) - center(2),...
                                 intersections(1, i) - center(1));
            angleSpan(2) = atan2(intersections(2, j) - center(2),...
                                 intersections(1, j) - center(1));
                             
            if angleSpan(2) < angleSpan(1)
                angleSpan(2) = angleSpan(2) + 2 * pi;
            end
                             
            [x, y] = NStep_arcPoints(center, radius, angleSpan, pointsPerRadian);
            newPart = [x; y];
        else
            % outside to inside intersection:
            % see if there are polygon points that are inside the disk
            if i == length(intersectionEdgeIndices)
                j = 1;
            else
                j = i + 1;
            end
            
            twoIntersectionsForOneEdge = intersectionEdgeIndices(i) == intersectionEdgeIndices(j);
            if ~twoIntersectionsForOneEdge
                % determine relevant vertex indices
                index1 = intersectionEdgeIndices(i) + 1;
                if index1 == n + 1
                    index1 = 1;
                end

                index2 = intersectionEdgeIndices(j);
                
                if index2 >= index1
                    % normal case
                    newPartIndices = index1 : index2;
                else
                    % 'special' case (index overflow)
                    newPartIndices = [index1 : n, 1 : index2];
                end

                newPart = polygon(:, newPartIndices);
            end
        end
        overlap = [overlap, newPart];
    end
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

function ret = circleLineSegmentIntersection(center, radius, point1, point2)
a = dot(point2 - point1, point2 - point1);
b = 2 * dot(point2 - point1, point1 - center);
c = dot(point1 - center, point1 - center) - radius^2;

t1 = (-b - sqrt(b^2 - 4 * a * c)) / (2 * a);
t2 = (-b + sqrt(b^2 - 4 * a * c)) / (2 * a);

ret = [t1; t2];
end

% faster dot product (assumes no complex numbers, vector size 2, no checks)
function ret = dot(a, b)
ret = a(1) * b(1) + a(2) * b(2);
end

%
function [ismember, setMember] = ismember(testMember, set, epsilon)
n = size(set, 2);
for i = 1 : n
    diff = testMember - set(:, i);
    if dot(diff, diff) < epsilon;
        ismember = true;
        setMember = set(:, i);
        return
    end
end
ismember = false;
setMember = [];
end
