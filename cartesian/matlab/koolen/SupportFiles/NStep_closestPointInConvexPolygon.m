function ret = NStep_closestPointInConvexPolygon(point, convexPolygon)
%NSTEP_closestPointInConvexPolygon
%   Resturns the closest point inside a convex polygon for a point 
%   outside the polygon    
%   RET = NStep_closestPointInConvexPolygon(POINT, CONVEXPOLYGON)
%   Resturns the closest point inside a convex polygon (RET [x,y]) for a point 
%   POINT [x,y] outside a counterclockwise ordered polygon with only unique
%   points>
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


n = size(convexPolygon, 2);

% special case: convex polygon is just one point
if n == 1
    ret = convexPolygon;
    return
end

% first find closest vertex
closestVertexIndex = [];
closestVertexDistance = inf;
for i = 1 : n % can do this in a smarter way, but hey
    distance = norm(convexPolygon(:, i) - point);
    if distance < closestVertexDistance
        closestVertexIndex = i;
        closestVertexDistance = distance;
    end
end
closestVertex = convexPolygon(:, closestVertexIndex);

% then compute the outward pointing normals to the edges that are in
% contact with the closest vertex
[prevPerpVector, nextPerpVector] = perpVectors(convexPolygon, closestVertexIndex);

% check if the point is in between the two perpendicular vectors
angle1 = vectorAngle(prevPerpVector, point - closestVertex);
angle2 = vectorAngle(nextPerpVector, point - closestVertex);

pointBetweenPerpVectors = (angle1 >= 0) && (angle2 <= 0);
if pointBetweenPerpVectors
    ret = closestVertex;
else
    % need to project point onto an edge of the polygon
    if angle2 > 0
        % next edge is closest
        closestEdge = [closestVertex, convexPolygon(:, closestVertexIndex + 1)];
    else
        % previous edge is closest
        closestEdge = [convexPolygon(:, closestVertexIndex - 1), closestVertex];
    end
    
    ret = projectPointOnLine(point, closestEdge);
end

end % function



% @return normalized perpendicular vectors to the edges that connect to the
% given vertex
function [prevPerpVector, nextPerpVector] = perpVectors(convexPolygon, vertexIndex)
n = size(convexPolygon, 2);
prevVertexIndex = vertexIndex - 1;
if prevVertexIndex == 0
    prevVertexIndex = n;
end
nextVertexIndex = vertexIndex + 1;
if nextVertexIndex == n + 1
    nextVertexIndex = 1;
end


prevVertex = convexPolygon(:, prevVertexIndex);
thisVertex = convexPolygon(:, vertexIndex);
nextVertex = convexPolygon(:, nextVertexIndex);

if norm(nextVertex - thisVertex) == 0
    error('nextVertex == thisVertex');
end
if norm(thisVertex - prevVertex) == 0
    error('thisVertex - prevVertex == 0');
end

% first compute angle between perpendicular vectors:
prevVector = thisVertex - prevVertex;
nextVector = nextVertex - thisVertex;

prevPerpVector = [prevVector(2); -prevVector(1)];
nextPerpVector = [nextVector(2); -nextVector(1)];

prevPerpLength = norm(prevPerpVector);
nextPerpLength = norm(nextPerpVector);

prevPerpVector = prevPerpVector / prevPerpLength;
nextPerpVector = nextPerpVector / nextPerpLength;

end % perpVectors

% @return the projection of the point onto the line
function ret = projectPointOnLine(point, pointsOnLine)
      v0 = point - pointsOnLine(:, 1);
      v1 = pointsOnLine(:, 2) - pointsOnLine(:, 1);

      dotProduct = dot(v0, v1);
      lengthSquared = dot(v1, v1);

      alpha = dotProduct/lengthSquared;

      ret = pointsOnLine(:, 1) + alpha * v1;
end

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

function ret = vectorAngle(vec1, vec2)
crossProduct = vec1(1) * vec2(2) - vec1(2) * vec2(1);
angleSign = sign(crossProduct);
if angleSign == 0
    angleSign = 1;
end
dotProduct = dot(vec1 / norm(vec1), vec2 / norm(vec2));
dotProduct = clipToMinMax(dotProduct, -1, 1); % deal with round off errors.
ret = angleSign * acos(dotProduct);
end
