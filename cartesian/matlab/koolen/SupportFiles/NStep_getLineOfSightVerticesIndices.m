function ret = NStep_getLineOfSightVerticesIndices(convexPolygon, observerPoint)
%NSTEP_getLineOfSightVerticesIndices
%   Returns vertices that are in line of sight from an observerpoint
%   RET = NStep_getLineOfSightVerticesIndices(CONVEXPOLYGON, OBSERVERPOINT)
%   Returns an array of indices of the convex clockwise ordered polygon 
%   CONVEXPOLYGON  [x1 x2 .. ; y1 y2 ..] which are in line of sight of 
%   an OBSERVERPOINT [x,y] 
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


numberOfEdges = size(convexPolygon, 2);

if numberOfEdges == 1 % single point
    ret = [convexPolygon, convexPolygon];
    return;
end

% figure(); hold on;
% for i = 1 : numberOfEdges
%     isEdgeEntering(convexPolygon, i, observerPoint);
% end
% hold off;

% At any time we'll hold onto 4 edge indices. -1 signifies not found yet:
leavingRightEdge = -1;
leavingLeftEdge = -1;
enteringLeftEdge = -1;
enteringRightEdge = -1;

% First choose an edge at random. An edge index will be signified by
% its first vertex in clockwise order.
firstEdgeIndex = floor(numberOfEdges / 2) + 1;

if (isEdgeEntering(convexPolygon, firstEdgeIndex, observerPoint))
    enteringLeftEdge = firstEdgeIndex;
    enteringRightEdge = firstEdgeIndex;
else
    leavingLeftEdge = firstEdgeIndex;
    leavingRightEdge = firstEdgeIndex;
end

% Now we need to search for the other two edges:
foundLeavingEdges = (leavingRightEdge >= 0) && (leavingLeftEdge >= 0);
foundEnteringEdges = (enteringRightEdge >= 0) && (enteringLeftEdge >= 0);

while (~foundLeavingEdges)
    edgeToTest = getMidEdgeOppositeClockwiseOrdering(enteringRightEdge, enteringLeftEdge, numberOfEdges);
    if ( (edgeToTest == enteringLeftEdge) || (edgeToTest == enteringRightEdge))
        error('Could not find a leaving edge! This should never happen!!');
    end

    if (isEdgeEntering(convexPolygon, edgeToTest, observerPoint))
        % Figure out if the edgeToTest should replace the leftEdge or the rightEdge:
        enteringLeftEdge = whichVertexIsToTheLeft(convexPolygon, edgeToTest, enteringLeftEdge, observerPoint); %% TODO
        enteringRightEdge = whichVertexIsToTheRight(convexPolygon, edgeToTest, enteringRightEdge, observerPoint); %% TODO
    else
        foundLeavingEdges = true;
        leavingRightEdge = edgeToTest;
        leavingLeftEdge = edgeToTest;
    end % if
end % while

while (~foundEnteringEdges)
    edgeToTest = getMidEdgeOppositeClockwiseOrdering(leavingLeftEdge, leavingRightEdge, numberOfEdges);

    if ( (edgeToTest == leavingLeftEdge) || (edgeToTest == leavingRightEdge))
        %         return null;
        error('Could not find an entering edge! Must be inside!!');
    end

    if (isEdgeEntering(convexPolygon, edgeToTest, observerPoint))
        foundEnteringEdges = true;
        enteringRightEdge = edgeToTest;
        enteringLeftEdge = edgeToTest;
    else
        % Figure out if the edgeToTest should replace the leftEdge or the rightEdge:
        newLeavingLeftEdge = whichVertexIsToTheLeft(convexPolygon, edgeToTest, leavingLeftEdge, observerPoint);
        newLeavingRightEdge = whichVertexIsToTheRight(convexPolygon, edgeToTest, leavingRightEdge, observerPoint);

        if ( (newLeavingLeftEdge == leavingLeftEdge) && (newLeavingRightEdge == leavingRightEdge))
            % Will loop forever if you don't do something about it!
            error('Looping forever!');
            % return null;
        end

        leavingLeftEdge = newLeavingLeftEdge;
        leavingRightEdge = newLeavingRightEdge;

        if (leavingLeftEdge == leavingRightEdge)
            %             return null;
            error('Start Point must have been inside the polygon!!');
        end
    end % if
end % while

% Now binary search till their are no gaps:
% left edge:
while (~areAdjacentInClockwiseOrder(enteringLeftEdge, leavingLeftEdge, numberOfEdges))% TODO
    edgeToTest = getMidEdgeOppositeClockwiseOrdering(leavingLeftEdge, enteringLeftEdge, numberOfEdges);
    if (isEdgeEntering(convexPolygon, edgeToTest, observerPoint))
        enteringLeftEdge = edgeToTest;
    else
        leavingLeftEdge = edgeToTest;
    end
end

% right edge
while (~areAdjacentInClockwiseOrder(leavingRightEdge, enteringRightEdge, numberOfEdges)) % TODO
    edgeToTest = getMidEdgeOppositeClockwiseOrdering(enteringRightEdge, leavingRightEdge, numberOfEdges);
    if (isEdgeEntering(convexPolygon, edgeToTest, observerPoint))
        enteringRightEdge = edgeToTest;
    else
        leavingRightEdge = edgeToTest;
    end
end

% Now the edges are adjacent. Want the common nodes:
ret = [leavingLeftEdge; enteringRightEdge];

end % getLineOfSightVerticesIndices

%%
function ret = isEdgeEntering(convexPolygon, edgeIndex, observerPoint)
nPoints = size(convexPolygon, 2);

vertex1 = convexPolygon(:, edgeIndex);
index2 = edgeIndex + 1;
if index2 > nPoints
    index2 = 1;
end
vertex2 = convexPolygon(:, index2);

normalY = vertex2(1) - vertex1(1);
normalX = -(vertex2(2) - vertex1(2));

pointToFirstVertexX = vertex1(1) - observerPoint(1);
pointToFirstVertexY = vertex1(2) - observerPoint(2);

dotProduct = normalX * pointToFirstVertexX + normalY * pointToFirstVertexY;

ret = (dotProduct < 0.0);

% if ret
%     plot([vertex1(1) vertex2(1)], [vertex1(2), vertex2(2)], 'g');
% else
%     plot([vertex1(1) vertex2(1)], [vertex1(2), vertex2(2)], 'r');
% end
end

%%
function ret = getMidEdgeOppositeClockwiseOrdering(leftEdgeIndex, rightEdgeIndex, numEdges)
if (rightEdgeIndex >= leftEdgeIndex)
    ret = floor(rightEdgeIndex + (leftEdgeIndex + numEdges - rightEdgeIndex + 1) / 2);
    if ret > numEdges
        ret = mod(ret, numEdges);
    end
else
    ret = floor((rightEdgeIndex + leftEdgeIndex + 1) / 2);
end

if ret == leftEdgeIndex || ret == rightEdgeIndex
    error('should never get here');
end

end

%%
function ret = areAdjacentInClockwiseOrder(index1, index2, numVertices)
indexToTest = index1 + 1;
if indexToTest > numVertices
    indexToTest = 1;
end
ret = (indexToTest == index2);
end

%%
function ret = whichVertexIsToTheLeft(convexPolygon, index1, index2, observerPoint)
      ret = whichVertexIsToTheLeftRight(convexPolygon, index1, index2, observerPoint, true);
  end

%%
function ret = whichVertexIsToTheRight(convexPolygon, index1, index2, observerPoint)
      ret = whichVertexIsToTheLeftRight(convexPolygon, index1, index2, observerPoint, false);
end

%%
function ret = whichVertexIsToTheLeftRight(convexPolygon, index1, index2, observerFramePoint2d, returnTheLeftOne)
      point1 = convexPolygon(:, index1);
      point2 = convexPolygon(:, index2);

      vectorToVertex1X = point1(1) - observerFramePoint2d(1);
      vectorToVertex1Y = point1(2) - observerFramePoint2d(2);

      vectorToVertex2X = point2(1) - observerFramePoint2d(1);
      vectorToVertex2Y = point2(2) - observerFramePoint2d(2);

      crossProduct = vectorToVertex1X * vectorToVertex2Y - vectorToVertex1Y * vectorToVertex2X;

      if (crossProduct < 0.0)
         if (returnTheLeftOne)
            ret = index1;
            return;
         else
            ret = index2;
            return;
         end
      else
         if (returnTheLeftOne)
            ret = index2;
            return;
         else
            ret = index1;
            return;
         end
      end
end
   