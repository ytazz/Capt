function oneStepCR = NStep_oneStepCaptureRegions(modelInfo, modelState, plotSettings, enableReactionMass, enableBoS)
%NSTEP_oneStepCaptureRegions
%   Returns the 1-step cature region for the different models
%   [ONESTEPCR] = NStep_oneStepCaptureRegions(MODELINFO, MODELSTATE,
%   PLOTSETTINGS, ENABLEREACTIONMASS, ENABLEBOS) returns the 1-step capture
%   region defined by the model variables MODELINFO, current model state 
%   MODELSTATE. PLOTSETTINGS describes the plotresolution and number of 
%   regions requested. Booleans ENABLEREACTIONMASS, 
%   ENABLEBOS set which model type should be selected for the 
%   calculations.ONESTEPCR [x1 x2 ..; y1 y2 ..] contains the point locations 
%   of border the 1-stepcapture region. NOt that this region should still 
%   be clipped to the maximum step length circlea and surrounded by an  
%   offset path at a distance determined by the maximum footlength.
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


%% CASE 1: Model LIPM PointFoot

if ~(enableReactionMass || enableBoS)
    rCPdTStep       = NStep_ricEvolve(modelState.rIC, modelState.rAnkle, modelState.dTWait);
    boundaryCase1   = case1_LIPMPointFoot(rCPdTStep,modelState,modelInfo);
    oneStepCR       = store(boundaryCase1);
    return
end

%% CASE 2: Model LIPM PointFoot + Reaction Mass

if ~enableBoS && enableReactionMass
    boundaryCase2   = case2_LIPMReactionMass(modelState,modelInfo,plotSettings);
    oneStepCR       = store(boundaryCase2);
    return
end

%% CASE 3: Model LIPM BoS 
if enableBoS && ~enableReactionMass
    boundaryCase3   = case3_LIPMBoS(modelState,modelInfo,plotSettings);
    oneStepCR       = store(boundaryCase3);
end


%% CASE 4: Model LIPM BoS + Reaction Mass
if enableBoS && enableReactionMass
windmilling = (modelState.t - modelState.tWmillStart) < 2 * modelInfo.dTWindmill;
if windmilling % case: currently Wmilling, rCP at dTStep can be calculated
    t0      = modelState.t;   % ON, pos
    t1      = max(modelState.tWmillStart + modelInfo.dTWindmill, modelState.t);  % ON, neg
    t2      = modelState.tWmillStart + 2*modelInfo.dTWindmill; % OFF
    t_step  = modelState.t + modelState.dTWait;
    if modelState.dT < modelInfo.dTmin
        % % case: rCPtfinal is a mapping of all possible rCoP locations
        rCPsNoWmill  = NStep_gridfun(@NStep_ricEvolve,{modelState.rIC},num2cell(modelState.BoS,1),{modelState.dTWait},'Squeeze','Transpose');
    else % case: rCPtfinal will reduce to a single point
        rCPsNoWmill  = modelState.rIC;
    end
    % Create boundary
    deltarC         = NStep_reactionMassOffset(modelState,modelInfo,'vector',t0,t_step,t1,t2);
    rCPtfinal       = rCPsNoWmill + repmat(deltarC,1,size(rCPsNoWmill,2)); %single or multiple points
    [in1 on1] = inpolygon(rCPtfinal(1,:),rCPtfinal(2,:),modelState.BoS(1, :),modelState.BoS(2,:));
    [in2 on2] = inpolygon(modelState.BoS(1,:),modelState.BoS(2,:),rCPtfinal(1,:),rCPtfinal(2,:)); % case rCPtfinal encloses the BoS
    % Could the current or other rCoP location bring the rCP within the BoS
    if max(in1) || max(on1) || max(in2) || max(on2)
        [boundaryX, boundaryY] = NStep_arcPoints(modelState.rAnkle,modelInfo.lMax,[-pi;pi], plotSettings.pointsPerRadian);
        boundaryCase4          = [boundaryX; boundaryY];
    else
        if modelState.dT < modelInfo.dTmin %case: a blunt type of wedge
            % Could also be created by lines of sight function: by translating the
            % rICNoWill by a factor proportional to rC and the distance
            % between the BoS and the rIC and rCPtfinal position
            BoSClock         = fliplr(modelState.BoS);
            rCPtfinalClock   = fliplr(rCPtfinal);
            WedgeRegion      = oneStepRegionBoSWmillWedge(BoSClock,rCPtfinalClock,modelInfo);
        else
            WedgeRegion    = oneStepRegionBoSWedgeSharp(rCPtfinal,modelState,modelInfo);     
        end
         boundaryCase4  = addOutliersWedge(WedgeRegion,modelState,modelInfo);
    end
elseif ~windmilling && ~modelState.haveUsedReactionMass; %case: wmill still possible, region is a wedge
    ZeroStep        = false;
    rcMagMax        = NStep_reactionMassOffset(modelState,modelInfo,'scalar',0,0); % Windmill offset
    haloBoSWmill    = NStep_makeHalo(modelState.BoS, rcMagMax, plotSettings.pointsPerRadian);
    if inpolygon(modelState.rIC(1),modelState.rIC(2),haloBoSWmill(1, :),haloBoSWmill(2,:));
        [boundaryX, boundaryY] = NStep_arcPoints(modelState.rAnkle,modelInfo.lMax,[-pi;pi], plotSettings.pointsPerRadian);
        boundaryCase4          = [boundaryX; boundaryY];
        ZeroStep               = true;
    end
    if ~ZeroStep
          %Expansion of region of case 3 due to windmilling
        if modelState.dT < modelInfo.dTmin 
            % case: region becomes a blunt wedge, with Wmill halo
            [boundaryCase3,rSight,rCPBoSReflect]    = oneStepRegionBoSWedgeBlunt(modelState.rIC,modelState,modelInfo);
        else % case: region becomes a sharp wedge, with Wmill halo
            [boundaryCase3,rSight,rCPBoSReflect]    = oneStepRegionBoSWedgeSharp(modelState.rIC,modelState,modelInfo);
        end

        % Make halo around rCP points (blunt wedge) or single point (sharp wedge)
        rcMagStep           = NStep_reactionMassOffset(modelState,modelInfo,'scalar',0,modelState.dTWait); % Windmill offset
        haloWmillatrCP      = NStep_makeHalo(fliplr(rCPBoSReflect), rcMagStep, plotSettings.pointsPerRadian);
        % Find line of sight between this halo and the BoS
        BoSClock         = fliplr(modelState.BoS);
        %haloWmillatrCP   = unique(haloWmillatrCP.','rows').';
        ConvIndex        = convhull(haloWmillatrCP(1,:),haloWmillatrCP(2,:));  % preferrably do not use convhull command to remove non unique points of makeHalo
        haloWmillatrCP   = haloWmillatrCP(:,ConvIndex(1:(end-1)));
        rCPtfinalClock   = fliplr(haloWmillatrCP);
        WedgeRegion      = oneStepRegionBoSWmillWedge(BoSClock,rCPtfinalClock,modelInfo);
        boundaryCase4    = addOutliersWedge(WedgeRegion,modelState,modelInfo);

    end
else % case: have already used Wmill, so continue with LIPM + BoS dynamics
    boundaryCase4       = case3_LIPMBoS(modelState,modelInfo,plotSettings);
end
oneStepCR  = store(boundaryCase4);
return
end


%% Store data
function oneStepCR = store(boundary)
oneStepCR.boundary.x    = boundary(1,:);
oneStepCR.boundary.y    = boundary(2,:);
end

%% End main function
end
%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                         "CASE" FUNCTIONS                            %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


%%  CASE 1
% point foot model
function boundaryCase1 = case1_LIPMPointFoot(rCPdTStep,modelState,modelInfo)

ZeroStep = norm(rCPdTStep - modelState.rAnkle) < 1e-10;
if ZeroStep
    boundaryCase1 = modelState.rAnkle;
else
    boundaryCase1 = oneStepRegionLine(rCPdTStep,modelState,modelInfo);
end
    
end
%% CASE 2
% reaction mass model with point foot
function boundaryCase2 = case2_LIPMReactionMass(modelState,modelInfo,plotSettings)
windmilling = (modelState.t - modelState.tWmillStart) < 2 * modelInfo.dTWindmill;
if windmilling % case: currently Wmilling, region is a line
    t0      = modelState.t;
    t1      = max(modelState.tWmillStart + modelInfo.dTWindmill, modelState.t);
    t2      = modelState.tWmillStart + 2*modelInfo.dTWindmill;
    t_step  = modelState.t + modelState.dTWait;
    deltarC = NStep_reactionMassOffset(modelState,modelInfo,'vector',t0,t_step,t1,t2);
    rCPnoWmill      = NStep_ricEvolve(modelState.rIC, modelState.rAnkle, modelState.dTWait);
    rCPtfinal       = rCPnoWmill + deltarC;
    % Cheat: if this "during Wmill CP" is very close to ankle, make ZeroStep
    % (other possibility: make line along Wmill angle)
    ZeroStep = norm(rCPtfinal - modelState.rAnkle) < 1e-10;
    if ZeroStep
        boundaryCase2   = modelState.rAnkle;
    else
        boundaryCase2   = oneStepRegionLine(rCPtfinal,modelState,modelInfo);
    end
elseif ~windmilling && ~modelState.haveUsedReactionMass; %case: wmill still possible, region is a wedge
    rcMagMax = NStep_reactionMassOffset(modelState,modelInfo,'scalar',0,0);
    ZeroStep = norm(modelState.rIC - modelState.rAnkle) < rcMagMax;
    if ZeroStep
        [boundaryX, boundaryY] = NStep_arcPoints(modelState.rAnkle,modelInfo.lMax,[-pi;pi], plotSettings.pointsPerRadian);
        boundaryCase2 = [boundaryX; boundaryY];
    else % case: not ZeroStep, calucalte OneStep
        rcMagdTStep     = NStep_reactionMassOffset(modelState,modelInfo,'scalar',0,modelState.dTWait);
        wedgeRegion     = oneStepRegionWmillWedge(rcMagdTStep,modelState,modelInfo,plotSettings);
        % Include points to make sure convex region is outside lMax
        boundaryCase2  = addOutliersWedge(wedgeRegion,modelState,modelInfo);
    end
else % case: have already used Wmill, so continue with LIPM + PointFoot dynamics
    rCPdTStepNoWmill    = NStep_ricEvolve(modelState.rIC, modelState.rAnkle, modelState.dTWait);
    boundaryCase2       = case1_LIPMPointFoot(rCPdTStepNoWmill,modelState,modelInfo);
end
end

%% CASE 3
% model with foot
function boundaryCase3   = case3_LIPMBoS(modelState,modelInfo,plotSettings)
ZeroStep       = false;
if norm(modelState.rIC-modelState.rAnkle) < modelInfo.maxBoSRadius % inexpensive rough check before expensive accurate check
    [in on] = inpolygon(modelState.rIC(1),modelState.rIC(2),modelState.BoS(1, :),modelState.BoS(2,:));
    if in || on
        [boundaryX, boundaryY] = NStep_arcPoints(modelState.rAnkle,modelInfo.lMax,[-pi;pi], plotSettings.pointsPerRadian);
        boundaryCase3          = [boundaryX; boundaryY];
        ZeroStep = true;
    end
end
if ~ZeroStep
    % Find the points of the BoS that are in line of sight of the rIC
    if modelState.dT < modelInfo.dTmin
        BoSWedge       = oneStepRegionBoSWedgeBlunt(modelState.rIC,modelState,modelInfo);
    else % the region reduces to a sharp wedge (triangle)
        BoSWedge       = oneStepRegionBoSWedgeSharp(modelState.rIC,modelState,modelInfo);
    end
    boundaryCase3  = addOutliersWedge(BoSWedge,modelState,modelInfo);
end
end

%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
%%%                         SUB-SUB FUNCTIONS                           %%%
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

%% pointsInSightBoS
% return BoS vertices in line of sight of observerpoint
function rSight = pointsInSightBoS(observerPoint,modelState)
BoSClockWise    = fliplr(modelState.BoS);
LineSightIndex  = NStep_getLineOfSightVerticesIndices(BoSClockWise,observerPoint); 
if LineSightIndex(2) > LineSightIndex(1) % 'sort' points
    rSightIndex  = [LineSightIndex(2):length(BoSClockWise), 1:LineSightIndex(1)];
else
    rSightIndex  = LineSightIndex(2):LineSightIndex(1);
end
rSight          = BoSClockWise(:,rSightIndex);

end


%%  BoSWedge Blunt
% create typical wedge shaped region with blunt tip due to the possibility
% of using the windmill and the foot
function [boundary,rSight,rCPBoSReflect] = oneStepRegionBoSWedgeBlunt(observerPoint,modelState,modelInfo)
rSight          = pointsInSightBoS(observerPoint,modelState);
% Map these points to corresponding rCP locations
rCPBoSReflect   = NStep_gridfun(@NStep_ricEvolve,{observerPoint},num2cell(rSight,1),{modelState.dTWait},'Squeeze','Transpose');
rOutsideCircle1 = rCPBoSReflect(:,1) + vecUnit(observerPoint,rCPBoSReflect(:,1))* 2*modelInfo.lMax;
rOutsideCircle2 = rCPBoSReflect(:,end) + vecUnit(observerPoint,rCPBoSReflect(:,end))* 2*modelInfo.lMax;
rPolygon        = [rOutsideCircle1,rCPBoSReflect,rOutsideCircle2];
boundary        = fliplr(rPolygon);

end
%% BoSWedge Sharp
% create typical wedge shaped region with sharp tip due to possibility of
% using only the windmill, no foot is present
function [boundary,rSight,rCPBoSReflect] = oneStepRegionBoSWedgeSharp(observerPoint,modelState,modelInfo)
rSight      = pointsInSightBoS(observerPoint,modelState);
vecSight1   = vecUnit(observerPoint,rSight(:,end));
vecSight2   = vecUnit(observerPoint,rSight(:,1));
rOutCircle1 = observerPoint - vecSight1 * 2 * modelInfo.lMax;
rOutCircle2 = observerPoint - vecSight2 * 2 * modelInfo.lMax;
rCPBoSReflect = observerPoint;
boundary    = [rOutCircle1, observerPoint,rOutCircle2];
end


%% oneStepRegionWmillWedge
% return the wedge shaped region that occurs for a point foot model with
% reaction mass
function boundary = oneStepRegionWmillWedge(rcMag,modelState,modelInfo,plotSettings)

rICAngle         = vecAngle(modelState.rAnkle,modelState.rIC);
rCPdTStep          = NStep_ricEvolve(modelState.rIC, modelState.rAnkle, modelState.dTWait);

% Calculate the location of the optimal windmill evolution at foreseen CircleCenter at StepDist, intersection
% point with rcMagDisk from ObsPoint
[TangPoints,relAngle] = circleTangPoints(modelState.rAnkle,rCPdTStep,rcMag,rICAngle);

% Create point along line from rAnkle to TangPoints that lie outside lMax circle
rOutside1  = modelState.rAnkle + vecUnit(modelState.rAnkle,TangPoints(:,1)) * (norm(TangPoints(:,1) -modelState.rAnkle) + modelInfo.lMax);
rOutside2  = modelState.rAnkle + vecUnit(modelState.rAnkle,TangPoints(:,2)) * (norm(TangPoints(:,2) -modelState.rAnkle) + modelInfo.lMax);

% Create nearArc circle
nearArcAngle            = (pi - relAngle -1/2*pi);
nearArcAngleSpan        = [(pi - nearArcAngle); (pi+nearArcAngle)] + rICAngle;
[nearArcX, nearArcY]    = NStep_arcPoints(rCPdTStep,rcMag, nearArcAngleSpan, plotSettings.pointsPerRadian);
boundary                = [rOutside2(1) nearArcX, rOutside1(1); rOutside2(2) nearArcY rOutside1(2)];

end


%% oneStepRegionBoSWmillWedge
% return the wedge shaped region that occurs for a model with foot and
% reaction mass
function boundary = oneStepRegionBoSWmillWedge(BoSClock,rCPtfinalClock,modelInfo)
rOutsideCircle   = cell(2,1);
rCPsortIndex     = cell(2,1);
run     = 0;
for line = 1:2
    indCPStart       = 1;
    doWhile          = true;
    while doWhile
        indCPtoBoS     = NStep_getLineOfSightVerticesIndices(BoSClock,rCPtfinalClock(:,indCPStart));
        indCPtoBoSDir  = indCPtoBoS(line);
        indBoStorCP    = NStep_getLineOfSightVerticesIndices(rCPtfinalClock,BoSClock(:,indCPtoBoSDir));
        indCPFinal     = indBoStorCP(line);
        run = run+1;
        if indCPStart ~= indCPFinal
            indCPStart = indCPFinal;
            doWhile = true;
            if run > 1e2
                error('Unexpected error in determining Lines of Sight in OneStep')
                doWhile = false; %#ok
            end
        else
            doWhile = false;
            % Store these points: create point outside lMax circle
            rOutsideCircle{line} = rCPtfinalClock(:,indCPFinal) + vecUnit(BoSClock(:,indCPtoBoSDir),rCPtfinalClock(:,indCPFinal))* 2*modelInfo.lMax;
            rCPsortIndex{line}   = indCPFinal;
        end
    end
end
% Only use convex points
if rCPsortIndex{2} > rCPsortIndex{1} % 'sort' points
    rCPfinalConvIndex  = [rCPsortIndex{2}:size(rCPtfinalClock,2), 1:rCPsortIndex{1}];
else
    rCPfinalConvIndex  = rCPsortIndex{2}:rCPsortIndex{1};
end
rCPfinalConv        = rCPtfinalClock(:,rCPfinalConvIndex);
boundary    = [rOutsideCircle{1},fliplr(rCPfinalConv),rOutsideCircle{2}];
end




%% addOutliersWedge
% add points to the wedge to make sure that points exist that lie outside the max step circle, so
% clipping is made possible
function wedgeRegionPlusOutliers = addOutliersWedge(wedgeRegion,modelState,modelInfo)

angTop      = vecAngle(modelState.rAnkle,wedgeRegion(:,1));
angBottom   = vecAngle(modelState.rAnkle,wedgeRegion(:,end));
angAverage  = (angTop + angBottom)/2;
angDiff     = diff([angTop;angBottom]);
if angDiff > pi && sign(angBottom) ~= sign(angTop)
    angAverage = pi + angAverage;
end
pointTop    = wedgeRegion(:,1) + [cos(angAverage);sin(angAverage)]*modelInfo.lMax;
pointBottom = wedgeRegion(:,end) + [cos(angAverage);sin(angAverage)]*modelInfo.lMax;
wedgeRegionPlusOutliers    = [pointTop wedgeRegion pointBottom];

end


%% oneStepRegionLine
% one step capture region line for a point foot model 
function boundary = oneStepRegionLine(rCPdTStep,modelState,modelInfo)
rCPdTStepDist   = norm(rCPdTStep- modelState.rCoP);
boundary        = [rCPdTStep, modelState.rCoP + vecUnit( modelState.rCoP,rCPdTStep)*(rCPdTStepDist + modelInfo.lMax)];
end

%% vecAngle
% Determine absolute angle with horziontal, of vector spanned by two points
function ret = vecAngle(Point1,Point2)
ret = atan2(Point2(2) - Point1(2), Point2(1) - Point1(1));
end

%% vecUnit
% Create vector of unity length
function ret = vecUnit(Point1,Point2)
ret = (Point2-Point1)/norm(Point2-Point1);
end


%% rotMat
% Roationmatrix
function R = rotMat(alfa)
R = [cos(alfa), -sin(alfa);  sin(alfa),  cos(alfa)];
end
%% dTSolve
%   Determines the step time for which the state is 1-Step Capturable
function dT = dTSolve(rIC,rAnkle,rICdes)
dTsol        = max(real(log((rAnkle - rICdes)/(rIC-rAnkle))));
dT           = max(dTsol);

end

%% circleTangPoints

function [TangPoints, AngleAtoC]  = circleTangPoints(observerPoint,circleCenter,circleRadius,relAngle)
   
%{              
                observerPoint
               /|
            C / | A
             /__| 
circleCenter  B  TangPoint
%}
% Determine tangential point (TangPoint) of line from observerpoint
% (observerPoint) to circle with center CircleCenter and radius B.

C                       = norm(circleCenter - observerPoint);
B                       = circleRadius;
A                       = sqrt(C^2-B^2);
AngleAtoC               = acos(A/C);
TangPoint1               = observerPoint + ([1,0]*rotMat(AngleAtoC  - relAngle)* C).';
TangPoint2               = observerPoint + ([1,0]*rotMat(-AngleAtoC - relAngle)* C).';
TangPoints               = [TangPoint1, TangPoint2];       
end
