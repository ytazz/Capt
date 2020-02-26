function NStep()
%NSTEP Main NStep function
%   Plots N-step capture regions for four model types:
%   - 3D-LIPM with point foot
%   - 3D-LIPM with finite-size foot
%   - 3D-LIPM with reaction mass and finite-size foot
%   - 3D-LIPM with reaction mass (without finite-size foot)
%   
%   This file is supplied as an addition to the draft paper:
%   "Capturability-Based Analysis and Control of Legged Locomotion, Part 2:
%   Application to Three Simple Gait Models" -
%
%   Details on the simulated models are to be found in the paper.
%   This function file contains the model settings and all settings related
%   to the graphical user interface.
%   This main function relies on the following other Matlab files:
%
%   - NStep_arcPoints.m: 
%       returns Cartesian coordinates of points on an arc with radius r at
%       a specified resolution
%   - NStep_BoSCreate.m:
%       returns Cartesian coordinates of points on a user defined convex
%       foot
%   - NStep_BoSPoints.mat:
%       contains local Cartesian coordinates of points on a user defined 
%       convex foot
%   - NStep_captureLimits.m:
%       returns the N-Step capture limits for a given model
%   - NStep_closestPointInConvexPolygon.m:
%       resturns the closest point inside a convex polygon for a point 
%       outside the polygon        
%   - NStep_equivalentConstantCMP.m:
%       computes the equivalent constant Centroidal Moment Pivot
%   - NStep_evolve.m:
%       simulates the dynamics of the 3D-LIPM         
%   - NStep_getCaptureRegions.m:
%       returns Cartesian coordinates to plot all capture regions
%   - NStep_getLineOfSightVerticesIndices.m:
%       returns vertices that are in line of sight from an observerpoint
%   - NStep_gridfun.m:
%       applies a function to each combination of input elements
%   - NStep_makeHalo.m:
%       returns multiple offset paths around a convex path 
%   - NStep_oneStepCaptureRegions.m:
%       returns the 1-step cature region for the different models
%   - NStep_polygonDiskOverlap.m:
%       clip a polygon to a circle
%   - NStep_r2Evolve.m:
%       calculates the second index of the state vector as presented in
%       paper eq. (10)
%   - NStep_reactionMassOffset.m:
%       Calculates the magnitude of rC as presented in paper eq. (37)
%   - NStep_ricEvolve.m:
%       returns the Instantaneous Capture Point position after time
%       interval dt
%   - NStep_simplePolygonDiskOverlap.m:
%       clip a polygon to a circle, assumes one intersection per polygon 
%       edge and either zero or two intersections in total
%       
%   For further information, contact:
%   Tomas de Boer, tomasdeboer@gmail.com, or    
%   Twan Koolen,   tkoolen@ihmc.us
%
%   Copyright 2010, Delft BioRobotics Laboratory
%   Delft University of Technology
%   $Revision: 1.0 $  $Date: February 2010 $
%%

home
close all hidden
addpath('./SupportFiles/')

%% VARIOUS SETTINGS:

%% SETTINGS: model
%   All settings that the define the model dynamics 
modelInfo.dTmin         = 1;        % Time in between steps (dimensionless)
modelInfo.lMax          = 1;        % Maximum length in between foot locations (dimensionless)
modelInfo.tauMax        = 0.5;      % Maximum reaction mass torque (dimensionless)
modelInfo.thetaMax      = 0.125;    % Maximum reaction mass angle;
modelInfo.dTWindmill    = sqrt(modelInfo.thetaMax / modelInfo.tauMax); % Curent windmill time interval
modelInfo.maxBoSRadius  = [];       % Max distance between foot (base of support) edge and ankle

if (2*modelInfo.dTWindmill) > modelInfo.dTmin
    disp(['Constraints not satisfied. Change tauMax to ', num2str(modelInfo.thetaMax/(modelInfo.dTmin/2)^2), ...
        ', or thetaMax to ', num2str(modelInfo.tauMax*(modelInfo.dTmin/2)^2)])
    return
end

%% SETTINGS: initialization of model states
%   initilaize variables so they are in scope in the main function, not just in the nested functions

modelState          = [];
rCPAtMinStep        = []; % Instantaneous CP at min step time
CPLineEnd           = []; % End of the line through the foot and the instcp
maxStepX            = []; % Max step length circle x
maxStepY            = []; % Max step length circle y
CoMTrajectory       = []; % CoM trajectory
footPositions       = []; % All previous and current foot positions
panLocations        = []; % Array with Qspline from current to next foot location, for panning
rAnkleNext          = []; % Next ankle location: will step to it once the minimum step time has passed
footAngleNext       = []; % Next foot angle: will step with foot in this orientation once the minimum step time has passed.


%% SETTINGS: figure
%   create the user interface window
CPfig = figure('DeleteFcn', @closeWindowCallBack, ...
    'WindowButtonDownFcn', @windowButtonDownCallBack,...
    'Name', 'N-Step Capturability',...
    'NumberTitle', 'off',...
    'Visible', 'off',...
    'KeyPressFcn',@keyHandler,...
    'Toolbar', 'figure',...
    'Interruptible', 'off');
CPaxes = axes('Parent',CPfig, 'FontSize',7,  'TickLength', [0.002 0 ]);

figureWidth = 1024; figureHeight = 800;
set(CPfig,'Position',[50 50 figureWidth figureHeight]);

% To enable zooming during animation
hManager = uigetmodemanager(CPfig);
%set(hManager.WindowListenerHandles,'Enable','off');

findfigs;

%% SETTINGS: plot/animation
%   animation plot settings such as animation speed, colors, etc..
plotSettings.NMax                   = 4;        % Maximum N for the N-Step Capture limit. Note that an N+1-Step Capture Region can be plotted using the N-Step Capture limit.
plotSettings.pointsPerRadian        = 10;       % Number of points per radian to plot an arc
plotSettings.animationTimeStep      = 1e-2;     % Dimensionless simulated time between frames
plotSettings.delay                  = 7;        % Delay factor of animation speed
plotSettings.marginFraction         = 1;        % Amount of white space around the max step length circle, fraction of step length
plotSettings.eraseMode              = 'normal';
plotSettings.captureRegionColors    = brighten(colormap(CPaxes, summer(plotSettings.NMax + 2)), 0.4);
plotSettings.axisSize               = modelInfo.lMax * (2.5 + plotSettings.marginFraction);



% Marker size
AnkleMarkerR                        = 0.015;
CoPMarkerR                          = 0.02;
CMPMarkerR                          = 0.025;
CoMMarkerR                          = 0.05;
ICMarkerR                           = 0.05;
CPMarkerR                           = 0.03;   

% Color stuff (also exist in CMYK color space for printing of the figure)
lightRed  =[0.96 0.3 0.3];
Red       =[0.93 0.11 0.14];
Blue      =[0 0.07 1];
Brown     =[0.78 0.6 .42];

%% SETTINGS: graphical user interface
%   all buttons settings

% Callback variables
closing                             = false;
stopped                             = true;
optimalWindmillAngle                = false;
optimalFootOrientation              = false;
optimalCoPPlacement                 = false;
autoCapture                         = false;
enableBoS                           = false;
enableReactionMass                  = false;
showStep                            = false;
nShowStep                           = [];
realStep                            = [];
panning                             = false;
nPan                                = [];

% Settings panel and children
settingsPanel = uipanel('Title','Model settings','FontSize',8,...
    'Units','pixels',...
    'BackgroundColor', get(CPfig, 'Color'));
autoCaptureCheckBox = uicontrol('Style','checkBox','String','Automatic capture',...
    'BackgroundColor', get(CPfig, 'Color'), ...
    'Callback',{@autoCaptureCallBack}, ...
    'KeyPressFcn',@keyHandler,...
    'Value', autoCapture);
optimalCoPPlacementCheckBox = uicontrol('Style','checkBox','String','Optimal CoP placement',...
    'Enable', 'off',...
    'BackgroundColor', get(CPfig, 'Color'), ...
    'Callback',{@optimalCoPPlacementCallBack}, ...
    'KeyPressFcn',@keyHandler,...
    'Value', optimalCoPPlacement); 
optimalFootOrientationCheckBox = uicontrol('Style','checkBox','String','Optimal foot orientation',...
    'Enable', 'off',...
    'BackgroundColor', get(CPfig, 'Color'), ...
    'Callback',{@optimalFootOrientationCallBack}, ...
    'KeyPressFcn',@keyHandler,...
    'Value', optimalFootOrientation); 
optimalWindmillAngleButton = uicontrol('Style','checkBox','String','Optimal lunge angle',...
    'Enable', 'off',...
    'BackgroundColor', get(CPfig, 'Color'), ...
    'Callback',{@optimalWindmillAngleCallBack}, ...
    'KeyPressFcn',@keyHandler,...
    'Value', optimalWindmillAngle);
windmillText = uicontrol('Style','text','String','Custom lunge angle: 0 - 2 pi',...
    'Enable','off',...
    'HorizontalAlignment', 'left', ...
    'KeyPressFcn',@keyHandler,...
    'BackgroundColor', get(CPfig, 'Color')); 
windmillAngleSlider = uicontrol('Style', 'slider',...
    'Enable', 'off',...
    'KeyPressFcn',@keyHandler,...
    'Min', 0, 'Max', 2 * pi, 'Value', 1/2*pi);
windmillButton = uicontrol('Style','pushbutton','String','Lunge',...
    'KeyPressFcn',@keyHandler,...
    'Callback',{@windmillButtonCallBack}, 'Enable', 'off');

% Model Panel objects and children
modelPanel = uipanel('Title','Model selection','FontSize',8,...
    'Units','pixels',...
    'BackgroundColor', get(CPfig, 'Color'));
BoSCheckBox = uicontrol('Style','checkBox','String','Enable foot',... 
    'BackgroundColor', get(CPfig, 'Color'), ...
    'Callback',{@BoSCheckBoxCallBack}, ...
    'KeyPressFcn',@keyHandler,...
    'Value', enableBoS);
ReactionMassCheckBox = uicontrol('Style','checkBox','String','Enable reaction mass',...
    'BackgroundColor', get(CPfig, 'Color'), ...
    'Callback',{@ReactionMassCheckBoxCallBack}, ...
    'KeyPressFcn',@keyHandler,...
    'Value', enableReactionMass);

% Simulation state
simulationPanel = uipanel('Title','Simulation state','FontSize',8,...
    'Units','pixels',...
    'BackgroundColor', get(CPfig, 'Color'));
timeText = uicontrol('Style','text','String','',...
    'BackgroundColor', get(CPfig, 'Color'));
stanceNrText = uicontrol('Style','text','String','',...
    'BackgroundColor', get(CPfig, 'Color'));
startButton = uicontrol('Style','pushbutton','String','Start',...
    'KeyPressFcn',@keyHandler,...
    'Callback',{@startButtonCallBack});
resetButton = uicontrol('Style','pushbutton','String','Reset',...
    'KeyPressFcn',@keyHandler,...
    'Callback',{@resetButtonCallBack});

leftMouseExplanationText = uicontrol('Style','text','String','Left click in the figure to step.    ',...
    'BackgroundColor', get(CPfig, 'Color'));
rightMouseExplanationText = uicontrol('Style','text','String','Right click in the foot to set CoP.',...
    'BackgroundColor', get(CPfig, 'Color'));


%% INITIALIZATION: plot window
%   draw an intial frame

resetState();
updateData();

% Capture Regions
for i = 1 : length(captureRegions)
    j = length(captureRegions) - i + 1; % reverse iterate
    N = captureRegions{j}.N;
    captureRegionPatches(i) = patch(0, 0,'k','EdgeColor','none'); %#ok<AGROW> %, plotSettings.captureRegionColors(j, :),'EdgeColor', 'none'); 
    captureRegionlegendStrings{i} = [num2str(N),'-step']; %#ok<AGROW>
end

% Create handles to everything in the figure
hold on; % do not remove, or handles can't be found again

for n = 1:50; % show upt to 50 footsteps
    BoSPatches(n) = patch(0,0,'-');  %#ok<AGROW> 
end
        
maxStepLengthCircle(1)  = plot(0, 0, '-', 'Color',[0 0 0],'EraseMode', plotSettings.eraseMode); % Max Step Length circle
maxStepLengthCircle(2) = plot(0, 0, '-', 'Color',[0 0 0],'EraseMode', plotSettings.eraseMode); % Max Step Length circle
CPLine              = plot(0, 0, ':', 'Color',Blue,'EraseMode', 'normal'); % CP line
CoMTrajectoryPlot   = plot(0, 0, ':', 'Color',[0 0 0],'LineWidth',0.5,'EraseMode', plotSettings.eraseMode); % Center of Mass trajectory
legLink             = plot(0, 0, '-', 'Color',[0 0 0],'LineWidth',2,'EraseMode', plotSettings.eraseMode); % Leg link
legLinkPiston       = plot(0, 0, '-', 'Color',[0 0 0],'LineWidth',3,'EraseMode', plotSettings.eraseMode,'Visible','on'); % Leg link
anklePositionsPlot  = plot(0, 0, 'o', 'Color',[0 0 0],'MarkerSize', 5, 'EraseMode', plotSettings.eraseMode); % rPointFoot position
ICMarker            = rectangle('EdgeColor',Blue,'FaceColor','none', 'Curvature',[1 1],'LineWidth',1,'EraseMode', plotSettings.eraseMode);
CPMarker            = rectangle('EdgeColor',Red,'FaceColor','none', 'Curvature',[1 1],'LineWidth',1,'EraseMode', plotSettings.eraseMode);
AnkleMarker         = rectangle('EdgeColor',[0 0 0],'FaceColor',Brown, 'Curvature',[1 1],'LineWidth',0.5,'EraseMode', plotSettings.eraseMode);
CoPMarker           = rectangle('EdgeColor',[0 0 0],'FaceColor','none', 'Curvature',[1 1],'LineWidth',0.5,'EraseMode', plotSettings.eraseMode);
CMPMarker           = rectangle('EdgeColor',[.1 .1 .1],'FaceColor','none', 'Curvature',[0 0],'LineWidth',0.5,'EraseMode', plotSettings.eraseMode);
CoMMarker           = rectangle('EdgeColor',[0 0 0],'FaceColor',[1 1 1], 'Curvature',[1 1],'LineWidth',2,'EraseMode', plotSettings.eraseMode);
stepLocator(1)      = rectangle('EdgeColor','none','FaceColor',Red, 'Curvature',[1 1],'LineWidth',1,'EraseMode', plotSettings.eraseMode,'Visible','off');
stepLocator(2)      = rectangle('EdgeColor',Red,'FaceColor','none', 'Curvature',[1 1],'LineWidth',2,'EraseMode', plotSettings.eraseMode,'Visible','off');
stepLocator(3)      = rectangle('EdgeColor',Red,'FaceColor','none', 'Curvature',[1 1],'LineWidth',2,'EraseMode', plotSettings.eraseMode,'Visible','off');
CoMFill             = patch(0,0,'k','Edgecolor','none');

% dummies: just to get the legend right
dummyICMarker       = plot(nan, nan, 'o', 'MarkerEdgeColor', Blue);
dummyCoPMarker      = plot(nan, nan, 'o', 'MarkerEdgeColor', [0 0 0], 'MarkerFaceColor', 'none');
dummyCMPMarker      = plot(nan, nan, 's', 'MarkerEdgeColor', [.1 .1 .1], 'MarkerFaceColor', 'none');

legendEntries = [captureRegionPatches, dummyICMarker, dummyCoPMarker, dummyCMPMarker];
legendStrings = [captureRegionlegendStrings, 'Inst. Capture Point', 'CoP', 'CMP'];

drawOnce();
setAxis();
legendHandle = legend(legendEntries, legendStrings);
hold off;

set(CPfig,'ResizeFcn', @resizeFcn)
set(CPfig, 'Visible', 'on');
uiwait();

%% ANIMATE
% update all variables at each animation step 
while (~closing)
    tic
    if ~panning
        updateData(plotSettings.animationTimeStep);
    end
    drawOnce();
    
    while toc < plotSettings.delay * plotSettings.animationTimeStep % speed control
        % waste time
    end
end

%% NESTED FUNCTIONS RELATED TO THE MODEL:

%% NESTED FUNCTION: (re)set the state of the model
    function resetState(varargin)
        % possible to externally overwrite modelState parameters (cautioun is advised)
        modelState.r                    = [-0.4; 0.4];    % CoM location projected onto the floor
        modelState.rd                   = [0.7; -0.3];    % CoM velocity
        modelState.rAnkle               = [0;0];          % ankle location.
        modelState.rCoP                 = [0;0];          % CoP location.
        modelState.rCMP                 = [0;0];          % CMP location.
        modelState.t                    = 0;              % absolute time
        modelState.dT                   = 0;              % time since last step
        modelState.tWmillStart          = NaN;            % time at which windmill was first applied
        modelState.BoS                  = [0;0];          % Base of Support vertices
        modelState.BoSAngle             = 0.11;           % angle of nominal BoS w.r.t. x-axis
        modelState.stanceNr             = 1;              % number of stance phases
        modelState.haveUsedReactionMass = false;          % whether or not the reaction mass has been used
        modelState.windmillAngle        = NaN;            % windmill direction
        modelState.tauWindmill          = 0;              % reaction mass torque
        % possible to extenally overwrite modelState parameters (cautioun is advised)
        if ~(isempty(varargin))
            for index = 1:size(varargin,2)
                modelState.(inputname(index)) = varargin{index};
            end
        end
        modelState.rIC                  = modelState.r + modelState.rd;  % instantaneous CP of current state, i.e. at dT since last step
        modelState.r2                   = modelState.r - modelState.rd;   % 'State 2'
        modelState.dTWait               = max(0, modelInfo.dTmin - modelState.dT); % Time remaining until a step is allowed
 
        % plot stuff
        CoMTrajectory           = [];
        footPositions           = modelState.rAnkle;
        showStep                = false;
        nShowStep               = [];
        realStep                = [];
        panning                 = false;
        nPan                    = [];
        
        %
        if enableBoS
            modelState.BoS = NStep_BoSCreate(modelState.rAnkle, 'Rotate', modelState.BoSAngle);
        end
        
        rAnkleNext = [];
    end

%% NESTED FUNCTION: update all states/variables
%   based on the selected model and the interface settings, calculate the
%   next model state
    function updateData(timeStep)
        if nargin == 0
            timeStep = 0;
        end
    
        %% 1 decision making based on current time/state
        epsilon = 1e-10;
        if norm(modelState.rIC - modelState.rAnkle) < epsilon
            modelState.rIC = modelState.rAnkle;
        end
        
        % step if there is an rAnkleNext and we're allowed to step
        allowedToStep = modelState.dTWait == 0;
        if allowedToStep && ~isempty(rAnkleNext)
            if ~isempty(footAngleNext)
                stepTo(rAnkleNext, false, footAngleNext);
            else
                stepTo(rAnkleNext, false);
            end
            rAnkleNext = []; footAngleNext = [];
        end
        
        % check if we're going to set CoP optimally
        if optimalCoPPlacement
            doOptimalCoPPlacement();
        end
        
        % check if we're going to auto-capture
        if autoCapture
            doAutoCaptureStrategy();
        end
        
        % handle CMP stuff
        if enableReactionMass
            modelState.rCMP = modelState.rCoP + reactionMassCMPDisplacement();
        else
            modelState.rCMP = modelState.rCoP;
        end

        %% 2 compute new modelState

        % Incrementally increase time
        if timeStep ~= 0
            modelState.dT   = round(modelState.dT/timeStep)*timeStep + timeStep; %fix rounding errors
            modelState.t    = round(modelState.t/timeStep)*timeStep + timeStep;
        end
        modelState.dTWait = max(0, modelInfo.dTmin - modelState.dT);
        
        % Compute new state
        [modelState.rIC, modelState.r2, modelState.r, modelState.rd] = NStep_evolve(modelState.rIC, modelState.r2, modelState.rCMP, timeStep);
        CoMTrajectory = [CoMTrajectory modelState.r];
        
        %% 3 compute capture points/regions
        % Compute Capture regions
        captureRegions = NStep_getCaptureRegions(modelInfo, modelState, plotSettings, enableReactionMass, enableBoS);
        
        % Compute Capture point
        windmilling = (modelState.t - modelState.tWmillStart) < 2 * modelInfo.dTWindmill;
        if windmilling && enableReactionMass
            t0      = modelState.t;   % ON, pos
            t1      = max(modelState.tWmillStart + modelInfo.dTWindmill, modelState.t);  % ON, neg
            t2      = modelState.tWmillStart + 2*modelInfo.dTWindmill; % OFF
            t_step  = modelState.t + modelState.dTWait;
            rCPNoWmill      = NStep_ricEvolve(modelState.rIC, modelState.rCoP, modelState.dTWait);
            deltarC         = NStep_reactionMassOffset(modelState,modelInfo,'vector',t0,t_step,t1,t2);
            rCPAtMinStep    = rCPNoWmill + deltarC;
        else % Compute capture point at first possible step time, no Wmilling 
            rCPAtMinStep    = NStep_evolve(modelState.rIC, modelState.r2, modelState.rCoP, modelState.dTWait);
        end
    end

%% NESTED FUNCTION: handle the change of the foot location
%   change the modelstate based on a stepping action
    function stepTo(newAnkleLocation, optimal, footAngle)
        epsilon = 1e-10;
        withinReach = (norm(newAnkleLocation - modelState.rAnkle) - modelInfo.lMax <= epsilon);
        minStepTimeHasPassed = (modelState.dTWait == 0);

        if (withinReach && minStepTimeHasPassed)
            modelState.rAnkle = newAnkleLocation;
            modelState.stanceNr = modelState.stanceNr + 1;
            footPositions = [footPositions, modelState.rAnkle];
            
            if enableBoS
                epsilon = 1e-10;
                ICPonCoP = norm(modelState.rIC - modelState.rAnkle) < epsilon;
                if (nargin < 3) || ICPonCoP
                    footAngle = 0;
                end
                if (optimalFootOrientation || optimal) && ~ICPonCoP
                    footAngle = atan2(modelState.rIC(2) - modelState.rAnkle(2), modelState.rIC(1) - modelState.rAnkle(1)); % @todo: discuss: is this what we want?
                end
                [modelState.BoS, modelInfo.maxBoSRadius] = NStep_BoSCreate(modelState.rAnkle, 'Rotate', footAngle); % rBoS points locations
                updateFootprints();

                modelState.rCoP = modelState.rAnkle;
            else
                modelState.BoS = modelState.rAnkle;                
                modelState.rCoP = modelState.rAnkle;
            end
            
            % Update modelState
            modelState.dT           = 0;
            modelState.dTWait       = modelInfo.dTmin;
            
            % Update foot positions
            footPositions = [footPositions, modelState.rAnkle];
            
            % gui step location showing stuff
            showStep = true;
            nShowStep = 1;
            panning = true;
            nPan = 1;
            realStep = true;
        elseif withinReach
            rAnkleNext = newAnkleLocation;
            if nargin == 3
                footAngleNext = footAngle;
            else
                footAngleNext = [];
            end
            
            showStep = true;
            nShowStep = 1;
            realStep = false;
        else
            beep
        end
    end

%% NESTED FUNCTION: handle desired change in CoP location
% change the CoP location as a result of a step or CoP manipulation within
% the foot
    function setCoP(newCoPLocation)
        % check if new CoP location is inside BoS
        if enableBoS
            insideBoS = inpolygon(newCoPLocation(1), newCoPLocation(2), modelState.BoS(1, :), modelState.BoS(2, :));
        else
            epsilon = 1e-10;
            insideBoS = norm(newCoPLocation - modelState.rAnkle) < epsilon;
        end
        
        % set new CoP if inside, error otherwise
        if insideBoS
            modelState.rCoP         = newCoPLocation;
        else
            error('New CoP location not inside BoS');
        end
    end

%% NESTED FUNCTION: do optimal CoP placement
%   place the CoP at an optimal location in the foot: closest to the
%   Instantaneous Capture Point
    function CoPInsideBoS = doOptimalCoPPlacement(desiredLocation)
        if nargin == 0
            desiredLocation = modelState.rIC;
        end

        try
            % try to put the CoP onto the ICP
            setCoP(desiredLocation);
            CoPInsideBoS = true;
        catch

            % optimal CoP location is the point in the BoS that's closest
            % to the ICP
            optimalCoPLocation = NStep_closestPointInConvexPolygon(desiredLocation, modelState.BoS);

            % move it in just a little
            epsilon = 1e-10;
            optimalCoPLocation = modelState.rAnkle + (optimalCoPLocation - modelState.rAnkle) * (1 - epsilon);
            setCoP(optimalCoPLocation);
            CoPInsideBoS = false;
        end
    end

%% NESTED FUNCTION: sets the varariables for using the reaction mass
%   based on the modelsettings, interface settings determine the 
%   parameters of the reaction mass

    function windmilling = windmill(angle, tau)
        if enableReactionMass && ~modelState.haveUsedReactionMass
            modelState.tWmillStart = modelState.t;
            if optimalWindmillAngle
                angle = atan2(modelState.rIC(2) - modelState.rCoP(2), modelState.rIC(1) - modelState.rCoP(1));
            end
            modelState.windmillAngle = angle;
            if tau <= modelInfo.tauMax
                modelState.tauWindmill = tau;
            else
                error('tau is too large');
            end
            modelState.haveUsedReactionMass = true;
            windmilling = true;
            
            set(windmillButton, 'Enable', 'off');
            set(optimalWindmillAngleButton, 'Enable', 'off');
            set(windmillAngleSlider, 'Enable', 'off');
            set(windmillText, 'Enable', 'off');
            set(ReactionMassCheckBox,'Enable','off');
        else
            windmilling = false;
        end
    end

%% NESTED FUNCTION: auto capture strategy
%   determine the optimal automated stepping strategy that results
%   in a captured state as soon as possible
    function doAutoCaptureStrategy()
        
        % ankle strategy (CoP modulation)
        windmilling = modelState.t - modelState.tWmillStart < 2 * modelInfo.dTWindmill;
        if windmilling
            % take use of reaction mass into account in determining where
            % to put CoP
            t0 = modelState.t;   % ON, pos
            t1 = max(modelState.tWmillStart + modelInfo.dTWindmill, modelState.t);  % ON, neg
            t2 = modelState.tWmillStart + 2*modelInfo.dTWindmill; % OFF
            t_step = modelState.t + modelState.dTWait;            
            deltarC = NStep_reactionMassOffset(modelState, modelInfo, 'vector', t0, t_step, t1, t2);
            rC = modelState.rIC + deltarC * exp(-(t_step - t0));
            zeroStepCapturable = doOptimalCoPPlacement(rC);
        else
            % just put the CoP as close as possible to the ICP
            zeroStepCapturable = doOptimalCoPPlacement(modelState.rIC);
        end

        
        % hip strategy (CMP modulation)
        if ~modelState.haveUsedReactionMass
            % compute CoP-ICP distance and CoP-ICP line angle
            dRIC = modelState.rIC - modelState.rCoP;
            rICDistance = norm(dRIC);
            rICangle = atan2(dRIC(2), dRIC(1));
            
            % flywheel torque needed to become zero step
            % capturable:
            tau = rICDistance / (exp(-2 * modelInfo.dTWindmill) - 2 * exp(-modelInfo.dTWindmill) + 1);

            if tau <= modelInfo.tauMax
                zeroStepCapturable = true;
            end
            
            % go.
            windmilling = windmill(rICangle, min(tau, modelInfo.tauMax));
        end
        

        % stepping strategy
        if (modelState.dTWait == 0) && (~zeroStepCapturable)
            if (windmilling)
                % determine where to step for first step (does not assume that windmilling is done before stepping!)
                t0 = modelState.t;   % ON, pos
                t1 = max(modelState.tWmillStart + modelInfo.dTWindmill, modelState.t);  % ON, neg
                t2 = modelState.tWmillStart + 2*modelInfo.dTWindmill; % OFF
                t_step = modelState.t + modelState.dTWait;
                rCNoHipTorque = NStep_ricEvolve(modelState.rIC, modelState.rCoP, modelState.dTWait);
                deltarC = NStep_reactionMassOffset(modelState,modelInfo,'vector',t0,t_step,t1,t2);
                rC = rCNoHipTorque + deltarC;
                if norm(rC - modelState.rAnkle) <= modelInfo.lMax;
                    stepTo(rC, true);
                else
                    stepTo((rC - modelState.rAnkle) * modelInfo.lMax / norm(rC - modelState.rAnkle) + modelState.rAnkle, true);
                end
            else
                % do usual strategy without flywheel.
                ICPWithinRange = norm(modelState.rIC - modelState.rAnkle) <= modelInfo.lMax;
                if ICPWithinRange
                    stepTo(modelState.rIC, true);
                else
                    % step in the right direction as far as you can
                    stepTo(modelState.rAnkle + modelInfo.lMax * (modelState.rIC - modelState.rAnkle) / norm(modelState.rIC - modelState.rAnkle), true);
                end
                doOptimalCoPPlacement();
            end
        end

    end %function

%% NESTED FUNCTION: return CMP displacement due to reaction mass
%   as a result of lunging, determine the resulting CMP location 
    function ret = reactionMassCMPDisplacement()
        windmilling = (modelState.haveUsedReactionMass && (modelState.t - modelState.tWmillStart < 2 * modelInfo.dTWindmill));
        if windmilling
            deltaCMP = modelState.tauWindmill * [cos(modelState.windmillAngle); sin(modelState.windmillAngle)];

            if (modelState.t - modelState.tWmillStart) < modelInfo.dTWindmill
                dt1 = min(plotSettings.animationTimeStep, modelInfo.dTWindmill - (modelState.t - modelState.tWmillStart));
                dt2 = plotSettings.animationTimeStep - dt1;
                if dt2 > 0
                    ret = NStep_equivalentConstantCMP(+deltaCMP, -deltaCMP, dt1, dt2);
                else
                    ret = +deltaCMP;
                end
            else
                dt1 = min(plotSettings.animationTimeStep, 2 * modelInfo.dTWindmill - (modelState.t - modelState.tWmillStart));
                dt2 = plotSettings.animationTimeStep - dt1;
                if dt2 > 0
                    ret = NStep_equivalentConstantCMP(-deltaCMP, 0, dt1, dt2);
                else
                    ret = -deltaCMP;
                end
            end
        else
            ret = [0; 0];
        end
    end

%% NESTED FUNCTIONS RELATED TO GRAPHICAL USER INTERFACE: 


%% NESTED FUNCTION: draw a new plot based on calculated data
    function drawOnce()
        % Capture Regions
        for k = 1 : length(captureRegions)
            j = length(captureRegions) - k + 1; % reverse iterate
            set(captureRegionPatches(k),...
                'XData',captureRegions{j}.boundary.x,...
                'YData',captureRegions{j}.boundary.y,...,
                'FaceColor',plotSettings.captureRegionColors(j, :),...
                'EdgeColor',plotSettings.captureRegionColors(j, :));
            set(captureRegionPatches(k), 'Visible', 'on')
        end
        for k = length(captureRegions) + 1 : length(captureRegionPatches)
            set(captureRegionPatches(k), 'Visible', 'off')
        end

        % ___________________________________________
        % Model specific plot settings
        if enableBoS;
            BoSVisible = 'on';
        else
            BoSVisible = 'off';
        end

        if enableReactionMass;
            CMPVisible  = 'on';
            CoMColor = 'modelState.r'; % +++tk: ?
            if enableReactionMass;
                CoMColor = lightRed;
            end
        else
            CMPVisible  = 'off';
            CoMColor    = [1 1 1];
        end

        %____________________________________________
        % State specific plot settings
        if modelState.dTWait == 0
            CPMarkerColor = 'r';
        else
            CPMarkerColor = 'none';
        end
        
        if  modelState.rCMP == modelState.rCoP
            CPLineVisible = 'on';
        else
            CPLineVisible = 'off';
        end
        
        CPLineEnd = rCPAtMinStep;

        % Set position data for all the plot elements
        set(CPLine,   'XData', [modelState.rCMP(1), CPLineEnd(1)], 'YData', [modelState.rCMP(2), CPLineEnd(2)],'Visible',CPLineVisible); % Capture Point line
        set(legLink, 'XData', [modelState.r(1); modelState.rAnkle(1)], 'YData', [modelState.r(2); modelState.rAnkle(2)]); % Leg link
        legPiston = modelState.r - 0.5*norm( modelState.r - modelState.rAnkle) * (modelState.r - modelState.rAnkle) / norm(modelState.r - modelState.rAnkle);
        set(legLinkPiston, 'XData', [modelState.r(1); legPiston(1)], 'YData', [modelState.r(2); legPiston(2)]); % Leg link
        set(anklePositionsPlot, 'XData', footPositions(1, :), 'YData', footPositions(2, :)); % Foot position
        set(CoMTrajectoryPlot, 'XData', CoMTrajectory(1, :), 'YData', CoMTrajectory(2, :)); % CoM Trajectory
        set(ICMarker, 'Position', [modelState.rIC(1)-ICMarkerR, modelState.rIC(2)-ICMarkerR,2*ICMarkerR, 2*ICMarkerR]); % Instantaneous Capture Point
        set(CPMarker, 'Position', [rCPAtMinStep(1)-CPMarkerR, rCPAtMinStep(2)-CPMarkerR,2*CPMarkerR, 2*CPMarkerR], 'FaceColor',CPMarkerColor); % A Capture Point
        set(CoPMarker, 'Position', [modelState.rCoP(1)-CoPMarkerR, modelState.rCoP(2)-CoPMarkerR,2*CoPMarkerR, 2*CoPMarkerR],'Visible',BoSVisible);
        set(CMPMarker, 'Position', [modelState.rCMP(1)-CMPMarkerR, modelState.rCMP(2)-CMPMarkerR,2*CMPMarkerR, 2*CMPMarkerR],'Visible',CMPVisible);
        set(AnkleMarker,'Position',[modelState.rAnkle(1)-AnkleMarkerR, modelState.rAnkle(2)-AnkleMarkerR,2*AnkleMarkerR, 2*AnkleMarkerR]); % Center of Mass position
        set(CoMMarker,'Position', [modelState.r(1)-CoMMarkerR, modelState.r(2)-CoMMarkerR,2*CoMMarkerR, 2*CoMMarkerR],'FaceColor', CoMColor); % Center of Mass position
        
        set(timeText, 'String', sprintf('Time = %0.2f', modelState.t)); % time
        set(stanceNrText, 'String', sprintf('Stance # %d ', modelState.stanceNr)); % time
        
        [CoMFillX1, CoMFillY1]  = NStep_arcPoints(modelState.r, CoMMarkerR, [0, 1/2*pi], plotSettings.pointsPerRadian);
        [CoMFillX2, CoMFillY2]  = NStep_arcPoints(modelState.r, CoMMarkerR, [-pi, -1/2*pi], plotSettings.pointsPerRadian);
        set(CoMFill,'XData', [CoMFillX1, fliplr(CoMFillX2)], 'YData', [CoMFillY1, fliplr(CoMFillY2)]);
     
        % show step location
        showStepLocation();
        
        % panning
        dynamicPanning();
        
        % max step length circle
        drawMaxStepCircles();
        
        drawnow;
        % This is and *should be* the only event that causes event queue
        % processing
        % Other such events:
        % * Returning to the MATLAB prompt
        % * executing figure, getframe, input, keyboard, pause
        % * Functions that wait for user input (i.e., waitforbuttonpress,
        %   waitfor, ginput)
        % Touching figure handles is not allowed from this point on (since
        % they might not exist, because the figure could have been deleted),
        % until we check ~closing.
    end


%% NESTED FUNCTION: remove all footprints
    function removeFootprints()
        for k = 1 : length(BoSPatches)
            set(BoSPatches(k), 'Visible', 'off')
        end
        %BoSPatches = [];
    end

%% NESTED FUNCTION: add a new footprint
    function updateFootprints()
        set(BoSPatches(modelState.stanceNr),...
            'XData',modelState.BoS(1,:),...
            'YData',modelState.BoS(2,:),...
            'LineWidth',1,'EdgeColor',[0 0 0],'FaceColor',lightRed,...
            'EraseMode', 'normal', 'Visible', 'on');
    end

%% NESTED FUNCTION: highlight the new step location
    function showStepLocation()
        % show red dot
        if ~isempty(rAnkleNext)
            circleRadius = 0.015 * 0.5;

            x = rAnkleNext(1) - circleRadius;
            y = rAnkleNext(2) - circleRadius;
            w = 2 * circleRadius;
            h = w;

            set(stepLocator(1),'Position', [x, y, w, h],...
                'Visible', 'on');
        end
        
        % flash step locators
        if showStep            
            if realStep
                stepLocation = modelState.rAnkle;
            else
                stepLocation = rAnkleNext;
            end

            % time setting
            dtFlash = 0.25;
            nFlash = ceil(dtFlash / plotSettings.animationTimeStep);

            if nShowStep <= nFlash
                flashStep = ceil(6 * nShowStep / nFlash);
                if flashStep < 4
                    circleRadius = 0.015 * 0.5* flashStep^2;
                    x = stepLocation(1) - circleRadius;
                    y = stepLocation(2) - circleRadius;
                    w = 2 * circleRadius;
                    h = w;
                    set(stepLocator(flashStep),'Position', [x, y, w, h],...
                        'Visible', 'on');
                else
                    set(stepLocator(flashStep - 3), 'Visible', 'off');
                end
            end

            if nShowStep > nFlash
                % done 
                showStep = false;
                nShowStep = [];
                realStep = [];
            end

            nShowStep = nShowStep + 1;
        end
    end

%% NESTED FUNCTION: set figure axis
    function setAxis(center)
        if nargin == 0
            center = modelState.rAnkle;
        end
        axisSize = plotSettings.axisSize;
        x_lim = [center(1) - 1/2 * axisSize, center(1) + 1/2 * axisSize];
        y_lim = [center(2) - 1/2 * axisSize, center(2) + 1/2 * axisSize];
        
        set(CPaxes, 'Xlim', x_lim,'Ylim', y_lim);
    end


%% NESTED FUNCTION: dynamically pan to a new ankle location
    function dynamicPanning()
        if panning
            % first time around: compute panning locations
            if isempty(panLocations)
                res = linspace(0, 1, 30);
                
                xCurrent = mean(get(CPaxes, 'Xlim'));
                yCurrent = mean(get(CPaxes, 'Ylim'));
                
                xSpan = [xCurrent, modelState.rAnkle(1)] ;
                ySpan = [yCurrent, modelState.rAnkle(2)] ;
                
                splineX = spline([0, 1],[0, xSpan, 0]);
                splineY = spline([0, 1],[0, ySpan, 0]);
                panLocations  = [ppval(splineX, res); ppval(splineY, res)];
            end

            % Set axis
            panLocation = panLocations(:, nPan);
            setAxis(panLocation);

            % Bookkeeping
            nPan = nPan + 1;
            if nPan > size(panLocations, 2)
                % done
                panning = false;
                panLocations = [];
                nPan = [];
            end
        end
    end

%% NESTED FUNCTION: draw max step length circles
    function drawMaxStepCircles()
        [maxStepX, maxStepY] = NStep_arcPoints(modelState.rAnkle, modelInfo.lMax, [-pi, pi], plotSettings.pointsPerRadian);
        maxStepX(end + 1) = maxStepX(1);
        maxStepY(end + 1) = maxStepY(1);
        
        if panning
            % Fade step length circles
            numPanLoc = size(panLocations,2);          
            
            newMaxStepColors = colormap(CPaxes, gray(numPanLoc));
            oldMaxStepColors = flipud(newMaxStepColors);
            
            newMaxStepColor = newMaxStepColors(nPan, :);
            oldMaxStepColor = oldMaxStepColors(nPan, :);
            
            set(maxStepLengthCircle(1),'Color', newMaxStepColor);
            set(maxStepLengthCircle(2),'XData', maxStepX, 'YData', maxStepY, 'Color', oldMaxStepColor, 'Visible', 'on');
        else
            % Just show it
            set(maxStepLengthCircle(1),'XData', maxStepX, 'YData', maxStepY);
            set(maxStepLengthCircle(1),'Color','k');
            set(maxStepLengthCircle(2),'Visible','off');
        end
    end

%% NESTED FUNCTION: adjust object positions on window resize
  function resizeFcn(varargin)
    %   
    figDim = get(CPfig, 'Position');
    
    % Axes
    set(CPaxes, 'DataAspectRatio', [1 1 1]);
    set(CPaxes, 'Position', [0.05, 0.1, 0.7, 0.8]);
    
    % Legend
    legendcolorbarlayout(CPaxes, 'on');
    set(legendHandle, 'Position', [0.815, 0.75, 150 / figDim(3), 0.13]); % this is in normalized coordinates
    legendcolorbarlayout(CPaxes, 'remove'); % saves a whole lot of CPU time

    % Panels 
    set(modelPanel,'Position',                      [figDim(3) * 0.81, figDim(4) * 0.56, 165, figDim(4)*0.11]);
    set(settingsPanel,'Position',                   [figDim(3) * 0.81, figDim(4) * 0.284, 165,figDim(4)*0.26]);
    set(simulationPanel,'Position',                 [figDim(3) * 0.81, figDim(4) * 0.11, 165,figDim(4)*0.17]);  
    % Panel 1
    set(BoSCheckBox, 'Position',                    [figDim(3) * 0.815, figDim(4) * 0.604, 150, 25]);
    set(ReactionMassCheckBox, 'Position',           [figDim(3) * 0.815, figDim(4) * 0.573, 150, 25]);
    % Panel 2 
    set(autoCaptureCheckBox, 'Position',            [figDim(3) * 0.815, figDim(4) * 0.485, 150, 25]);
    set(optimalFootOrientationCheckBox, 'Position', [figDim(3) * 0.815, figDim(4) * 0.454, 150, 25]); 
    set(optimalCoPPlacementCheckBox, 'Position',    [figDim(3) * 0.815, figDim(4) * 0.423, 150, 25]);
    set(optimalWindmillAngleButton, 'Position',     [figDim(3) * 0.815, figDim(4) * 0.392, 150, 25]);
    set(windmillText,'Position',                    [figDim(3) * 0.815 + 5, figDim(4) * 0.351, 150, 25]);
    set(windmillAngleSlider,'Position',             [figDim(3) * 0.815, figDim(4) * 0.330, 150, 25]);
    set(windmillButton,'Position',                  [figDim(3) * 0.815+15, figDim(4) * 0.289, 115, 25]);
    % Panel 3
    set(timeText, 'Position',                       [figDim(3) * 0.815, figDim(4) * 0.21, 70, 25]);
    set(stanceNrText, 'Position',                   [figDim(3) * 0.815, figDim(4) * 0.19, 70, 25]);
    set(startButton,'Position',                     [figDim(3) * 0.815, figDim(4) * 0.16, 150, 25]);
    set(resetButton,'Position',                     [figDim(3) * 0.815, figDim(4) * 0.12, 150, 25]);
    
    set(leftMouseExplanationText, 'Position',       [figDim(3) * 0.815, figDim(4) * 0.05, 160, 25]);
    set(rightMouseExplanationText, 'Position',      [figDim(3) * 0.815, figDim(4) * 0.02, 160, 25])
    
  end

%% NESTED FUNCTION: callback methods for the user interface buttons
    function closeWindowCallBack(source, eventdata) %#ok<INUSD>
        closing = true;
        rmpath('./SupportFiles/')
    end

    function windowButtonDownCallBack(source, eventdata) %#ok<INUSD>
        currentPoint = get(CPaxes, 'CurrentPoint');
        currentPointVec = currentPoint(2, 1 : 2)';
        
        xLim = get(CPaxes, 'XLim');
        yLim = get(CPaxes, 'YLim');
        currentPointInAxes = currentPointVec(1) >= xLim(1) && ...
                             currentPointVec(1) <= xLim(2) && ...
                             currentPointVec(2) >= yLim(1) && ...
                             currentPointVec(2) <= yLim(2);

        if currentPointInAxes
            altClick = strcmp(get(source, 'SelectionType'), 'alt');
            normalClick = strcmp(get(source, 'SelectionType'), 'normal');
            doubleClick = strcmp(get(source, 'SelectionType'), 'open');
            if altClick
                try
                    setCoP(currentPointVec);
                catch
                    beep
                end
            elseif normalClick || doubleClick
                stepTo(currentPointVec, false);
            end
        end
    end


    function optimalCoPPlacementCallBack(source, eventdata) %#ok<INUSD>
        optimalCoPPlacement = ~optimalCoPPlacement;
    end


    function optimalFootOrientationCallBack(source, eventdata) %#ok<INUSD>
        optimalFootOrientation = ~optimalFootOrientation;
    end


    function optimalWindmillAngleCallBack(source, eventdata) %#ok<INUSD>
        optimalWindmillAngle = ~optimalWindmillAngle;
        if optimalWindmillAngle
            set(windmillAngleSlider, 'Enable', 'off');
            set(windmillText, 'Enable', 'off');
        else
            set(windmillAngleSlider, 'Enable', 'on');
            set(windmillText, 'Enable', 'on');
        end
    end

    function autoCaptureCallBack(source, eventdata) %#ok<INUSD>
        autoCapture = ~autoCapture;
    end

    function BoSCheckBoxCallBack(source, eventdata) %#ok<INUSD>
        enableBoS = ~enableBoS;
        if ~enableBoS
            set(BoSPatches(modelState.stanceNr), 'Visible', 'off');
            modelState.BoS = modelState.rAnkle;
            modelState.rCoP = modelState.rAnkle;
            set(optimalCoPPlacementCheckBox, 'Enable', 'off');
            set(optimalFootOrientationCheckBox, 'Enable', 'off');
        else
            [modelState.BoS, modelInfo.maxBoSRadius] = NStep_BoSCreate(modelState.rAnkle, 'Rotate', modelState.BoSAngle);
            updateFootprints();
            set(optimalCoPPlacementCheckBox, 'Enable', 'on');
            set(optimalFootOrientationCheckBox, 'Enable', 'on');
        end

        updateData();
        drawOnce();
    end

    function ReactionMassCheckBoxCallBack(source, eventdata) %#ok<INUSD>
        enableReactionMass = ~enableReactionMass;
        if enableReactionMass
            if ~modelState.haveUsedReactionMass
                boolText = 'on';
            end
        else
            boolText = 'off';
        end
        set(windmillButton, 'Enable', boolText);
        set(windmillAngleSlider,'Enable', boolText)
        set(windmillText,'Enable', boolText)
        set(optimalWindmillAngleButton,'Enable',boolText);
        updateData();
        drawOnce();
    end


    function windmillButtonCallBack(source, eventdata) %#ok<INUSD>
        windmill(get(windmillAngleSlider, 'Value'), modelInfo.tauMax);
    end


    function startButtonCallBack(source, eventdata) %#ok<INUSD>
        stopped = ~stopped;
        if stopped
            set(source, 'String', 'Start');
            uiwait();
        else
            set(source, 'String', 'Stop');
            uiresume();
        end
    end

    function resetButtonCallBack(source, eventdata) %#ok<INUSD>
        stopped = true;
        resetState();
        set(ReactionMassCheckBox,'Enable','on');
        if enableReactionMass
            enableReactionMassControls = 'on';
        else
            enableReactionMassControls = 'off';
        end
        set(windmillButton, 'Enable', enableReactionMassControls);
        set(windmillAngleSlider, 'Enable', enableReactionMassControls)
        set(windmillText, 'Enable', enableReactionMassControls)
        set(optimalWindmillAngleButton, 'Enable', enableReactionMassControls);
        
        removeFootprints();
        if enableBoS
            updateFootprints();
        end
        updateData();
        setAxis();
        drawOnce();
        set(startButton, 'String', 'Start');
        uiwait();
    end


    function keyHandler(source,evnt) %#ok<INUSL>
        key = num2str(evnt.Key);
        if strcmp(key,'l')
            if enableReactionMass && ~modelState.haveUsedReactionMass
            windmillButtonCallBack(windmillButton)
            end
        elseif strcmp(key,'s')  || strcmp(key,'return') || (length(evnt.Modifier) == 1 && strcmp(evnt.Modifier{:},'control'))
            startButtonCallBack(startButton)
        elseif strcmp(key,'r') 
            resetButtonCallBack(resetButton)
        elseif strcmp(key,'b')
            BoSCheckBoxCallBack()
            set(BoSCheckBox,'Value', enableBoS);
        elseif strcmp(key,'e')
            ReactionMassCheckBoxCallBack()
            set(ReactionMassCheckBox,'Value', enableReactionMass);
        elseif strcmp(key,'a')
            autoCaptureCallBack()
            set(autoCaptureCheckBox,'Value', autoCapture);
        else
            return
        end
    end

%%
end % NStep