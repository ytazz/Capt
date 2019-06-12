%% Initialize
clear all;
close all;

%% Param
sfwt_r_min=0.1;
sfwt_r_max=0.2;
swft_th_min=20; %[deg]
swft_th_max=160; %[deg]

%% Plot
figure('Name','Result');
hold on;
view(-90, 90);
axis equal;
xlim([-0.3 0.3]);
ylim([-0.3 0.3]);
% grid on;
% grid minor;

% Grid
% minor
for grid_r = 0:0.01:0.3
    draw_circle([0 0],grid_r,[0 360],[0.9 0.9 0.9]);
    for grid_th = 0:pi/18:2*pi
        line([0 grid_r*cos(grid_th)],[0 grid_r*sin(grid_th)],'Color',[0.9 0.9 0.9]);
    end
end
    
% Support foot
draw_foot([0 0],0,[0.5 0.5 0.5]);
draw_foot_polygon([0 0],0,'black');

% Steppable region
swft_th_min_ = swft_th_min * pi/180;
swft_th_max_ = swft_th_max * pi/180;
line([sfwt_r_min*cos(swft_th_min_) sfwt_r_max*cos(swft_th_min_)], ... 
        [sfwt_r_min*sin(swft_th_min_) sfwt_r_max*sin(swft_th_min_)], ...
        'Color', [0.5 0.5 0.5])
line([sfwt_r_min*cos(swft_th_max_) sfwt_r_max*cos(swft_th_max_)], ...
        [sfwt_r_min*sin(swft_th_max_) sfwt_r_max*sin(swft_th_max_)], ...
        'Color', [0.5 0.5 0.5])
draw_circle([0 0],sfwt_r_min,[swft_th_min swft_th_max],[0.5 0.5 0.5]);
draw_circle([0 0],sfwt_r_max,[swft_th_min swft_th_max],[0.5 0.5 0.5]);

% ICP
data = csvread('icp.csv');
scatter(data(1,1),data(1,2),20,'filled','MarkerFaceColor',[1 0 0]);

% Swing foot
data = csvread('swft.csv');
draw_foot([data(1,1) data(1,2)],1,[0.5 0.5 0.5]);
draw_foot_polygon([data(1,1) data(1,2)],1,'black');

% Modified capture region
% data = csvread('region2.csv');
% scatter(data(:,1),data(:,2),20,'filled','MarkerFaceColor',[0 0 1]);

% Capture region
data = csvread('region.csv');
scatter(data(:,1),data(:,2),20,'filled','MarkerFaceColor',[0 0 1]);

%com
% data = csvread('com.csv');
% scatter(data(:,1),data(:,2),20,'filled','MarkerFaceColor',[0 1 1]);

%com ok
% data = csvread('com_ok.csv');
% scatter(data(:,1),data(:,2),10,'filled','MarkerFaceColor',[1 1 0]);