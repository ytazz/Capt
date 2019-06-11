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
grid on;
grid minor;

% Support foot
draw_foot([0 0],0,'black');

% Steppable region
swft_th_min_ = swft_th_min * pi/180;
swft_th_max_ = swft_th_max * pi/180;
line([sfwt_r_min*cos(swft_th_min_) sfwt_r_max*cos(swft_th_min_)], ... 
        [sfwt_r_min*sin(swft_th_min_) sfwt_r_max*sin(swft_th_min_)], ...
        'Color', 'black')
line([sfwt_r_min*cos(swft_th_max_) sfwt_r_max*cos(swft_th_max_)], ...
        [sfwt_r_min*sin(swft_th_max_) sfwt_r_max*sin(swft_th_max_)], ...
        'Color', 'black')
draw_circle([0 0],sfwt_r_min,[swft_th_min swft_th_max],'black');
draw_circle([0 0],sfwt_r_max,[swft_th_min swft_th_max],'black');

% ICP
data = csvread('icp.csv');
scatter(data(1,1),data(1,2),10,'filled','MarkerFaceColor',[1 0 0]);

% Swing foot
data = csvread('swft.csv');
draw_foot([data(1,1) data(1,2)],1,'black');

% 2-step
% data = csvread('result2.csv',1,0);
% scatter(data(:,7),data(:,8),10,'filled','MarkerFaceColor',[1 0 0]);

% 1-step
data = csvread('region.csv');
scatter(data(:,1),data(:,2),10,'filled','MarkerFaceColor',[0 0 1]);