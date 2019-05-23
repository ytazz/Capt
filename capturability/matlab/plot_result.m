%% Initialize
clear all;
close all;

%% Load 
data = csvread('result.csv',1,0);

%% Plot
figure('Name','Result');
hold on;
view(-90, 90);
axis equal;
xlim([-0.3 0.3])
ylim([-0.3 0.3])
grid on;
grid minor;

% support foot
draw_circle([0 0],0.04,[0 360],'black');

% steppable region
line([0.08457 0.20673], [0.03078 0.07524], 'Color', 'black')
line([-0.08457 -0.20673], [0.03078 0.07524], 'Color', 'black')
draw_circle([0 0],0.09,[20 160],'black');
draw_circle([0 0],0.22,[20 160],'black');

% 1-step
scatter(data(:,7),data(:,8),10,'filled','MarkerFaceColor',[0 0 1]);