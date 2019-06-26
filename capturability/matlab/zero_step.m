%% Initialize
clear all;
close all;

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
draw_foot([0 0],0,[0.5 0.5 0.5]);
draw_foot_polygon([0 0],0,'black');

% ICP
data = csvread('~/study/capturability/build/cap.csv');
scatter(data(:,1),data(:,2),20,'filled','MarkerFaceColor',[1 0 0]);

% Swing foot
draw_foot([0.05 0.11],1,[0.5 0.5 0.5]);
draw_foot_polygon([0.05 0.11],1,'black');