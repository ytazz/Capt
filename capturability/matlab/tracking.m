%% Initialize
clear all;
close all;

%% Plot
figure('Name','Com');
hold on;
% axis equal;
grid on;
grid minor;

% CoM
data = csvread('~/choreonoid/build/data.csv',1,0);
line('XData',data(:,1),'YData',data(:,2),'Color','r','LineStyle','--');
line('XData',data(:,1),'YData',data(:,3),'Color','g','LineStyle','--');
line('XData',data(:,1),'YData',data(:,4),'Color','b','LineStyle','--');
line('XData',data(:,1),'YData',data(:,5),'Color','r');
line('XData',data(:,1),'YData',data(:,6),'Color','g');
line('XData',data(:,1),'YData',data(:,7),'Color','b');

%% Plot
figure('Name','Torso');
hold on;
% axis equal;
grid on;
grid minor;

% RLEG
data = csvread('~/choreonoid/build/data.csv',1,0);
line('XData',data(:,1),'YData',data(:,8),'Color','r','LineStyle','--');
line('XData',data(:,1),'YData',data(:,9),'Color','g','LineStyle','--');
line('XData',data(:,1),'YData',data(:,10),'Color','b','LineStyle','--');
line('XData',data(:,1),'YData',data(:,11),'Color','r');
line('XData',data(:,1),'YData',data(:,12),'Color','g');
line('XData',data(:,1),'YData',data(:,13),'Color','b');

%% Plot
figure('Name','Right');
hold on;
% axis equal;
grid on;
grid minor;

% RLEG
data = csvread('~/choreonoid/build/data.csv',1,0);
line('XData',data(:,1),'YData',data(:,14),'Color','r','LineStyle','--');
line('XData',data(:,1),'YData',data(:,15),'Color','g','LineStyle','--');
line('XData',data(:,1),'YData',data(:,16),'Color','b','LineStyle','--');
line('XData',data(:,1),'YData',data(:,17),'Color','r');
line('XData',data(:,1),'YData',data(:,18),'Color','g');
line('XData',data(:,1),'YData',data(:,19),'Color','b');


%% Plot
figure('Name','Left');
hold on;
% axis equal;
grid on;
grid minor;

% LLEG
data = csvread('~/choreonoid/build/data.csv',1,0);
line('XData',data(:,1),'YData',data(:,20),'Color','r','LineStyle','--');
line('XData',data(:,1),'YData',data(:,21),'Color','g','LineStyle','--');
line('XData',data(:,1),'YData',data(:,22),'Color','b','LineStyle','--');
line('XData',data(:,1),'YData',data(:,23),'Color','r');
line('XData',data(:,1),'YData',data(:,24),'Color','g');
line('XData',data(:,1),'YData',data(:,25),'Color','b');

%% Tracking
figure('Name','IK');
hold on;
% axis equal;
grid on;
grid minor;

data = csvread('~/choreonoid/build/data_ik.csv',1,0);
line('XData',data(:,1),'YData',data(:,2),'Color','r','LineStyle','--');
%line('XData',data(:,1),'YData',data(:,3),'Color','g','LineStyle','--');
%line('XData',data(:,1),'YData',data(:,4),'Color','b','LineStyle','--');

%% Joint
figure('Name','Joints(Left)');
hold on;
% axis equal;
grid on;
grid minor;

data = csvread('~/choreonoid/build/data_joint.csv',1,0);
line('XData',data(:,1),'YData',data(:,2),'Color','r','LineStyle','--');
line('XData',data(:,1),'YData',data(:,4),'Color','g','LineStyle','--');
line('XData',data(:,1),'YData',data(:,6),'Color','b','LineStyle','--');
line('XData',data(:,1),'YData',data(:,8),'Color','y','LineStyle','--');
line('XData',data(:,1),'YData',data(:,10),'Color','m','LineStyle','--');
line('XData',data(:,1),'YData',data(:,12),'Color','k','LineStyle','--');
line('XData',data(:,1),'YData',data(:,3),'Color','r');
line('XData',data(:,1),'YData',data(:,5),'Color','g');
line('XData',data(:,1),'YData',data(:,7),'Color','b');
line('XData',data(:,1),'YData',data(:,9),'Color','y');
line('XData',data(:,1),'YData',data(:,11),'Color','m');
line('XData',data(:,1),'YData',data(:,13),'Color','k');