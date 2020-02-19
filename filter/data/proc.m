%% Init.
clear all;
close all;

%% 
figure('Name','real');

data = csvread('data.csv',1,0);
plot(data(:,1),data(:,2),'k');
hold on;
plot(data(:,1),data(:,3),'b');

grid on;
grid minor;
ylim([0 0.7]);

%% 
figure('Name','LPF_09');

data = csvread('data.csv',1,0);
plot(data(:,1),data(:,2),'k');
hold on;
plot(data(:,1),data(:,4),'r');

grid on;
grid minor;
ylim([0 0.7]);

%% 
figure('Name','LPF_08');

data = csvread('data.csv',1,0);
plot(data(:,1),data(:,2),'k');
hold on;
plot(data(:,1),data(:,5),'r');

grid on;
grid minor;
ylim([0 0.7]);

%% 
figure('Name','LPF_05');

data = csvread('data.csv',1,0);
plot(data(:,1),data(:,2),'k');
hold on;
plot(data(:,1),data(:,6),'r');

grid on;
grid minor;
ylim([0 0.7]);

%% 
figure('Name','LPF_02');

data = csvread('data.csv',1,0);
plot(data(:,1),data(:,2),'k');
hold on;
plot(data(:,1),data(:,7),'r');

grid on;
grid minor;
ylim([0 0.7]);