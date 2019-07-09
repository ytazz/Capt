%% Initialize
clear all;
close all;

%% range
global t_max;
t_max = 10000; %[us]
global t_step;
t_step = 200; %[us]

%% 0step
time_plot('0step','../build/csv/analysis_0.csv');

%% 1step
time_plot('1step','../build/csv/analysis_1.csv');

%% 2step
time_plot('2step','../build/csv/analysis_2.csv');


%% function
function time_plot(name, data_file)
global t_max;
global t_step;
    figure('Name',name);
    hold on;
    % axis equal;
    grid on;
    grid minor;

    data = csvread(data_file,1,0);
    edge = [0:t_step:t_max];
    histogram(data(:,1),edge);

    % average
    ave = mean(data(:,1));
    line('XData',[ave ave],'YData',[10^0 10^6],'Color','k','LineWidth',2);

    % x軸対数化
    %set(gca, 'Xscale', 'log');

    % y軸対数化
    set(gca, 'Yscale', 'log');
    
    xlim([0 t_max]);
    xlabel("Time [us]");
    ylabel("Frequency [-]");
    set(gca,'FontSize',14);
    
    disp(max(data(:,1)));
end