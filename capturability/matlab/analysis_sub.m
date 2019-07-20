%% Initialize
clear all;
close all;

%% range
global t_max;
t_max = 300; %[us]
global t_step;
t_step = 15; %[us]

time_plot('1','../build/bin/csv/2step_sub.csv',1);
time_plot('2','../build/bin/csv/2step_sub.csv',2);
time_plot('3','../build/bin/csv/2step_sub.csv',3);


%% function
function time_plot(name, data_file,i)
global t_max;
global t_step;
    figure('Name',name);
    hold on;
    % axis equal;
    grid on;
    grid minor;

    data = csvread(data_file,0,0);
    edge = [0:t_step:t_max];
    histogram(data(:,i),edge);

    % average
    ave = mean(data(:,i));
    line('XData',[ave ave],'YData',[10^0 10^6],'Color','k','LineWidth',2);

    % x軸対数化
    %set(gca, 'Xscale', 'log');

    % y軸対数化
    set(gca, 'Yscale', 'log');
    
    xlim([0 t_max]);
    xlabel("Time [us]");
    ylabel("Frequency [-]");
    set(gca,'FontSize',14);
    
    disp('Average: ');
    disp(mean(data(:,i)));
    disp('Max: ');
    disp(max(data(:,i)));
end