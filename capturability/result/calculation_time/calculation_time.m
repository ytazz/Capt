clear all;
close all;

%% load .csv files

exe_ = csvread("calculation_time_exe.csv",1,0);
save_ = csvread("calculation_time_save.csv",1,0);
sum_ = csvread("calculation_time_sum.csv",1,0);

%% Color Setting
% color
cpu_color = 'blue';
gpu_color = 'red';
% monochro
%cpu_color = 'black';
%gpu_color = [0.7 0.7 0.7];

%% plot EXE time

figure('Name',"exe");
set( gca,'FontSize',12 ); 
hold on;

xlabel('Grid size [-]');
ylabel('Time [s]');
ylim([10^(-4) 10^3]);
grid on;

% CPU
loglog(exe_(:,1),exe_(:,4), 'Color', cpu_color, 'LineWidth', 2);
% errorbar(exe_(:,1),exe_(:,4),exe_(:,5));
set(gca, 'XScale','log', 'YScale','log');
% GPU
loglog(exe_(:,1),exe_(:,2), 'Color', gpu_color, 'LineWidth', 2);
% errorbar(exe_(:,1),exe_(:,2),exe_(:,3));
set(gca, 'XScale','log', 'YScale','log');

legend('CPU','GPU','Location','northwest');

%% plot SAVE time

figure('Name',"save");
set( gca,'FontSize',12 ); 
hold on;

xlabel('Grid size [-]');
ylabel('Time [s]');
ylim([10^(-4) 10^3]);
grid on;

% CPU
loglog(save_(:,1),save_(:,4), 'Color', cpu_color, 'LineWidth', 2);
set(gca, 'XScale','log', 'YScale','log');
% GPU
loglog(save_(:,1),save_(:,2), 'Color', gpu_color, 'LineWidth', 2);
set(gca, 'XScale','log', 'YScale','log');

legend('CPU','GPU','Location','northwest');

%% plot SUM time

figure('Name',"sum");
set( gca,'FontSize',12 ); 
hold on;

xlabel('Grid size [-]');
ylabel('Time [s]');
ylim([10^(-4) 10^3]);
grid on;

% CPU
loglog(sum_(:,1),sum_(:,4), 'Color', cpu_color, 'LineWidth', 2);
set(gca, 'XScale','log', 'YScale','log');
% GPU
loglog(sum_(:,1),sum_(:,2), 'Color', gpu_color, 'LineWidth', 2);
set(gca, 'XScale','log', 'YScale','log');

legend('CPU','GPU','Location','northwest');

%% plot EXE & SUM time

figure('Name',"exe & sum");
set( gca,'FontSize',12 ); 
hold on;

xlabel('Grid size [-]');
ylabel('Time [s]');
ylim([10^(-4) 10^3]);
grid on;

% CPU
loglog(sum_(:,1),sum_(:,4), 'Color', cpu_color, 'LineWidth', 2);
loglog(exe_(:,1),exe_(:,4), '--', 'Color', cpu_color, 'LineWidth', 2);
set(gca, 'XScale','log', 'YScale','log');
% GPU
loglog(sum_(:,1),sum_(:,2), 'Color', gpu_color, 'LineWidth', 2);
loglog(exe_(:,1),exe_(:,2), '--', 'Color', gpu_color, 'LineWidth', 2);
set(gca, 'XScale','log', 'YScale','log');

legend('CPU','CPU(without file output time)','GPU','GPU(without file output time)','Location','northwest');