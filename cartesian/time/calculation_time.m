%% Clear
clear all;
close all;

%% Data Setting
numCase = 12;
cpu=zeros(numCase, 6);
gpu=zeros(numCase, 6);
cpu_=zeros(numCase, 6);
gpu_=zeros(numCase, 6);

numIterate = 5;
numType = 6;
data=zeros(numIterate,numType);

%% Color Setting
% color
cpu_color = 'blue';
gpu_color = 'red';
% monochro
%cpu_color = 'black';
%gpu_color = [0.7 0.7 0.7];

%% Load .csv files
for i=1:1:numCase
    % CPU
    for j=1:1:numIterate
        file = "setting" + i + "/log_cpu" + j + ".csv";
        data(j,:) = csvread(file,1,0);
    end
    for k=1:1:numType
        cpu_(i,k) = mean(data(:,k));
    end
    
    %GPU
    for j=1:1:numIterate
        file = "setting" + i + "/log_gpu" + j + ".csv";
        if exist(file) ~= 0
            data(j,:) = csvread(file,1,0);
        else
            data(j,:) = ones(1,numType);
        end
    end
    for k=1:1:numType
        gpu_(i,k) = mean(data(:,k));
    end
end

%sort
cpu = sortrows(cpu_,3);
gpu = sortrows(gpu_,3);

%% Plot (execution time)

figure('Name',"execution time");
set( gca,'FontSize',12 ); 
hold on;

xlabel('Grid size [-]');
ylabel('Time [s]');
xlim([10^5 10^10]);
ylim([10^(-2) 10^3]);
grid on;

% CPU
loglog(cpu(:,3),cpu(:,4)/1000, 'Color', cpu_color, 'LineWidth', 2);
set(gca, 'XScale','log', 'YScale','log');
% GPU
loglog(gpu(:,3),gpu(:,4)/1000, 'Color', gpu_color, 'LineWidth', 2);
set(gca, 'XScale','log', 'YScale','log');

legend('CPU','GPU','Location','northwest');