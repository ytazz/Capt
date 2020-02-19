%% Load
data = csvread('computation.csv',1,0);

%% シミュレーション時間ごとの計算時間
figure('Name','calculation_time');

hold on;
grid on;
grid minor;

xlim([0 10]);
ylim([0 20]);
xlabel("Simulation Time [s]");
ylabel("Computation Time [ms]");
set(gca,'FontSize',14);

line(data(:,1)-2,data(:,2),'Color','k','LineWidth',0.5);

%% 計算時間のヒストグラム
figure('Name','calculation_time');

hold on;
grid on;
grid minor;

histogram(data(:,2),[0:0.5:20]);

xlim([0 20]);
% ylim([0 1000]);
xlabel("Time [ms]");
ylabel("Frequency [-]");
set(gca,'FontSize',14);

ave = mean(data(:,2));
line('XData',[ave ave],'YData',[0 500],'Color','k','LineWidth',1);

disp('Average: ');

disp(mean(data(:,2)));

disp('Max: ');

disp(max(data(:,2)));