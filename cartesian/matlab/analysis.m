% clear all;
close all;

%% load .csv file
data = csvread("simulation.csv",2003,0);
dataRef = csvread("../reference/simulation.csv",2003,0);
dataSize = max(size(data(:,1)));
dataRefSize = max(size(dataRef(:,1)));

%% Color Setting
% color
com_color = 'green';
cop_color = 'red';
icp_color = 'blue';
pendulum_color = 'black';

%% plot XY

figure('Name',"xy");
set(gca,'FontSize',12 );
set(gcf,'renderer','painters');
hold on;

xlabel('x [m]');
ylabel('y [m]');
grid on;
grid minor;

axis equal;

xlabel('x [m]');
ylabel('y [m]');

xlim([-0.5 2.5]);
ylim([-1 1]);
yticks(-1:0.5:1);

% CoMRef
plot(dataRef(:,8),dataRef(:,9),'lineWidth', 1,'Color', com_color,'LineStyle','--');
% ICPRef
line(dataRef(:,12),dataRef(:,13),'lineWidth', 1,'Color', icp_color,'LineStyle','--');

% CoM
plot(data(:,8),data(:,9),'lineWidth', 1,'Color', com_color);
% CoP
scatter(data(:,4),data(:,5), 2, cop_color, 'filled');
% ICP
line(data(:,12),data(:,13),'lineWidth', 1,'Color', icp_color);

% Right Foot
count=2;
for i=1:1:dataSize-1
    vel = (data(i+1,16) - data(i,16))/0.001; % foot velocity z
    if(vel>0&&count==0)
        count=1;
    end
    if(vel<0&&count==1)
        count=2;
    end
    if(vel>0&&count==2)
        count=0;
    end
    if(count==0)
        plotSquare(data(i,14),data(i,15));
    end
end
plotSquare(data(dataSize,14),data(dataSize,15));

% Left Foot
count=2;
for i=1:1:dataSize-1
    vel = (data(i+1,22) - data(i,22))/0.001; % foot velocity z
    if(vel>0&&count==0)
        count=1;
    end
    if(vel<0&&count==1)
        count=2;
    end
    if(vel>0&&count==2)
        count=0;
    end
    if(count==0)
        plotSquare(data(i,20),data(i,21));
    end
end
plotSquare(data(dataSize,20),data(dataSize,21));

%% plot XZ

figure('Name',"xz");
set(gca,'FontSize',12 );
set(gcf,'renderer','painters');
hold on;

xlabel('x [m]');
ylabel('z [m]');
grid on;
grid minor;

axis equal;

xlabel('x [m]');
ylabel('z [m]');

xlim([-0.5 2.5]);
ylim([-0.5 1.5]);
yticks(-0.5:0.5:1.5);

% Reference Pendulum foot
for i=1:1:dataRefSize
    if(rem(i,50)==0)
        x = [dataRef(i,8),dataRef(i,4)];
        z = [1,0];
        line(x,z,'lineWidth', 0.1,'Color',[0.8 0.8 0.8]);
    end
end
% Pendulum foot
for i=1:1:dataSize
    if(rem(i,50)==0)
        x = [data(i,8),data(i,4)];
        z = [1,0];
        line(x,z,'lineWidth', 0.1,'Color', pendulum_color);
    end
end
% CoM
x = [data(1,8),data(dataSize,8)];
z = [1,1];
line(x,z,'lineWidth', 1,'Color', com_color);
% CoP
% z = zeros(i,1);
% scatter(data(:,4),z,2,cop_color, 'filled');
for i=1:1:dataSize
    if(rem(i,50)==0)
        scatter(data(i,4),0,2,cop_color, 'filled');
    end
end

%% square plot function
function plotSquare(x, y)
    footLength = 0.25;
    footWidth = 0.15;
    vertexX(1) = x + footLength/2;
    vertexX(2) = x + footLength/2;
    vertexX(3) = x - footLength/2;
    vertexX(4) = x - footLength/2;
    vertexX(5) = x + footLength/2;
    vertexY(1) = y - footWidth/2;
    vertexY(2) = y + footWidth/2;
    vertexY(3) = y + footWidth/2;
    vertexY(4) = y - footWidth/2;
    vertexY(5) = y - footWidth/2;
    line('XData',vertexX,'YData',vertexY, 'Color', 'black', 'LineWidth', 1);
end