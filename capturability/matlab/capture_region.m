%% Initialize
clear all;
close all;

%% Parameter
r_foot=0.04;
omega=5.718391;
l_max=0.22;
v_max=1.0;
t_min =0.1;

icp0_r=0.06;
icp0_th=120*3.1415926/180;
swf0_r=0.14;
swf0_th=40*3.1415926/180;

%% Main

icp0=[icp0_r*cos(icp0_th) icp0_r*sin(icp0_th)]';
cop=[r_foot*cos(icp0_th) r_foot*sin(icp0_th)]';
swf0=[swf0_r*cos(swf0_th) swf0_r*sin(swf0_th)]';

swf_r=0;
figure('Name','Capture Region');
one_step_r=0;
two_step_r=0;
three_step_r=0;
one_step_k=1;
two_step_k=1;
three_step_k=1;
one_step_cr_x(1)=0;
one_step_cr_y(1)=0;
two_step_cr_x(1)=0;
two_step_cr_y(1)=0;
three_step_cr_x(1)=0;
three_step_cr_y(1)=0;

for t = 0: 0.01: 0.4
    %% Calculate
    if t<t_min
        swf_r = 0;
    else
        swf_r = v_max * (t - t_min);
    end
    
    icp = (icp0-cop)*exp(omega*t)+cop;
    
    one_step_r = r_foot;
    two_step_r = l_max*exp(-omega*t)+r_foot;
    three_step_r = l_max*(1+exp(-omega*t))*exp(-omega*t)+r_foot;
    
    for j = 0:1:360
        swf_th = j * 3.14159265358 / 180;
        swf_x = swf0(1) + swf_r * cos(swf_th);
        swf_y = swf0(2) + swf_r * sin(swf_th);
        swf = [swf_x swf_y];
        dist = sqrt((swf(1)-icp(1))^2+(swf(2)-icp(2))^2);
        
        sw_theta=atan2(swf_y,swf_x);
        sw_dist=sqrt(swf(1)^2+swf(2)^2);
        if 0.349<=sw_theta && sw_theta<=2.793 && 0.09<=sw_dist && sw_dist<=0.22
            if dist <= one_step_r
                one_step_cr_x(one_step_k) = swf_x;
                one_step_cr_y(one_step_k) = swf_y;
                one_step_k = one_step_k+1;
            elseif dist <= two_step_r
                two_step_cr_x(two_step_k) = swf_x;
                two_step_cr_y(two_step_k) = swf_y;
                two_step_k = two_step_k+1;
            elseif dist <= three_step_r
                three_step_cr_x(three_step_k) = swf_x;
                three_step_cr_y(three_step_k) = swf_y;
                three_step_k = three_step_k+1;
            end
        end
    end

    %% Draw
    clf;
    hold on;
    view(-90, 90);
    axis equal;
    xlim([-0.3 0.3])
    ylim([-0.3 0.3])
    grid on;
    grid minor;

    % support foot
    draw_circle([0 0],r_foot,[0 360],'black');
    
    % steppable region
    line([0.08457 0.20673], [0.03078 0.07524], 'Color', 'black')
    line([-0.08457 -0.20673], [0.03078 0.07524], 'Color', 'black')
    draw_circle([0 0],0.09,[20 160],'black');
    draw_circle([0 0],0.22,[20 160],'black');
    
    % swing foot
    draw_circle(swf0,swf_r,[0 360],'m');
    
    % icp
    draw_circle(icp,0.002,[0 360],'blue');
    
    % 1-step
    draw_circle(icp,one_step_r,[0 360],[1 0 0]);
    scatter(one_step_cr_x,one_step_cr_y,5,'filled','MarkerFaceColor',[0 1 0]);
    
    % 2-step
    draw_circle(icp,two_step_r,[0 360],[0.91 0.42 0.47]);
    scatter(two_step_cr_x,two_step_cr_y,5,'filled','MarkerFaceColor',[0 .7 .7]);
    
    % 3-step
%     draw_circle(icp,three_step_r,[0 360],[0.91 0.42 0.47]);
%     scatter(three_step_cr_x,three_step_cr_y,5,'filled','MarkerFaceColor',[0 .5 .5]);
    
    % pause
    pause(0.1);
end