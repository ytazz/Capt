function draw_circle(center, radius, limit, color)
    center_x = center(1);
    center_y = center(2);
    min = limit(1); %[deg]
    max = limit(2);

    resolution = 1;

    k=1;
    for i=min:resolution:max
        rad = i * 3.14159265358979 / 180;
        x(k)=center_x+radius*cos(rad);
        y(k)=center_y+radius*sin(rad);
        k=k+1;
    end
    
    line('XData',x,'YData',y,'Color',color);
end