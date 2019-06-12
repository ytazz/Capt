function draw_foot_polygon(center, mode, color) % mode: 0->right, 1->left
    center_x = center(1);
    center_y = center(2);
   
     vertex_x= [-0.065
                -0.055
                +0.015
                +0.035
                +0.065
                +0.065
                +0.035
                -0.065
                -0.065];
 
     vertex_y= [-0.025
                -0.035
                -0.035
                -0.035
                -0.015
                +0.025
                +0.035
                +0.025
                -0.025];
    if mode == 0
        line('XData',center_x+vertex_x,'YData',center_y+vertex_y,'Color',color);
    elseif mode ==1
        line('XData',center_x+vertex_x,'YData',center_y-vertex_y,'Color',color);
    end
end