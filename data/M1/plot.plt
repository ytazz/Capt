set datafile separator ","
set xlabel "l_{step}"
set ylabel "r_{ic}"
set xrange [0.0:0.25]
set yrange [-0.2:0.2]

# P
plot 'Biped_4_plot.csv' using 1:2 t 'P_00' with points pt 7 lc rgb 'grey10',\
'Biped_2_plot.csv' using 1:2 t 'P_2' with points pt 7 lc rgb 'grey50',\
'Biped_1_plot.csv' using 1:2 t 'P_1' with points pt 7 lc rgb 'grey70',\
'Biped_0_plot.csv' using 1:2 t 'P_0' with points pt 7 lc rgb 'grey100'

# C
plot 'Biped_4_plot.csv' using 1:2 t 'P_00' with points pt 7 lc rgb 'grey10',\
'2d_const_3_non(0.6).csv' using 1:2 t 'C_00' with points pt 7 lc rgb 'grey70',\
'Biped_0_plot.csv' using 1:2 t 'P_0' with points pt 7 lc rgb 'grey100'

# C short
plot '2d_const_short_3_non.csv' using 1:2 t 'C_00' with points pt 7 lc rgb 'grey70',\
'Biped_4_plot.csv' using 1:2 t 'P_00' with points pt 7 lc rgb 'grey10',\
'Biped_0_plot.csv' using 1:2 t 'P_0' with points pt 7 lc rgb 'grey100'

# simulation result
plot 'Biped_4_plot.csv' using 1:2 t 'P_00' with points pt 7 lc rgb 'grey10',\
'Biped_2_plot.csv' using 1:2 t 'P_2' with points pt 7 lc rgb 'grey50',\
'Biped_1_plot.csv' using 1:2 t 'P_1' with points pt 7 lc rgb 'grey70',\
'Biped_0_plot.csv' using 1:2 t 'P_0' with points pt 7 lc rgb 'grey100',\
'1step-2100.csv' using 1:2 t 'ss' with points pt 11 ps 1.5 lc rgb 'grey10'
