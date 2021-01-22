set encoding utf8
#set terminal qt size 600,600

#set terminal gif animate optimize delay 50 size 600,900
#set output 'plot.gif'
set terminal svg
set output 'plot.svg'
#set terminal png size 400,400
#set output 'plot.png'
#set terminal postscript eps enhanced
#set output 'plot.eps'

# グラフサイズ設定
#set size square
set size ratio -1

# 軸ラベル設定
set xlabel 'y [m]'
set ylabel 'x [m]'
set xlabel  font "Arial,15"
set ylabel  font "Arial,15"
set tics    font "Arial,15"
set cblabel font "Arial,15"
set key     font "Arial,15"

# 座標軸の目盛り設定
set xtics 1
set ytics 1
set mxtics 2
set mytics 2
set xtics scale 0,0.001
set ytics scale 0,0.001

set xrange [-0.55:0.10]
set yrange [-0.45:0.45]

# カラーバーの設定
set palette gray negative
set palette defined ( 0 '#ffffff', 1 '#cbfeff', 2 '#68fefe', 3 '#0097ff', 4 '#0000ff')
#set cbrange [0:5]
#set cbtics 0.5
#set palette maxcolors 5
#set cbtics scale 0,0.001
#set cblabel "N-step capture point"
unset colorbox

# 描画
plot "data.dat" using ($1):($2):($5+1) with points palette pt 5 ps 0.2 notitle,\
     "data.dat" using ($3):($4):($5+1) with points palette pt 5 ps 0.2 notitle,\
     "landing0.dat" with lines  lw 1 lc "dark-blue" notitle,\
     "landing1.dat" with lines  lw 1 lc "dark-blue" notitle,\
     "sup.dat"     with lines  lw 1 lc "black"     notitle,\
     "swg.dat"     with lines  lt 0 dt 1 lw 2 lc "black" notitle,\
     "icp.dat"     with points pt 1 lc 1 ps 2            notitle

set size ratio 0.5
set xlabel 'N'
set ylabel ''
set xrange [-0.5:10.5]
set yrange [0:0.025]
set ytics 0.005
set boxwidth 0.2 relative
set style fill solid
set output 'basin_size.svg'
plot "basin_size_verylow.csv" using ($1-0.3):($3)   with boxes title '0.080[m] x 0.080[m] x 0.20[rad]',\
     "basin_size_low.csv" using ($1-0.1):($3)       with boxes title '0.040[m] x 0.040[m] x 0.10[rad]',\
     "basin_size_mid.csv" using ($1+0.1):($3)       with boxes title '0.020[m] x 0.020[m] x 0.05[rad]',\
     "basin_size_midhigh.csv" using ($1+0.3):($3)   with boxes title '0.015[m] x 0.015[m] x 0.05[rad]'

