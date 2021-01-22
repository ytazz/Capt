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
#set palette gray negative
set palette defined ( 0 '#cbfeff', 1 '#68fefe', 2 '#0097ff', 3 '#0000ff')
#set cbrange [0:5]
#set cbtics 0.5
#set palette maxcolors 5
#set cbtics scale 0,0.001
#set cblabel "N-step capture point"
unset colorbox

# 描画
plot "data3.dat" using 1:2:5 with points palette pt 5 ps 0.2 notitle,\
     "data3.dat" using 3:4:5 with points palette pt 5 ps 0.2 notitle,\
     "data2.dat" using 1:2:5 with points palette pt 5 ps 0.2 notitle,\
     "data2.dat" using 3:4:5 with points palette pt 5 ps 0.2 notitle,\
     "data1.dat" using 1:2:5 with points palette pt 5 ps 0.2 notitle,\
     "data1.dat" using 3:4:5 with points palette pt 5 ps 0.2 notitle,\
     "data0.dat" using 1:2:5 with points palette pt 5 ps 0.2 notitle,\
     "data0.dat" using 3:4:5 with points palette pt 5 ps 0.2 notitle,\
     "landing0.dat" with lines  lw 1 lc "dark-blue" notitle,\
     "landing1.dat" with lines  lw 1 lc "dark-blue" notitle,\
     "sup.dat"     with lines  lw 1 lc "black"     notitle,\
     "swg.dat"     with lines  lt 0 dt 1 lw 2 lc "black" notitle,\
     "icp.dat"     with points pt 1 lc 1 ps 2            notitle

set xlabel 'N'
set ylabel ''
set xrange [-0.5:10.5]
set yrange [0:3000000]
set boxwidth 0.5 relative
set style fill solid
set output 'basin_size.svg'
plot "basin_size_mid.csv" with boxes,\
     "basin_size_low.csv" with boxes

