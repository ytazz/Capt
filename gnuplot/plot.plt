set size ratio 1
set palette gray negative
set autoscale xfix
set autoscale yfix
set xtics 1
set ytics 1
set title "Resolution Matrix for E"

set cbrange[0:3]
set cbtics 1

set tics scale 0,0.001
set mxtics 2
set mytics 2
set grid front mxtics mytics lw 1.5 lt -1 lc rgb 'white'
plot "datafile.dat" matrix w image noti
