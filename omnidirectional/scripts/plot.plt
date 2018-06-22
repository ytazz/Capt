
splot "statesSpace.csv" u 1:2:4 with points palette pointsize 0.5 pointtype 7

set polar
set grid polar 30

plot "case1/0step.csv" u 2:1 with points pointsize 1.5 pointtype 7 lc "blue", \
     "case1/1step.csv" u 2:1 with points pointsize 1.5 pointtype 7 lc "red", \
     "case1/2step.csv" u 2:1 with points pointsize 1.5 pointtype 7 lc "black"

plot "case2/0step.csv" u 2:1 with points pointsize 1.5 pointtype 7 lc "blue", \
     "case2/1step.csv" u 2:1 with points pointsize 1.5 pointtype 7 lc "red", \
     "case2/2step.csv" u 2:1 with points pointsize 1.5 pointtype 7 lc "black"

plot "case3/0step.csv" u 2:1 with points pointsize 1.5 pointtype 7 lc "blue", \
     "case3/1step.csv" u 2:1 with points pointsize 1.5 pointtype 7 lc "red", \
     "case3/2step.csv" u 2:1 with points pointsize 1.5 pointtype 7 lc "black"
