#!/bin/sh

# NOTE: build/binにinstallして使用すること!

for itr_setting in 1 2 3 4 5 6 7 8 9 10 11 12
do

unset root
${root:=./time/setting${itr_setting}}
rm ${root}/*.csv

echo "${itr_setting}-th setting"

cp ${root}/valkyrie_xy.xml data/

for itr_analysis in 1 2 3 4 5
do
  echo "CPU ${itr_analysis}-th analysis"
  ./main_cpu
  cp log.csv ${root}/
  mv ${root}/log.csv ${root}/log_cpu${itr_analysis}.csv
done

for itr_analysis in 1 2 3 4 5
do
  echo "GPU ${itr_analysis}-th analysis"
  ./main_gpu
  cp log.csv ${root}/
  mv ${root}/log.csv ${root}/log_gpu${itr_analysis}.csv
done

done