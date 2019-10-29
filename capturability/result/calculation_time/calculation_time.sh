#!/bin/sh

# NOTE:
# build/binにinstallして使用すること!

for itr in 1 2 3 4 5
do
  echo "GPU ${itr}-th analysis"
  ./main_gpu
done

for itr in 1 2 3 4 5
do
  echo "CPU ${itr}-th analysis"
  ./main_cpu
done