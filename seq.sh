#!/bin/sh

for i in `seq 2 11` 
do
	batch=`echo "2^$i"|bc`
	echo "$batch"
	nvprof ./a.out -b $batch 2>&1 | grep kernel | sed -e 's/^.*1000  //' -e 's/\s.*//g' -e 's/us//g' -e 's/ms//g'
done
