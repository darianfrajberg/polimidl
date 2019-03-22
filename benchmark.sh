#!/bin/sh

out_directory="./out"

rm -rf $out_directory
mkdir $out_directory

./benchmarks/build.sh "./benchmarks" $out_directory

./benchmarks/run.sh $out_directory

rm -rf $out_directory
