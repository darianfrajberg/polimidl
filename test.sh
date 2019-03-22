#!/bin/sh

out_directory="./out"

rm -rf $out_directory
mkdir $out_directory

./tests/build.sh "./tests/layers" $out_directory
./tests/build.sh "./tests/mobilenet" $out_directory
./tests/build.sh "./tests/network" $out_directory

./tests/run.sh $out_directory

rm -rf $out_directory
