#!/bin/sh

if [ $# -ne 2 ]; then
    echo "Error! Missing arguments";
    exit 1;
fi

OS="`uname`"
case $OS in
  'Linux')
    pthread_flag='-pthread'
    ;;
  'FreeBSD')
    pthread_flag='-pthread'
    ;;
  'WindowsNT')
    ;;
  'Darwin')
    pthread_flag='-lpthread'
    ;;
  'SunOS')
    ;;
  'AIX') ;;
  *) ;;
esac

export GCCPAR="-Iinclude -Ieigen -std=c++17 -lstdc++ -fexceptions -Wall $pthread_flag -O3 -DNDEBUG -DEIGEN_DONT_PARALLELIZE -DEIGEN_NO_MALLOC"

tests_directory=$1
out_directory=$2

# Build tests
for file in "$tests_directory"/*.cpp; do
    echo "Building $(basename "${file%.*}")..."
g++ ${GCCPAR} $file  -o "$out_directory"/$(basename "${file%.*}");
done
