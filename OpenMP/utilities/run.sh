#!/usr/bin/env bash
set -e
root=`pwd`
echo $root
for d in */ ; do
    cd $root
    echo "$d"
    cd $d
    #make DYN=1
    #./*_acc
    #make DYN=1 PRINT=1 2> .output.txt
done

cd $root
for d in */ ; do
    cd $root
    cd $d
    make veryclean
done
