#!/usr/bin/env bash

result="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $result
cd ../
root=`pwd`
makefiles=`find . -name Makefile`

export RUN_MINI=1
export RUN_DUMP=1

for m in $makefiles
do
    cd $root
    dir=`dirname $m`
    echo "Generating output of $dir"
    src=$root/$dir
    dest=$result/$dir
    mkdir -p $src

    stdout=`make -C $src 2>&1`
    ret=$?
    if [ $ret -ne 0 ]; then
            echo -e "* Compiler Error!"
    fi
    $src/run 2> $dest/output.txt
    make clean -C $src &>/dev/null
done
