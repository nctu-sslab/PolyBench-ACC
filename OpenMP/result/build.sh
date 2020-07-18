#!/usr/bin/env bash

result="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"
cd $result
cd ../
root=`pwd`
makefiles=`find . -name Makefile`

export RUN_MINI=1
export RUN_DUMP=1
export RUN_NOOMP=1

for m in $makefiles
do
    cd $root
    dir=`dirname $m`

    if [[ $dir == "./result" ]]; then
        continue
    fi

    src=$root/$dir
    dest=$result/$dir

    if [[ -s $dest/output.txt ]]; then
        echo "$dir/output.txt exists"
        continue
    fi

    echo "Generating $dir/output.txt"
    mkdir -p $dest

    stdout=`make -C $src 2>&1`
    ret=$?
    if [ $ret -ne 0 ]; then
        echo -e "* Compiler Error!"
        continue
        make clean -C $src &>/dev/null
    fi
    $src/run 2> $dest/output.txt
    ret=$?
    if [[ ! $ret -eq 0 ]]; then
        echo -e "* Runtime Error!"
        rm -f $dest/output.txt
    fi
    make clean -C $src &>/dev/null
done
