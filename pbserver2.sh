#!/bin/bash

#cpuconfig #only needs to be run once 

sudo cset proc -m -f saphira2 -t system --force #empty saphira2
echo assigning $1 to saphira2
sudo cset proc -m -p $1 -t saphira2 --force

pidlist=$( pgrep -x pbserver | xargs echo | sed 's/ /,/g' )
sudo chrt -f -p 70 ${pidlist} 
sudo chrt -f -p 50 $1

cset set -l
