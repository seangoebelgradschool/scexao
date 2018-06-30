cpuconfig #only needs to be run once. before 20180412 was in pbserver2.sh

sudo cset proc -m -f saphira -t system --force #empty saphira
sudo cset proc -s saphira -e ./pbserver

#pidlist=$( pgrep pbserver | xargs echo | sed 's/ /,/g' )
#sudo chrt -f -p 70 ${pidlist}



