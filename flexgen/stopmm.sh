#!/bin/bash -e
/opt/memverge/bin/mvmcli snapshot list

if [[ $( /opt/memverge/bin/mvmcli snapshot list | grep 'No instance is running' | wc -l ) -ne 1 ]] ; then
    echo ERROR: there are still apps running on Memory Machine
    exit
fi

systemctl stop mvmallocd mvsvcd mvcsnapd mmagent mmctl

echo 0 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
echo 0 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages

rm /dev/hugepages/dax0

echo 0 > /sys/devices/system/node/node2/hugepages/hugepages-2048kB/nr_hugepages
