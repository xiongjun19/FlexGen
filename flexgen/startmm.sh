#!/bin/sh -e

echo 64119 > /sys/devices/system/node/node2/hugepages/hugepages-2048kB/nr_hugepages
#echo 128238 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
fallocate -l $(numastat -m | grep HugePages_Free | awk '{print $4}')M /dev/hugepages/dax0
chmod 777 /dev/hugepages/dax0
/opt/memverge/sbin/mvmallocd --format_pmem --config=/etc/memverge/mvmallocd.yml

systemctl restart mvmallocd mvsvcd mvcsnapd mmagent mmctl

echo 20480 > /sys/devices/system/node/node0/hugepages/hugepages-2048kB/nr_hugepages
echo 20480 > /sys/devices/system/node/node1/hugepages/hugepages-2048kB/nr_hugepages
exit 0
