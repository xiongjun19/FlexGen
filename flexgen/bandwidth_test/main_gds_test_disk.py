import os
import time
cmd = ["/usr/local/cuda-12/gds/tools/gdsio -V -f /tmp/test -d 0 -w 1 -s 4k -x 0 -i 1k:4k:4k -I 1",
"/usr/local/cuda-12/gds/tools/gdsio -V -f /tmp/test -d 0 -w 1 -s 16k -x 0 -i 1k:16k:4k -I 1",
"/usr/local/cuda-12/gds/tools/gdsio -V -f /tmp/test -d 0 -w 1 -s 64k -x 0 -i 1k:64k:4k -I 1",
"/usr/local/cuda-12/gds/tools/gdsio -V -f /tmp/test -d 0 -w 1 -s 256k -x 0 -i 1k:256k:4k -I 1",
"/usr/local/cuda-12/gds/tools/gdsio -V -f /tmp/test -d 0 -w 1 -s 1M -x 0 -i 1k:1M:4k -I 1",
"/usr/local/cuda-12/gds/tools/gdsio -V -f /tmp/test -d 0 -w 4 -s 4M -x 0 -i 1k:1M:4k -I 1",
"/usr/local/cuda-12/gds/tools/gdsio -V -f /tmp/test -d 0 -w 4 -s 16M -x 0 -i 1k:1M:4k -I 1",
"/usr/local/cuda-12/gds/tools/gdsio -V -f /tmp/test -d 0 -w 4 -s 64M -x 0 -i 1k:1M:4k -I 1",
"/usr/local/cuda-12/gds/tools/gdsio -V -f /tmp/test -d 0 -w 4 -s 256M -x 0 -i 1k:1M:4k -I 1"]
for c in cmd: 
    os.system("sudo "+ c)
    time.sleep(10)
    # os.remove("/local2/tmp/test")
