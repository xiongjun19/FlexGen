We have updated the script so now it needs to run outside Docker container.

There are multiple options to run the demo, but before running we need to start the 
ui server. 

For this open up a terminal and activate the conda environment. Please make sure that execute the script as:

cd FlexGen/project_v4/
python serv/main.py

and wait then run the application by going to a new Terminal and then,


cd FlexGen/flexgen
sudo chmod +x rundk-large.sh 
./rundk-large.sh --cxl-offload

During the initial run if you find any Permission denied error, please sudo chmod 777 /tmp, in the case of directory.
And use sudo chmod 777 /path/to/bashscript.sh

You can kill the process by using general Cltr + C key and keep it pressing until you see Memlogger Stopped!!!
The Stopped message will come soon, if it does not come after a while then close the Terminal and use a new one, it will only happen in case of some sever performance degradation simulation. 


You can also try :

./rundk-large.sh --memverge-offload

wait for the memverge to start its service and it will take up all cxl memory and then it will run. The UI will auto take the data 

You can also try :

./rundk-large.sh --normal-offload


For the cxl-sim to show performance degration:
./rundk-large.sh --cxl-offload-sim

Please note that the performance degration is observed by setting the value of memory limitation of cxl to
# Set the memory maximum size for the control group
MEMSIZE_B=70000 #70000->0.842 tokens/s,   #90000->1.65 tokens/s ,  #80000->1.671 tokens/s,  # better use above 80000# Calculate the memory limit in bytes based on the memory size.
Note that the application would probably get killed in case of extremely insufficient memory by the Linux Kernel.

You can wait for the first time in the Browser. 
Use these links:

http://127.0.0.1:9980/#/screen1

http://127.0.0.1:9980/#/screen2

