docker build -t flexgen -f Dockerfile.torch .
cnt_name='cnt_fg'
loc_data='/data/flexgen'
docker run --name ${cnt_name} -it --gpus all --ipc=host -v `pwd`:/workspace/flex -v ${loc_data}:/workspace/data flexgen
