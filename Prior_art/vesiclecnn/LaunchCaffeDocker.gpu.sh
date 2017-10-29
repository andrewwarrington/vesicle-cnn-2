sudo usermod -aG docker $USER
source ~/.bashrc

CONTAINER_NAME=$(date +%s%N)
echo $CONTAINER_NAME

CUR_DIR=$(pwd)
echo $CUR_DIR

# Create the docker session with the name CAFFE.
nvidia-docker run -u $(id -u):$(id -g) -v $CUR_DIR:/workspace --name $CONTAINER_NAME -it bvlc/caffe:gpu /bin/bash ./ProcessScript.sh # ./ProcessScript.sh
