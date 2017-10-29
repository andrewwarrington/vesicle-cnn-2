CONTAINER_NAME=$(date +%s%N)
echo $CONTAINER_NAME

CUR_DIR=$(pwd)
echo $CUR_DIR

# Create the docker session with the name CAFFE.
sudo docker run -v $CUR_DIR:/workspace --name $CONTAINER_NAME -it bvlc/caffe:cpu make train-all

