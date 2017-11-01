
GPU := "1"

default: 
	@echo "Please specify a target to make."

train-only:
	python vesicle-cnn-2.py --train_new \
				--gpu ${GPU} 

train-deploy-all:
	python vesicle-cnn-2.py --train_new \
				--gpu ${GPU} \
				--deploy_train \
				--deploy_validation \
				--deploy_test \

pretrained-deploy-all:
	python vesicle-cnn-2.py --deploy_pretrained "vesicle-cnn-2-example" \
				--gpu ${GPU} \
				--deploy_train \
                                --deploy_validation \
                                --deploy_test  # --deploy_pretrained "vesicle-cnn-2-example"




