
GPU := "0"

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
	python vesicle-cnn-2.py --deploy_pretrained "VCNN-2_train_2017_10_19_16_17_25_0" \
				--gpu ${GPU} \
				--deploy_train \
                                --deploy_validation \
                                --deploy_test




