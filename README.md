# catbot

A jet bot that tracks my cats

python3 train.py --logtostderr --train_dir=../dataset/tf --pipeline_config_path=../dataset/tf/ssd_mobilenet_v2_coco.config

python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path ../dataset/tf/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix ../dataset/tf/model.ckpt-7849 --output_directory ../dataset/tf/trained-inference-graphs/output_inference_graph_v1.pb


export PYTHONPATH=$PYTHONPATH:/var/code/models/research:/var/code/models/research/slim

pip3 install "tensorflow==1.15.*"

export PYTHONPATH=$PYTHONPATH:`pwd`:`pwd`/slim
export PATH=/usr/local/cuda/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda/lib64:$LD_LIBRARY_PATH
export PATH=/usr/local/cuda-10.1/bin:$PATH
export LD_LIBRARY_PATH=/usr/local/cuda-10.1/lib64:$LD_LIBRARY_PATH
export LD_LIBRARY_PATH=$LD_LIBRARY_PATH:/opt/cuda/targets/x86_64-linux/lib/

ldconfig

python3 -c "from tensorflow.python.client import device_lib; print(device_lib.list_local_devices())"

sudo ln -s /usr/local/cuda/targets/x86_64-linux/lib/libcusolver.so.10 /usr/local/cuda/lib64/libcusolver.so.10.0
sudo ln -s /usr/local/cuda/targets/x86_64-linux/lib/libcurand.so.10 /usr/local/cuda/lib64/libcurand.so.10.0
sudo ln -s /usr/local/cuda/targets/x86_64-linux/lib/libcufft.so.10 /usr/local/cuda/lib64/libcufft.so.10.0
sudo ln -s /usr/local/cuda/targets/x86_64-linux/lib/libcudart.so /usr/local/cuda/lib64/libcudart.so.10.0
sudo ln -s /usr/local/cuda/targets/x86_64-linux/lib/libcusparse.so.10 /usr/local/cuda/lib64/libcusparse.so.10.0
sudo ln -s /opt/cuda/targets/x86_64-linux/lib/libcublas.so /opt/cuda/targets/x86_64-linux/lib/libcublas.so.10.0
sudo ln -s /usr/lib/x86_64-linux/libcublas.so /opt/cuda/targets/x86_64-linux/lib/libcublas.so.10.0

python3 train.py --logtostderr --train_dir=../dataset/tf --pipeline_config_path=../dataset/tf/ssd_mobilenet_v2_coco.config

python3 export_inference_graph.py --input_type image_tensor --pipeline_config_path ../dataset/tf/ssd_mobilenet_v2_coco.config --trained_checkpoint_prefix ../dataset/tf/model.ckpt-7849 --output_directory ../dataset/tf/trained-inference-graphs/output_inference_graph_v1.pb
