tmux new -s train_session

# pull docker image from docker hub
sudo docker pull waynechu1109/droplab_research:a100_latest

# run docker  
sudo docker run --gpus all -it \
  -v ~/Wayne/home/waynechu/droplab_research:/root/droplab_research \
  waynechu1109/droplab_research:a100_latest \
  /bin/bash