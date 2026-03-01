# Fast Image-to-Neural Surface: FINS [ICRA 2026 Accepted]

<h3 align="center">
  <a href="https://arxiv.org/abs/2509.20681">Paper</a>
  ·
  <a href="#citation">Citation</a>
</h3>

<p align="center">
  <img src="./media/dtu_114_32.png" height="220" />
  <img src="./media/sdf_buddha.gif" height="220" />
  <img src="./media/franka_side.gif" height="220" />
</p>


## Abstract
**Fast Image-to-Neural Surface (FINS)** reconstructs high-fidelity signed distance fields (SDFs) from as little as a single RGB image in just a few seconds.

Unlike traditional neural surface methods that require dense multi-view supervision and long optimization times, FINS leverages pretrained 3D foundation models to generate geometric priors, combined with multi-resolution hash encoding and lightweight SDF heads for rapid convergence.

The resulting implicit representation enables real-time surface reconstruction and supports downstream robotics tasks such as motion planning, obstacle avoidance, and surface following.

FINS bridges single-image perception and fast neural implicit modeling, making SDF construction practical for real-world robotic systems.

## Setup
```bash
git clone https://github.com/waynechu1109/FINS.git
cd FINS
```

### Conda
```bash
cd ~/FINS
pip install -r requirements.txt
```

### Docker
```bash
# pull docker image from docker hub
sudo docker pull waynechu1109/droplab_research:latest

# run docker  
docker run -it --gpus all \
  -p 8000:8000 \
  -e DISPLAY=$DISPLAY \
  -v /tmp/.X11-unix:/tmp/.X11-unix \
  -v /home/waynechu/FINS:/FINS \
  -v /etc/passwd:/etc/passwd:ro \
  -v /etc/group:/etc/group:ro \
  --name FINS \
  waynechu1109/droplab_research:latest /bin/bash

cd FINS
pip install -r requirements.txt
```

### Dataset Preparation
- DTU Training dataset. Please download the preprocessed DTU dataset provided by [MVSNet](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view).


### Image Preprocess
First clone VGGT
```bash
# clone VGGT for preprocess data
mkdir deps && cd deps
git clone https://github.com/facebookresearch/vggt.git
cd ..
```

The image should be placed in ```/data```. 

```bash
cd tools

# vggt preprocess
python3 vggt_pointcloud_generate.py --file dtu_118_60 --thres 65 --max_points 90000
```
- To tune the confidence threshold in percentage, set ```--thres```.
- When the scene is concave, set ```--concave true```. The direction of point clouds' normals are important for the training.
- When the computing resource is limited, set ```--max_points```. The default value is 200,000. You can also tune higher if higher mesh quality is needed. 

For more options, see ```python3 vggt_pointcloud_generate.py -h```. 

After preprocess, you can find the preprocessed point cloud file in ```/data/vggt_preprocessed/<file_name>```. It is convenient to view preprocessed point clouds with F3D. You can simply install it with ```sudo apt install f3d```.

### Training and Inferring
The script for the whole pipeline can be found in ```/scripts/experiment.sh```, which include the commands for both training and inferring. If you want to run series trainging (for example, multiple scenes at a single run), see ```/scripts/run_exp_series.sh```.

```bash
# Start series training
./scripts/run_exp_series.sh
```

<!-- ### Results
The mesh render script can also be found in ```/tools``` folder. You can render the result mesh with the commands below:
```bash
cd tools
python3 mesh_video_render.py --input <path_to_output_mesh>
```
- To show color, set ```--colored```. -->

<!-- You can download the [DTU results](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/EpvCB9YC1FZEtrsrbEkd8AwBGdnymfTQLJIdXFIeIOcqsw?e=3hb9Zn) and [BMVS results](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/EpLOwBek671NmgzmmLresT0Bt9JKgIYBkHogeQsukzfttQ?e=rodRih) of the paper reports here. -->

<!-- ## Evaluation
The output iso-surface reconstruction result can be evaluated with ground truth mesh with the command below:
```bash
cd tools
python3 chamfer_dist_eval.py --result_mesh <path_to_output_mesh> --gt_mesh <path_to_gt_mesh>
``` -->

## Citation

If you find this repository useful, please cite our arXiv paper:

```bibtex
@article{chu2025efficient,
  title   = {Efficient Construction of Implicit Surface Models From a Single Image for Motion Generation},
  author  = {Wei-Teng Chu and Tianyi Zhang and Matthew Johnson-Roberson and Weiming Zhi},
  journal = {arXiv preprint arXiv:2509.20681},
  year    = {2025}
}
```

