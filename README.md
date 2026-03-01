# Project name: ... [conferecne]
(README abstract)

![](./docs/images/teaser.jpg)

## [Project Page]() | [Paper]() 

## Setup
```bash
git clone https://...
cd ...
```

### Conda
```bash
cd ~/droplab_research
pip install -r requirements.txt
```

### Docker
```bash
# pull docker image from docker hub
sudo docker pull waynechu1109/droplab_research:a100_latest

# run docker  
sudo docker run --gpus all -it \
  -v ~/Wayne/home/waynechu/droplab_research:/root/droplab_research \
  waynechu1109/droplab_research:a100_latest \
  /bin/bash
```

### Dataset (should modify in the future)
- DTU Training dataset. Please download the preprocessed DTU dataset provided by [MVSNet](https://drive.google.com/file/d/1eDjh-_bxKKnEuz5h-HXS7EDJn59clx6V/view). As stated in the paper, we preprocess the images to obtain the masks about the "black empty background" to remove image noises. The preprocessed masks can be downloaded [here](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/xxlong_connect_hku_hk/EW_v7RA73HNEquScVtNJ34gB4hYlRfEatW4TOg086F0_Lg?e=3SKiif). Training without the masks will not be a problem, just ignore the "masks" in the dataloader.
- DTU testing dataset. Since our target neural reconstruction with sparse views, we select two set of three images from the 15 testing scenes (same as [IDR](https://github.com/lioryariv/idr)) for evaluation. Download our prepared [testing dataset](https://connecthkuhk-my.sharepoint.com/:u:/g/personal/xxlong_connect_hku_hk/EU22HEv48nRLnnnliRvJNA0BILozsMLbhsnMQh1WZLY5kg?e=Lh7kWM).


### Image Preprocess
The input image should be placed in ```/data```. 

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

### Training 
The training script can be found in ```/scripts/experiment.sh```, which include the commands for both training and inferencing. If you want to run series trainging (for example, multiple scenes at a single run), see ```/scripts/run_exp_series.sh```.

```bash
# Start series training
./scripts/run_exp_series.sh
```

### Results
The mesh render script can also be found in ```/tools``` folder. You can render the result mesh with the commands below:
```bash
cd tools
python3 mesh_video_render.py --input <path_to_output_mesh>
```
- To show color, set ```--colored```.

<!-- You can download the [DTU results](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/EpvCB9YC1FZEtrsrbEkd8AwBGdnymfTQLJIdXFIeIOcqsw?e=3hb9Zn) and [BMVS results](https://connecthkuhk-my.sharepoint.com/:f:/g/personal/xxlong_connect_hku_hk/EpLOwBek671NmgzmmLresT0Bt9JKgIYBkHogeQsukzfttQ?e=rodRih) of the paper reports here. -->

## Evaluation
The output iso-surface reconstruction result can be evaluated with ground truth mesh with the command below:
```bash
cd tools
python3 chamfer_dist_eval.py --result_mesh <path_to_output_mesh> --gt_mesh <path_to_gt_mesh>
```

## Citation

Cite as below if you find this repository is helpful to your project:

```
...
```

## Acknowledgement
The pipeline is plotted with the helpful tool provided by [PlotNeuralNet](https://github.com/HarisIqbal88/PlotNeuralNet) [https://doi.org/10.5281/zenodo.2526396](https://doi.org/10.5281/zenodo.2526396)

<!-- Some code snippets are borrowed from [IDR](https://github.com/lioryariv/idr), [NeuS](https://github.com/Totoro97/NeuS) and [IBRNet](https://github.com/googleinterns/IBRNet). Thanks for these great projects. -->