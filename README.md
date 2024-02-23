# 3D Shot Posture Dataset
Todo
- add the requirement.txt for visualization of 2d and 3d annotation

This dataset consists of the 3d and 2d postures of professional football players under shot situations.

<p align="center"><img src="https://github.com/calvinyeungck/3D-Shot-Posture-Dataset/blob/master/3dsp_utils/image/00001.gif?raw=true&v=4" width="60%" alt="" /></p>

### Content of the dataset:
- In 3dsp/train 
    - 20 cropped image x 200 shot
    - Tracklet, 2d and 3d keypoints 
- In 3dsp/test
    - 20 cropped image x 10 shots
    - Tracklet

### [Clustering](https://github.com/calvinyeungck/3D-Shot-Posture-Dataset/tree/master/analysis):
We provide 2 clustering methods, with pca+kmean or deep learning+kmeam. For the result of the clustering, refer to our paper.
<div style="display:flex;">
  <img src="https://github.com/calvinyeungck/3D-Shot-Posture-Dataset/blob/master/3dsp_utils/image/elbow_method.png" alt="Image 1" style="width:45%;">
  <img src="https://github.com/calvinyeungck/3D-Shot-Posture-Dataset/blob/master/3dsp_utils/image/t-SNE.png" alt="Image 2" style="width:45%;">
</div>

### Automated 3D Shooter Posture Extraction:
Extracting the shooter 2D&3D Posture from broadcast video
Data source

Result

### Data collection method: 
The broadcast videos were collected from [SoccerNet](https://www.soccer-net.org/), and with annotation on actions, the videos were clipped (0.5 before and after the annotated ms, 25 frames total). The tracklet for the clips was then generated using a fine-tuned [YOLO v8](https://github.com/ultralytics/ultralytics) and [BoT-Sort](https://github.com/NirAharon/BoT-SORT). Furthermore, the shooter's traklet id was manually selected, and the cropped image was generated using the Bbox. The first 20 images' 2D postures (determined empirically) were manually annotated and lifted to 3D using the [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer). This repository includes the modified code for [BoT-Sort](https://github.com/NirAharon/BoT-SORT) and [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer).

The guide for downloading the dataset, the dataset structure explanation, and performing additional annotation and visualization are available below.
## Getting Started
### Download dataset
Step 1. Clone this repo
```
git clone https://github.com/calvinyeungck/3D-Shot-Posture-Dataset.git
```
Step 2. Locate the repo
```
cd path/to/this/repo/
```
Step 3. Unzip the file 
```
unzip 3dsp.zip
```
### Annotation and Visualization functions (Optional)
Step 4. Install required package
```
pip install -r requirements.txt
```
Step 5. Download the required models parameters

Download the parameters for the following models and place them accordingly.
- [MotionAGFormer_2d](https://drive.google.com/drive/folders/1_ENAMOsPM7FXmdYRbkwbFHgzQq_B_NQA?usp=sharing) at 3dsp_utils/MotionAGFormer/checkpoint/
- [MotionAGFormer_3d](https://drive.google.com/file/d/1Iii5EwsFFm9_9lKBUPfN8bV5LmfkNUMP/view) at 3dsp_utils/MotionAGFormer/demo/lib/checkpoint/
- [YOLOv8](https://drive.google.com/file/d/1zkFLB9VKK0axqq6WOwgth82BVBMglYjB/view?usp=sharing) at 3dsp_utils/bot_sort/yolov8_player/best.pt


Step 6. Perform additional annotation on [SoccerNet](https://www.soccer-net.org/)
```
cd path/to/this/repo/
cd 3dsp_utils
python dataset_annotation.py
```
Step 7. Visualize the 2d and 3d posture
```
cd path/to/this/repo/
cd 3dsp_utils
python vis_post.py
```
Step 8. Pre-process the 3d posture
```
cd path/to/this/repo/
cd 3dsp_utils
python pre_process.py
```

## Dataset Structure
The following shows the overall structure of the dataset and the format of each file.
```
xxxxx  --- img --- 001.jpg
            |   |- 002.jpg
            |   |- ...
            |   |- 020.jpg
            |   
            posture --- 001.json (H3WB format)
            |        |- 002.json
            |        |- ...
            |        |- 020.json
            |
            gt --- gt.txt (MOT20 format)
            |
            info --- info.ini
```
### [H3WB format](https://github.com/wholebody3d/wholebody3d)
- Removed the sample id and added the name of the joint in the original format
- Use tlwh (xywh) instead of tlbr (xyxy) in the original format (following SoccerNet)
- 2d posture has the xy coordinate for the cropped image
- 3d posture has the xyz coordinate from the output of [MotionAGFormer](https://github.com/TaatiTeam/MotionAGFormer)
```
xxx.json   --- 'image_path'
            |
            -- 'shooter_tracklet_id' --- 'id'
            |
            -- 'bbox' --- 't'
            |          |- 'l'
            |          |- 'w'
            |          |- 'h'
            |
            -- 'keypont_2d' --- joint id --- 'name'
            |                             |- 'x'
            |                             |- 'y'
            |
            -- 'keypont_3d' --- joint id --- 'name'
                                            |- 'x'
                                            |- 'y'
                                            |- 'z'                                                   
```

### [MOT20 format](https://github.com/SoccerNet/sn-tracking)
- frame ID, track ID, top left coordinate of the bounding box, top y coordinate, width, height, confidence score for the detection 
(always 1. for the ground truth) and the remaining values are set to -1 as they are not used


### info.ini
- info from SoccerNet
```
[info]
id = folder id 
previous_id = id previously given to the clips
gameTime = time of the game
label = shot on target or shot off target
annotated_position = the ms where the action is annotated
start_position = the ms where the first image is captured
end_position = the ms where the last image is captured
position_step = the ms between each image
team = home or away
visibility = visibility of the player
game = info of the game
half = first or second half
time = time of the video in seconds
```
## Reference
 Please consider citing our work if you find it helpful to yours:

```
TBA
```
