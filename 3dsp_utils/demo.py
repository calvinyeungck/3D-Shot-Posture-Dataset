import os
import torch
import pdb

def bbox_tracklet(args):
    from bot_sort.tools.shot_post import sn_demo, make_parser
    args_track = make_parser().parse_args()
    args_track.ablation = False
    args_track.mot20 = not args_track.fuse_score
    args.save_result=args.save_image
    os.makedirs(args.root+"/tracking_result", exist_ok=True)
    file=args.traget_clip.split("/")[-1]
    sn_demo(args.traget_clip,args.yolov8_param,args.root+"/tracking_result",file, args_track,args)

def select_tracklets(args):
    import pandas as pd
    import numpy as np
    import cv2
    import warnings
    from tracklet_selection.CNN import CNN

    warnings.filterwarnings("ignore", message="The frame.append method is deprecated")
    # read the tracklet
    tracklet_df=pd.read_csv(args.root+"/tracking_result/"+args.traget_clip.split("/")[-1].split(".")[0]+".txt")
    tracklet_df.columns = ['frame', 'id', 't', 'l', 'w', 'h', 'score', 'a', 'b', 'c']
    #drop tracklet with less than 15 frames
    tracklet_df = tracklet_df.groupby('id').filter(lambda x: len(x) >= 15)
    #copy from nearest frame if the frame is less than 20
    for traklet_id in tracklet_df['id'].unique():
        traklet = tracklet_df[tracklet_df['id'] == traklet_id]
        if len(traklet) < 20:
            for k in range(1,21):
                if k not in traklet['frame'].values:
                    nearest_frame = traklet['frame'].iloc[np.argmin(np.abs(traklet['frame']-k))]
                    nearest_bbox = traklet[traklet['frame']==nearest_frame].copy()
                    nearest_bbox['frame'] = k
                    #append the nearest bbox to the tracklet_df
                    tracklet_df = tracklet_df.append(nearest_bbox, ignore_index=True)
    image_dict = {}
    for traklet_id in tracklet_df['id'].unique():
        image_dict[traklet_id] = []
        traklet = tracklet_df[tracklet_df['id'] == traklet_id]
        if len(traklet) != 20:
            print(f"traklet {traklet_id} has {len(traklet)} frames")
            pdb.set_trace()
    #sort the tracklet_df by id then frame
    tracklet_df = tracklet_df.sort_values(by=['id', 'frame'])
    #read video
    cap = cv2.VideoCapture(args.traget_clip)
    fps = 25
    image_width = 96
    image_height = 96
    for j in range(fps):
        ret, frame = cap.read()
        if ret:
            if j<=14 and j>=9:
                frame_height, frame_width, _ = frame.shape
                frame_id = j+1
                frame_bbox = tracklet_df[tracklet_df['frame']==frame_id]
                #draw bbox
                for index, row in frame_bbox.iterrows():
                    flag = ""
                    #preprocess the bbox such it has the shape of (160,160)
                    bbox_width = row['w']
                    bbox_height = row['h']
                    bbox_left = row['l']
                    bbox_top = row['t']
                    #get the center of the bbox
                    center_x = int(bbox_top + bbox_width/2)
                    center_y = int(bbox_left + bbox_height/2)
                    #get the top left corner of the bbox
                    bbox_top = int(center_x - image_width/2)
                    bbox_left = int(center_y - image_height/2)
                    #check if the bbox is out of the frame
                    if bbox_top < 0:
                        bbox_top = 0
                        flag="top_left"
                    if bbox_left < 0:
                        bbox_left = 0
                        flag="top_left"
                    #cut the bbox
                    height = min(image_height, abs(frame_height - bbox_left))
                    width = min(image_width, abs(frame_width - bbox_top))
                    if height < image_height or width < image_width:
                        flag="bottom_right" 
                    bbox = frame[int(bbox_left):int(bbox_left+height), int(bbox_top):int(bbox_top+width)]

                    # pad the bbox if the bbox is smaller than (96,96) according to the flag
                    pad_height = max(0, image_height - height)
                    pad_width = max(0, image_width - width)
                    if flag == "top_left":
                        bbox = np.pad(bbox, ((pad_height, 0), (pad_width, 0), (0, 0)), 'constant', constant_values=0)
                    elif flag == "bottom_right":
                        bbox = np.pad(bbox, ((0, pad_height), (0, pad_width), (0, 0)), 'constant', constant_values=0)
                    # save the bbox
                    if args.save_image:
                        out_path = os.path.join(args.root+"/bbox_image", args.traget_clip.split("/")[-1].split(".")[0]+"_"+str(int(row["id"])))
                        os.makedirs(out_path, exist_ok=True)
                        cv2.imwrite(os.path.join(out_path, f"{frame_id}.jpg"), bbox)
                    #convert the bbox to np array
                    bbox = np.array(bbox)
                    image_dict[row['id']].append(bbox)
                    
        else:
            print("no frame")
            pdb.set_trace()

    cap.release()
    cv2.destroyAllWindows()
    #create the tensor from the image_dict
    import torch
    batch = []
    for k,v in image_dict.items():
        batch.append(np.array(v))
    batch = np.array(batch)
    batch = torch.tensor(batch)
    #change the channel from last to before the x and y
    batch = batch.permute(0, 1, 4, 2, 3)
    #preprocess the batch
    batch= batch/255
    batch = batch.float()


    #set the device
    
    input_channels = 3                              # Adjust based on your input image channels
    hidden_size = 256                               # Hidden size of LSTM (fintune) [128,256,512] 256
    num_layers = 2                                  # Number of LSTM layers (fintune) [1,2,3] 2 (not used)
    num_linear_layers = 3                           # Number of linear layers (fintune) [1,2,3] 3
    drop_out = 0.2                                  # Dropout rate
    kernel_size = 3                                 # Size of the convolutional kernel
    num_image = 6                                   # Number of images in the sequence

    model = CNN(input_channels, hidden_size, num_layers,num_linear_layers, kernel_size,num_image,drop_out).to(args.device)
    model.load_state_dict(torch.load(os.getcwd()+"/tracklet_selection/params/best.pth"))

    with torch.no_grad():
        batch = batch.to(args.device)
        output=model(batch)
        if torch.cuda.is_available():
            output=output.cpu().numpy()
        else:
            output=output.numpy()

    #convert the output to the list
    output = output.tolist()
    tracklet_id = list(image_dict.keys())
    max_index =output.index(max(output))
    selected_tracklet = tracklet_id[max_index]

    tracklet_df=tracklet_df[tracklet_df['id']==selected_tracklet]
    tracklet_img = []
    cap = cv2.VideoCapture(args.traget_clip)
    fps = 25
    image_width = 100
    image_height = 100
    for j in range(fps):
        ret, frame = cap.read()
        if ret:
            if j<20 :
                frame_height, frame_width, _ = frame.shape
                frame_id = j+1
                frame_bbox = tracklet_df[tracklet_df['frame']==frame_id]
                #draw bbox
                for index, row in frame_bbox.iterrows():
                    flag = ""
                    #preprocess the bbox such it has the shape of (160,160)
                    bbox_width = row['w']
                    bbox_height = row['h']
                    bbox_left = row['l']
                    bbox_top = row['t']
                    #get the center of the bbox
                    center_x = int(bbox_top + bbox_width/2)
                    center_y = int(bbox_left + bbox_height/2)
                    #get the top left corner of the bbox
                    bbox_top = int(center_x - image_width/2)
                    bbox_left = int(center_y - image_height/2)
                    #check if the bbox is out of the frame
                    if bbox_top < 0:
                        bbox_top = 0
                        flag="top_left"
                    if bbox_left < 0:
                        bbox_left = 0
                        flag="top_left"
                    #cut the bbox
                    height = min(image_height, abs(frame_height - bbox_left))
                    width = min(image_width, abs(frame_width - bbox_top))
                    if height < image_height or width < image_width:
                        flag="bottom_right" 
                    bbox = frame[int(bbox_left):int(bbox_left+height), int(bbox_top):int(bbox_top+width)]

                    # pad the bbox if the bbox is smaller than (96,96) according to the flag
                    pad_height = max(0, image_height - height)
                    pad_width = max(0, image_width - width)
                    if flag == "top_left":
                        bbox = np.pad(bbox, ((pad_height, 0), (pad_width, 0), (0, 0)), 'constant', constant_values=0)
                    elif flag == "bottom_right":
                        bbox = np.pad(bbox, ((0, pad_height), (0, pad_width), (0, 0)), 'constant', constant_values=0)
                    # save the bbox
                    out_path = os.path.join(args.root,"shooter_image")
                    os.makedirs(out_path, exist_ok=True)
                    cv2.imwrite(os.path.join(out_path, f"{frame_id}.jpg"), bbox)
                    #convert the bbox to np array
                    bbox = np.array(bbox)
                    # image_dict[row['id']].append(bbox)
                    tracklet_img.append(bbox)

        else:
            print("no frame")
            pdb.set_trace()

    cap.release()
    cv2.destroyAllWindows()
    return tracklet_img

def gen_2d_pose(shooter_img, args):
    from rtmlib import Body
    from tqdm import tqdm
    import numpy as np
    device = 'cpu'
    backend = 'onnxruntime'
    openpose_skeleton = True
    body = Body(det='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/yolox_x_8xb8-300e_humanart-a39d44ed.zip',
            det_input_size=(640, 640),
            pose='https://download.openmmlab.com/mmpose/v1/projects/rtmposev1/onnx_sdk/rtmpose-x_simcc-body7_pt-body7_700e-384x288-71d7b7e9_20230629.zip',
            pose_input_size=(288, 384),
            backend=backend,
            device=device)
    
    keypoints_list = []
    print("start to estimate 2d pose")
    for img in tqdm(shooter_img):
        keypoints, _ = body(img)
        #process the keypoints for the 3d pose estimation
        if keypoints.shape[0]!=1:
            best_distance=100000
            best_keypoints=0
            for k in range(keypoints.shape[0]):
                pose_i=keypoints[k]
                # #calculate the center of hips
                center_hips = (pose_i[11]+pose_i[12])/2
                # #calculate the center of shoulder
                center_shoulder = (pose_i[5]+pose_i[6])/2
                # calculate the center of body
                center_body = (center_hips+center_shoulder)/2
                #calculate the distance between the center of body and the center of the image (50,50)
                distance=np.linalg.norm(center_body-np.array([50,50]))
                if distance<best_distance:
                    best_distance=distance
                    best_keypoints=k
            keypoints=keypoints[best_keypoints]
        else:
            keypoints = keypoints[0]

        #add 1 to the keypoints (x,y,1)
        ones_column = np.ones((17, 1), dtype=keypoints.dtype)
        keypoints = np.concatenate([keypoints, ones_column], axis=1)
        # #calculate the center of hips
        center_hips = (keypoints[11]+keypoints[12])/2
        # #calculate the center of shoulder
        center_shoulder = (keypoints[5]+keypoints[6])/2
        # calculate the center of body
        center_body = (center_hips+center_shoulder)/2
        # calculate the neck
        neck = (keypoints[0]+center_shoulder)/2
        # #reorder the pose to match the desire order (ch),12,14,16,13,15,17,(cb),(cs),(n),1,7,9,11,6,8,10
        desired_order=[center_hips, keypoints[11],keypoints[13],keypoints[15],keypoints[12],keypoints[14],keypoints[16],center_body,center_shoulder,neck,keypoints[0],keypoints[6],keypoints[8],keypoints[10],keypoints[5],keypoints[7],keypoints[9]]
        keypoints = np.array(desired_order)

        keypoints_list.append(keypoints)
    
    keypoints_list = np.array(keypoints_list)
    #save the keypoints as npz
    os.makedirs(args.root+"/input_2D", exist_ok=True)
    np.savez(args.root+"/input_2D/keypoints.npz", reconstruction=keypoints_list)

def gen_3d_pose(args):
    from MotionAGFormer.demo.vis_sn import get_pose3D_demo,img2video_demo
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    args.video_length=args.num_frame
    get_pose3D_demo(args.root+"/shooter_image",args.root,args.video_length)
    img2video_demo(args.root+"/shooter_image", args.root,args.traget_clip.split("/")[-1].split(".")[0])

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--root','-p', type=str, default=os.getcwd()+"/output", help="root path to save the dataset")
    parser.add_argument('--yolov8_param', type=str, default=os.getcwd()+"/bot_sort/yolov8_player/best.pt", help='path to the yolov8 parameter')
    parser.add_argument('--save_image', action='store_true', default=False, help='Flag to save the image. If present, the value will be True.')
    parser.add_argument('--num_frame', type=int, default=20, help='number of frames to track')
    parser.add_argument("--device", type=str, default=torch.device('cuda' if torch.cuda.is_available() else 'cpu'), help="device to be used")
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    #clips to be considered
    #we use the image in 3dsp dataset test/00001 as an example
    parser.add_argument("-t","--traget_clip", type=str, default=os.getcwd()+"/example/test_00003.mp4", help="target clip to be considered")
    args = parser.parse_args()
    args.root = os.path.join(args.root, args.traget_clip.split("/")[-1].split(".")[0])

    #get traklets w/ yolov8+botsort
    bbox_tracklet(args)

    #select tracklets w/ cnn
    shooter_img = select_tracklets(args)

    #estimate 2d pose w/ rtmpose
    gen_2d_pose(shooter_img, args)

    #estimate 3d pose w/  motionagformer
    gen_3d_pose(args)
    