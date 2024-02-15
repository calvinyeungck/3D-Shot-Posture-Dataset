'''
3d shot posture annotation
## This script is used to get the 3d shot posture from the SoccerNet dataset
1. get soccernet video
2. get the action annotation
3. get the bbox and tracklet
4. identify the shooter tracklet   
5. crop the tracklet bbox  for 2d pose estimation
6. get the 2d pose and lift the 2d pose to 3d pose
7. structuring the data 
'''

import os
import numpy as np
import pdb

def download_soccernet_video(args):
    import SoccerNet
    from SoccerNet.Downloader import SoccerNetDownloader
    dataset_path=args.root+"/SoccerNet"
    if not os.path.exists(dataset_path):
        os.makedirs(dataset_path)

    mySoccerNetDownloader=SoccerNetDownloader(LocalDirectory=dataset_path)
    mySoccerNetDownloader.downloadGames(files=["Labels-v2.json"], split=["train","valid","test"])
    mySoccerNetDownloader.downloadGames(files=["1_720p.mkv", "2_720p.mkv"], split=["train","valid","test"])

def get_clips(root,dirs,files,output_path,second_before,second_after,target,count=1):
    from moviepy.editor import VideoFileClip
    from PIL import Image
    import json
    video_count={}
    #check and connect the video and annotation in SoccerNet
    if len(files)==3:
        video1=os.path.join(root,"1_720p.mkv")
        video2=os.path.join(root,"2_720p.mkv")
        label=os.path.join(root,"Labels-v2.json")
        output_folder=output_path

        #check if video exist
        if not os.path.exists(video1):
            print("video1 not exist", video1)
            exit()
        if not os.path.exists(video2):
            print("video2 not exist", video2)
            exit()
        if not os.path.exists(label):
            print("label not exist", label)
            exit()

        if not os.path.exists(output_folder):
            os.makedirs(output_folder)

        #read label
        with open(label, 'r') as file:
            data = json.load(file)

        #filter the target action
        Target=target #'Direct free-kick' could be consider in the future
        timestamps_dict_1={}
        timestamps_dict_2={}
        i=0
        j=0
        game=root.split('/')[-1]
        #get the annotation in the first half and second half
        for annotation in data['annotations']:
            if annotation["label"] in Target:
                annotation["game"]=game
                annotation["half"]=annotation["gameTime"].split(' - ')[0]
                annotation["time"]=annotation["gameTime"].split(' - ')[1]
                if annotation["half"]=="1":
                    timestamps_dict_1[i]=annotation
                    i+=1
                elif annotation["half"]=="2":
                    timestamps_dict_2[j]=annotation
                    j+=1

        # print("total number of actions: ", len(timestamps_dict_1)+len(timestamps_dict_2))

        clip1 = VideoFileClip(video1)
        clip2 = VideoFileClip(video2)

        #save the required clip from video at timestamps
        for key, value in timestamps_dict_1.items():
            position_ms = int(value["position"])
            start_time = max(0, position_ms - second_before*1000)
            end_time = position_ms + second_after*1000
            clip = clip1.subclip(start_time / 1000, end_time / 1000)
            # add the position_ms to each frame
            start_position_ms = start_time
            end_position_ms = end_time

            fps = clip.fps
            step_ms=(end_position_ms-start_position_ms)/fps
            position_ms-step_ms
            start_position_ms+step_ms*12.5

            save_path = os.path.join(output_folder, f'{count}.mp4')
            clip.write_videofile(save_path, audio=False)
            video_count[count]=value
            count+=1

        for key, value in timestamps_dict_2.items():
            position_ms = int(value["position"])
            start_time = max(0, position_ms - second_before*1000)
            end_time = position_ms + second_after*1000
            clip = clip2.subclip(start_time / 1000, end_time / 1000)
            save_path = os.path.join(output_folder, f'{count}.mp4')
            clip.write_videofile(save_path, audio=False)
            video_count[count]=value
            count+=1
        # pdb.set_trace()
    return count,video_count

def action_annotation(args):
    if not os.path.exists(args.output+"/clips"):
        os.makedirs(args.output+"/clips")
        count=1
        exist_df = None
    else:
        count=len(os.listdir(args.output+"/clips"))+1
        if os.path.exists(os.path.join(args.output, 'clips_info.csv')):
            exist_df = pd.read_csv(os.path.join(args.output, 'clips_info.csv'))
    exist_count=count
    video_info={}
    # cut the video into clips
    for root, dirs, files in os.walk(args.root+args.league): # only england_epl is considered in this example
        count,video_info_i=get_clips(root,dirs,files,args.output+"/clips",args.second_before,args.second_after,args.target,count)
        #merge the video_info_i to video_info
        video_info.update(video_info_i)

    #save the clips_info to csv
    df = pd.DataFrame.from_dict(video_info, orient='index')
    df.insert(0, 'id', range(exist_count, len(df)+exist_count))
    #add an empty column "shooter_id" as the last column
    df['shooter_id'] = np.nan
    if exist_df is not None:
        df = pd.concat([exist_df, df])
    df.to_csv(os.path.join(args.output, 'clips_info.csv'), index=False)

def bbox_tracklet(args):
    from bot_sort.tools.shot_post import sn_demo, make_parser
    args_track = make_parser().parse_args()
    args_track.ablation = False
    args_track.mot20 = not args_track.fuse_score
    args.save_result=args.save_image
    os.makedirs(args.root+"/tracking", exist_ok=True)
    
    file_path=args.root+"/clips"
    file_list=os.listdir(file_path)
    for file in file_list:
        sn_demo(os.path.join(file_path,file),args.yolov8_param,args.root+"/tracking",file, args_track,args)

def crop_bbox(args):
    import cv2
    info_path = os.path.join(args.output, 'clips_info.csv')
    video_folder_path = args.output+"/clips" 
    tracklet_path = args.root+"/tracking"
    output_path = args.root+"/cropped_images"
    if not os.path.exists(output_path):
        os.makedirs(output_path) 

    info = pd.read_csv(info_path)
    #remove the rows with empty shooter_id
    info = info.dropna(subset=['shooter_id'])
    
    #cut the video with the shooter bbox
    print("cropping the following clips with the shooter bbox :")
    for index, row in info.iterrows():
        print(row['id'])
        bbox = pd.read_csv(os.path.join(tracklet_path, str(int(row['id']))+'.txt'), sep=',', header=None)
        bbox.columns = ['frame', 'id', 't', 'l', 'w', 'h', 'score', 'a', 'b', 'c']    #set col name
        shooter_bbox = bbox[bbox['id']==int(row['shooter_id'])] #get the tracklet of the shooter with the shooter id
        shooter_bbox = shooter_bbox.sort_values(by=['frame']) #sort by frame
        shooter_bbox = shooter_bbox.set_index('frame', drop=False) #set the index as frame with out droping the frame col

        if len(shooter_bbox) < args.num_frame:
            #replace the frame bbox with the nearest frame bbox
            for i in range(1,args.num_frame+1):
                if i not in shooter_bbox['frame'].values:
                    nearest_frame = shooter_bbox['frame'].iloc[np.argmin(np.abs(shooter_bbox['frame']-i))] #find the nearest frame
                    nearest_bbox = shooter_bbox[shooter_bbox['frame']==nearest_frame].copy() #get the bbox of the nearest frame
                    nearest_bbox['frame'] = i #change the frame number
                    shooter_bbox.loc[i] = nearest_bbox.values[0] #replace the frame bbox with the nearest frame bbox

        #sort the shooter bbox by the frist col
        shooter_bbox = shooter_bbox.sort_index()
        #save the bbox to the txt
        shooter_bbox.to_csv(os.path.join(output_path, str(int(row['id']))+'.txt'), index=False, header=False, sep=',')

        video_path=os.path.join(video_folder_path, str(int(row['id']))+'.mp4')
        #check if the video exists
        if not os.path.exists(video_path):
            print(video_path)
            print(f"video with id {row['id']} not exists")
            continue
        cap = cv2.VideoCapture(video_path)
        frame_rate=25
        max_width = shooter_bbox['w'].max()
        max_width = max(max_width,100) #round up to int
        max_width = int(np.ceil(max_width))
        max_height = shooter_bbox['h'].max()
        max_height = max(max_height,100)
        max_height = int(np.ceil(max_height)) #round up to int
        # cut the video with the shooter bbox and save as mp4
        if cap.isOpened():

            video_width = max_width
            video_height = max_height
            frame_count = 1  

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if frame_count > 20:
                    break
                
                # Assuming shooter_bbox contains the shooter's bounding box coordinates
                # Extract coordinates
                shooter_coords = shooter_bbox[['l', 't', 'w', 'h']].iloc[frame_count-1].values
                left, top, width, height = shooter_coords
                #Enlarge the bbox to fit the video size
                #get the center of the bbox
                center_y = left + height/2
                center_x = top + width/2
                #get the new bbox
                top= center_x - video_width/2
                left = center_y - video_height/2
                width = video_width
                height = video_height
                #convert to int
                left = int(left)
                top = int(top)
                width = int(width)
                height = int(height)
                #check if the bbox is out of the frame
                if left < 0:
                    left = 0
                if top < 0:
                    top = 0

                # Crop the frame based on shooter's bbox
                cropped_frame = frame[left:left+height, top:top+width]
                #pad the cropped frame to the width and height and put it in the center
                # Get the dimensions of the cropped frame
                cropped_height, cropped_width, _ = cropped_frame.shape

                # Calculate the padding needed for width and height
                pad_width = max(0, (video_width - cropped_width) // 2)
                pad_height = max(0, (video_height - cropped_height) // 2)

                # Calculate remainders for odd dimensions
                remainder_width = (video_width - cropped_width) % 2
                remainder_height = (video_height - cropped_height) % 2

                # Calculate border widths for each side considering remainders
                top_pad = pad_height + remainder_height
                bottom_pad = pad_height
                left_pad = pad_width + remainder_width
                right_pad = pad_width

                # Create a constant border around the cropped frame to achieve desired dimensions
                padded_frame = cv2.copyMakeBorder(cropped_frame, top_pad, bottom_pad, left_pad, right_pad, borderType=cv2.BORDER_CONSTANT, value=(0, 0, 0))

                #save the cropped frame
                image_output_path = os.path.join(output_path, str(int(row['id'])),f"{frame_count}"+'.jpg')
                if not os.path.exists(os.path.join(output_path, str(int(row['id'])))):
                    os.makedirs(os.path.join(output_path, str(int(row['id']))))
                cv2.imwrite(image_output_path, padded_frame)
                
                frame_count += 1

            # Release everything when done
            cap.release()

def get_pose(args):
    from MotionAGFormer.demo.vis_sn import get_pose2D,get_pose3D
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    all_file_path=args.root+"/cropped_images"
    output_path=args.root+"/posture"
    if not os.path.exists(output_path):
        os.makedirs(output_path)
    folder_list = os.listdir(all_file_path)
    #drop the folder that ends with .txt
    folder_list = [folder_i for folder_i in folder_list if not folder_i.endswith('.txt')]
    for folder_i in folder_list:
        folder_path= os.path.join(all_file_path,folder_i)
        shooter_tracklet_path_i=os.path.join(all_file_path,folder_i+".txt")
        get_pose2D(args,folder_path, os.path.join(output_path,folder_i),shooter_tracklet_path_i)
        get_pose3D(folder_path,os.path.join(output_path,folder_i),args.video_length)

def structuring(args):
    import json
    from PIL import Image
    from tqdm import tqdm
    from collections import OrderedDict
    import configparser
    import pandas as pd

    
    #params
    dataset_name=args.dataset_name

    #paths
    save_path = args.root
    image_path = args.root+"/cropped_images"
    pose_path=args.root+"/posture"
    info_csv_path=args.root+"/clips_info.csv"
    tracklet_path=args.root+'/tracking'
    tracklet_correction_path=args.root+'/cropped_images'

    #create dataset folder
    save_path = os.path.join(save_path, dataset_name)
    if not os.path.exists(save_path):
        os.makedirs(save_path)
        start_num = 0
    else:
        #get the folders in the dataset path
        folders = os.listdir(save_path)
        #convert the string to int
        folders = [int(folder) for folder in folders]
        #get the max folder number
        start_num = max(folders)


    #get the folders in the img path
    folders = os.listdir(image_path)
    #drop the folder that ends with .txt
    folders = [folder for folder in folders if not folder.endswith('.txt')]
    #convert the string to int
    folders = [int(folder) for folder in folders]
    folders.sort()

    #read the info csv
    info_csv = pd.read_csv(info_csv_path)

    for i in tqdm(range(len(folders))):
        folder = str(folders[i])
        folder_path = os.path.join(image_path, folder)

        #create the img folder
        save_img_path = os.path.join(save_path, str(i+1+start_num).zfill(5),"img")
        if not os.path.exists(save_img_path):
            os.makedirs(save_img_path)

        #move the images to the dataset path
        imgs = os.listdir(folder_path)
        for img in imgs:
            img_path = os.path.join(folder_path, img)
            save_img_path = os.path.join(save_path, str(i+1+start_num).zfill(5),"img",img.split(".")[0].zfill(3)+".jpg")
            # copy the image to the dataset path
            os.system('cp {} {}'.format(img_path, save_img_path))

        #create the pose folder
        save_pose_path = os.path.join(save_path, str(i+1+start_num).zfill(5) ,"posture")
        if not os.path.exists(save_pose_path):
            os.makedirs(save_pose_path)

        #read the tracklet txt
        tracklet=pd.read_csv(os.path.join(tracklet_path,f"{folder}.txt"), sep=",", header=None)
        tracklet_correction=pd.read_csv(os.path.join(tracklet_correction_path,f"{folder}.txt"), sep=",", header=None)
        #set column names for tracklet [frame, id, x, y, w, h, score, c1, c2, c3]
        tracklet.columns = ["frame", "id", "x", "y", "w", "h", "score", "c1", "c2", "c3"]
        tracklet_correction.columns = ["frame", "id", "x", "y", "w", "h", "score", "c1", "c2", "c3"]
        #get the tracklet id in trcaklet_correction
        tracklet_correction_id = tracklet_correction["id"].unique()
        #remove the tracklet id in tracklet
        tracklet = tracklet[~tracklet["id"].isin(tracklet_correction_id)]
        #concatenate the tracklet and tracklet_correction
        tracklet = pd.concat([tracklet, tracklet_correction], ignore_index=True)
        #drop all frame after 20
        tracklet = tracklet[tracklet["frame"]<=20]
        #sort the tracklet by id and frame
        tracklet = tracklet.sort_values(by=['id', 'frame'], ignore_index=True)
        tracklet_output_path = os.path.join(save_path, str(i+1+start_num).zfill(5),"gt") 
        if not os.path.exists(tracklet_output_path):
            os.makedirs(tracklet_output_path)
        tracklet.to_csv(os.path.join(tracklet_output_path,"gt.txt") , header=False, index=False, sep=",")

        body_parts = [
                "Center of Hips",
                "Left Hip",
                "Left Knee",
                "Left Ankle",
                "Right Hip",
                "Right Knee",
                "Right Ankle",
                "Center of Body",
                "Center of Shoulder",
                "Neck",
                "Head",
                "Right Shoulder",
                "Right Elbow",
                "Right Wrist",
                "Left Shoulder",
                "Left Elbow",
                "Left Wrist"
            ]

        #read the npz for 3d and 2d pose
        pose_3d_path = os.path.join(pose_path, folder,"3d_keypoints.npz")
        pose_3d = np.load(pose_3d_path, allow_pickle=True)['reconstruction']
        pose_2d_path = os.path.join(pose_path, folder,"2d_keypoints.npz")
        pose_2d = np.load(pose_2d_path, allow_pickle=True)['reconstruction']


        
        for frame in range(20):
            pose_dict={}
            pose_dict["image_path"] = f"{str(i+1+start_num).zfill(5)}/img/{frame+1}.jpg"
            pose_dict["shooter_tracklet_id"] = int(info_csv[info_csv["id"]==int(folder)].shooter_id.iloc[0])
            pose_dict["bbox"] = tracklet[(tracklet["frame"] == frame + 1) & (tracklet["id"] == pose_dict["shooter_tracklet_id"])][["x", "y", "w", "h"]].to_dict(orient='records')[0]
            #pad the bbox
            left, top, width, height = pose_dict["bbox"]["y"], pose_dict["bbox"]["x"], pose_dict["bbox"]["w"], pose_dict["bbox"]["h"]
            #get the center of the bbox
            center_y = left + height/2
            center_x = top + width/2
            #get the new bbox
            top= center_x - max(100,width)/2
            left = center_y - max(100,height)/2
            width = max(100,width)
            height = max(100,height)
            #convert to int
            left = int(left)
            top = int(top)
            width = int(width)
            height = int(height)
            if left < 0:
                left = 0
            if top < 0:
                top = 0
            
            #update the bbox
            pose_dict["bbox"]["x"] = left
            pose_dict["bbox"]["y"] = top
            pose_dict["bbox"]["w"] = width
            pose_dict["bbox"]["h"] = height

            pose_dict["keypont_2d"] = {}
            for count, joint in enumerate(pose_3d[frame]):
                pose_dict["keypont_2d"][str(count)] = {"name":body_parts[count],"x": float(joint[0]), "y": float(joint[1])}
            pose_dict["keypont_3d"] = {}
            for count, joint in enumerate(pose_3d[frame]):
                pose_dict["keypont_3d"][str(count)] = {"name":body_parts[count],"x": float(joint[0]), "y": float(joint[1]), "z": float(joint[2])}

            #save the json
            save_pose_json_path = os.path.join(save_path, str(i+1+start_num).zfill(5),"posture",f"{str(frame+1).zfill(3)}.json")
            json_string = json.dumps(pose_dict, indent=4)
            # with open(save_pose_json_path, 'w') as outfile:
            #     json.dump(pose_dict, outfile)
            with open(save_pose_json_path, "w") as json_file:
                json_file.write(json_string)

        #write the info.ini
        info_dict = configparser.ConfigParser()
        info_dict["info"]={}
        info_dict["info"]["id"] = str(i+1+start_num).zfill(5)
        info_dict["info"]["previous_id"] = folder
        info_dict["info"]["gameTime"] = info_csv[info_csv["id"]==int(folder)].gameTime.values[0]
        info_dict["info"]["label"] = info_csv[info_csv["id"]==int(folder)].label.values[0]
        info_dict["info"]["annotated_position"] = str(info_csv[info_csv["id"]==int(folder)].position.values[0])
        info_dict["info"]["start_position"] = str(int(int(info_dict["info"]["annotated_position"])-40*12.5))
        info_dict["info"]["end_position"] = str(int(int(info_dict["info"]["annotated_position"])+40*2.5))
        info_dict["info"]["position_step"] = "40"
        info_dict["info"]["team"] = info_csv[info_csv["id"]==int(folder)].team.values[0]
        info_dict["info"]["visibility"] = info_csv[info_csv["id"]==int(folder)].visibility.values[0]
        info_dict["info"]["game"] = info_csv[info_csv["id"]==int(folder)].game.values[0]
        info_dict["info"]["half"] = str(info_csv[info_csv["id"]==int(folder)].half.values[0])
        info_dict["info"]["time"] = info_csv[info_csv["id"]==int(folder)].time.values[0]

        save_info_path = os.path.join(save_path, str(i+1+start_num).zfill(5),"info.ini")
        with open(save_info_path, 'w') as configfile:
            info_dict.write(configfile)

if __name__ == "__main__":
    import argparse
    import pandas as pd

    parser = argparse.ArgumentParser(description='Download SoccerNet files')
    parser.add_argument('--root','-p', type=str, default=os.getcwd()+"/3dsp_addition", help="root path to save the dataset")
    parser.add_argument('--league','-l', type=str, default="/SoccerNet/england_epl", help="league in SoccerNet")
    #clips cutting parameters
    parser.add_argument('--second_before','-sb', type=float, default=0.5,help='how many seconds before the target frame')
    parser.add_argument('--second_after','-sa', type=float, default=0.5,help='how many seconds after the target frame')
    parser.add_argument('--target', '-t', nargs='+', type=int, default=['Goal','Shots off target','Shots on target'], help='target action')
    #tracking parameters
    parser.add_argument('--yolov8_param', type=str, default=os.getcwd()+"/bot_sort/yolov8_player/best.pt", help='path to the yolov8 parameter')
    parser.add_argument('--save_image', action='store_true', default=True, help='Flag to save the image. If present, the value will be True.')
    parser.add_argument('--num_frame', type=int, default=20, help='number of frames to track')
    #2d posture parameters
    parser.add_argument('--video', type=str, default='2.mp4', help='input video')
    parser.add_argument('--gpu', type=str, default='0', help='input video')
    parser.add_argument('--video_length', type=int, default=20, help='input video')
    #dataset parameters
    parser.add_argument('--dataset_name', type=str, default="3dsp/train", help='name of the dataset')
    args=parser.parse_args()

    args.yolov8_param= os.getcwd()+"/bot_sort/yolov8_player/best.pt"
    args.output=args.root

    #1.download the dataset from soccernet, auto skip if the dataset already exists
    # comment out the following line if the dataset already exists
    download_soccernet_video(args)

    #2.get the action annotation and cut the video into clips
    # the clips numbering will start from the last number in the clips folder 
    # carful of re-run the code, it will duplicate the clips
    action_annotation(args)

    #3.get the bbox and tracklet of all clips
    bbox_tracklet(args)

    #4.identify the shooter tracklet   
    # identify the tracklet images in folder "tracking" and save the shooter id to the clips_info.csv
    # if the shooter is not identified or the clip should be skipped, leave the shooter_id empty
    # a sample clips_info.csv is provided for the SoccerNet epl league
    
    #5.crop the tracklet bbox for 2d pose estimation
    # adjust the bbox in the folder "tracking" is needed or remove the shooter_id in the clips_info.csv to skip the clip
    # the shooter tracklet (interpolated) will also be saved in the folder "cropped_images"
    crop_bbox(args)

    #6.get the 2d and 3d posture with MotionAGFormer
    # replace the 2d or 3d posture annotation for more accurate results
    get_pose(args)

    #7.structuring the data
    structuring(args)




