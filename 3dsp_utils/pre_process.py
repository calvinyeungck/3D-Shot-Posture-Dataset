import numpy as np
import os
import json
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from MotionAGFormer.demo.vis_sn import show3Dpose
from scipy.signal import savgol_filter
import cv2
import imageio
import pdb

def get_keypoints_3d(folder):
    posture_folder=os.path.join(args.root, folder, "posture")
    posture=os.listdir(posture_folder)
    keypoints_3d = []
    for i in range(len(posture)):
        json_path = os.path.join(posture_folder, f'{str(i+1).zfill(3)}.json')
        with open(json_path, 'r') as f:
            data = json.load(f)
            data_3d = data['keypont_3d']
            data_3d_list = []
            for k,v in data_3d.items():
                data_3d_list.append([v["x"],v["y"],v["z"]])
            keypoints_3d.append(data_3d_list)
            f.close()
    
    keypoints_3d = np.array(keypoints_3d)
    return keypoints_3d

def rotate_yaxis(angle, point):
    # Convert angle to radians
    theta = np.radians(angle)

    rotation_x = np.array([[1, 0, 0],
                           [0, np.cos(theta), -np.sin(theta)],
                           [0, np.sin(theta), np.cos(theta)]])
    
    rotation_y = np.array([[np.cos(theta), 0, np.sin(theta)],
                           [0, 1, 0],
                           [-np.sin(theta), 0, np.cos(theta)]])
    
    rotation_z = np.array([[np.cos(theta), -np.sin(theta), 0],
                           [np.sin(theta), np.cos(theta), 0],
                           [0, 0, 1]])

    # Define the rotation matrix for rotation around the y-axis
    rotation_matrix = rotation_z

    # Convert the input point to a column vector
    point_vector = np.array([[point[0]], [point[1]], [point[2]]])

    # Perform the rotation
    rotated_point_vector = np.dot(rotation_matrix, point_vector)
    # Convert the result back to a tuple
    rotated_point = rotated_point_vector.flatten().tolist()

    return rotated_point

def rotate_3d_keypoints(keypoints_3d):
    rotated_keypoints_3d = []
    for frame in keypoints_3d:
        rotation_dict = {}
        rotation_dict[0] = np.array(frame)
        for angle in range(10,370,10):
            rotated_keypoints_3d_frame = []
            for keypoint in frame:
                rotated_keypoints_3d_frame.append(rotate_yaxis(angle, keypoint))
            rotation_dict[angle] = rotated_keypoints_3d_frame

        # calculate the x,z coordinate distance between point 2 (Left Hip) and 5 (right hip)
        largest_distance = 0
        largest_angle = None
        for k,v in rotation_dict.items():
            distance = np.sqrt((v[11][0]-v[14][0])**2 + (v[11][2]-v[14][2])**2)
            if distance > largest_distance and v[11][0] > v[14][0]:
                largest_distance = distance
                largest_angle = k
                # print(largest_angle)
        #ensure the left side is always on the left
        # if rotation_dict[largest_angle][1][0] < rotation_dict[largest_angle][4][0]:
        #     rotated_keypoints_3d.append(rotation_dict[abs(largest_angle-180)])
        # else:
        #     rotated_keypoints_3d.append(rotation_dict[largest_angle]) 
        if largest_angle is not None:
            rotated_keypoints_3d.append(rotation_dict[largest_angle])
        else:
            pdb.set_trace()
    return np.array(rotated_keypoints_3d)

def rotate_3d_keypoints_testing(folder):
    keypoints_3d = get_keypoints_3d(folder)
    rotated_keypoints_3d = []
    for frame in keypoints_3d:
        rotation_dict = {}
        rotation_dict[0] = np.array(frame)
        for angle in range(10,370,10):
            rotated_keypoints_3d_frame = []
            for keypoint in frame:
                rotated_keypoints_3d_frame.append(rotate_yaxis(angle, keypoint))

            rotated_keypoints_3d.append(rotated_keypoints_3d_frame)   
        return np.array(rotated_keypoints_3d)

def plot_3d_keypoints(key_points, folder):    
    video_length = len(key_points)
    joints_left = [4, 5, 6, 11, 12, 13]
    joints_right = [1, 2, 3, 14, 15, 16]


    
    for j, post_out in enumerate(key_points):
        fig = plt.figure(figsize=(9.6, 5.4))
        gs = gridspec.GridSpec(1, 1)
        gs.update(wspace=-0.00, hspace=0.05) 
        ax = plt.subplot(gs[0], projection='3d')
        # pdb.set_trace()
        show3Dpose(post_out, ax)
        
        #update post_out_all
        # post_out_all[j]=post_out

        output_dir_3D = os.path.join(args.save_path,folder,'pose3D')
        os.makedirs(output_dir_3D, exist_ok=True)

        plt.savefig(os.path.join(output_dir_3D, str(('%03d'% (j+1))) + '.png'), dpi=200, format='png', bbox_inches='tight')
        plt.close(fig)

def rescale_key_point(stats,stats_id,fixed_pt,update_pt,relative_pt,frame):
    #rescale the distance between the keypoints fixed_pt and update_pt to stats[stats_id]
    initial_distance =stats[stats_id]
    original_distance = np.sqrt((frame[fixed_pt][0]-frame[update_pt][0])**2 + (frame[fixed_pt][1]-frame[update_pt][1])**2 + (frame[fixed_pt][2]-frame[update_pt][2])**2)
    scaling_factor = initial_distance/original_distance
    #update the position of point relative_pt
    relative_position = {}
    for i in relative_pt:
        relative_position[i] = frame[i] - frame[update_pt]
    #update the position of point update_pt
    frame[update_pt] = frame[fixed_pt]+(frame[update_pt]-frame[fixed_pt])*scaling_factor
    #update the position of point relative_pt
    for i in relative_pt:
        frame[i] = frame[update_pt]+relative_position[i]
    return frame

def get_stats(keypoints_3d):
    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    #get the stats of the distance between each pair of keypoints
    stats = {}
    for i in range(16):
        stats[i] = []
    for frame in keypoints_3d:
        for i in range(16):
            stats[i].append(np.sqrt((frame[I[i]][0]-frame[J[i]][0])**2 + (frame[I[i]][1]-frame[J[i]][1])**2 + (frame[I[i]][2]-frame[J[i]][2])**2))
    for i in range(16):
        stats[i] = np.array(stats[i])
        #calculate the trimmed mean of the distance
        mean = np.mean(stats[i])
        std = np.std(stats[i])
        #remove the outliers
        stats[i] = stats[i][np.where(np.abs(stats[i]-mean)<std)]
        #update the mean and std
        mean = np.mean(stats[i])
        stats[i] = [mean]
    return stats

def get_stats_testing(keypoints_3d):
    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])
    #get the stats of the distance between each pair of keypoints
    stats = {}
    for i in range(16):
        stats[i] = []
    for frame in keypoints_3d:
        for i in range(16):
            stats[i].append(np.sqrt((frame[I[i]][0]-frame[J[i]][0])**2 + (frame[I[i]][1]-frame[J[i]][1])**2 + (frame[I[i]][2]-frame[J[i]][2])**2))
    for i in range(16):
        stats[i] = np.array(stats[i])
        # #calculate the trimmed mean of the distance
        # mean = np.mean(stats[i])
        # std = np.std(stats[i])
        # #remove the outliers
        # stats[i] = stats[i][np.where(np.abs(stats[i]-mean)<std)]
        # #update the mean and std
        # mean = np.mean(stats[i])
        # stats[i] = [mean]
    return stats

def normalize_keypoints_3d(keypoints_3d):
    #normalize
    stats=get_stats(keypoints_3d)
    new_keypoints_3d = []
    for frame in keypoints_3d:
        #rescale the distance between the keypoints 0 and 7 to stats[6]
        frame=rescale_key_point(stats,6,0,7,[1,2,3,4,5,6],frame)
        #rescale the distance between the keypoints 7 and 8 to stats[7]
        frame=rescale_key_point(stats,7,7,8,[9,10,11,12,13,14,15,16],frame)   
        #rescale the distance between the keypoints 8 and 9 to stats[14]
        frame=rescale_key_point(stats,14,8,9,[10],frame)
        #rescale the distance between the keypoints 9 and 10 to stats[15]
        frame=rescale_key_point(stats,15,9,10,[],frame)
        #rescale the distance between the keypoints 8 and 11 to stats[9]
        frame=rescale_key_point(stats,9,8,11,[12,13],frame)
        #rescale the distance between the keypoints 8 and 14 to stats[8]
        frame=rescale_key_point(stats,8,8,14,[15,16],frame)
        #rescale the distance between the keypoints 0 and 1 to stats[0]
        frame=rescale_key_point(stats,0,0,1,[2,3],frame)
        #rescale the distance between the keypoints 0 and 4 to stats[1]
        frame=rescale_key_point(stats,1,0,4,[5,6],frame)
        #rescale the distance between the keypoints 1 and 2 to stats[2]
        frame=rescale_key_point(stats,2,1,2,[3],frame)
        #rescale the distance between the keypoints 2 and 3 to stats[4]
        frame=rescale_key_point(stats,4,2,3,[],frame)
        #rescale the distance between the keypoints 4 and 5 to stats[3]
        frame=rescale_key_point(stats,3,4,5,[6],frame)
        #rescale the distance between the keypoints 5 and 6 to stats[5]
        frame=rescale_key_point(stats,5,5,6,[],frame)
        #rescale the distance between the keypoints 11 and 12 to stats[12]
        frame=rescale_key_point(stats,12,11,12,[13],frame)
        #rescale the distance between the keypoints 12 and 13 to stats[13]
        frame=rescale_key_point(stats,13,12,13,[],frame)
        #rescale the distance between the keypoints 14 and 15 to stats[10]
        frame=rescale_key_point(stats,10,14,15,[16],frame)
        #rescale the distance between the keypoints 15 and 16 to stats[11]
        frame=rescale_key_point(stats,11,15,16,[],frame)

        new_keypoints_3d.append(frame)

    return np.array(new_keypoints_3d)

def smoothing(keypoints_3d, window_size=5, poly_order=3, window_size_2=5, poly_order_2=3):
    smoothed_keypoints = np.zeros_like(keypoints_3d)
    for i in range(17):
        if i in [2,3,5,6,12,13,15,16]:
            if window_size>poly_order:
                for j in range(3):
                    data=keypoints_3d[:,i,j]
                    smoothed_data = savgol_filter(data, window_size, poly_order)
                    smoothed_keypoints[:, i, j] = smoothed_data
            else:
                smoothed_keypoints[:, i, :] = keypoints_3d[:, i, :]
        else:
            if window_size_2>poly_order_2:
                for j in range(3):
                    data=keypoints_3d[:,i,j]
                    smoothed_data = savgol_filter(data, window_size_2, poly_order_2)
                    smoothed_keypoints[:, i, j] = smoothed_data
            else:
                smoothed_keypoints[:, i, :] = keypoints_3d[:, i, :]
    return np.array(smoothed_keypoints)

def create_gif(image_folder_path, output_gif_path, enlarge=False):

    folder = os.listdir(image_folder_path)

    # Sort the files to ensure they are in the correct order
    folder.sort()

    images = []

    for filename in folder:
        img_path = os.path.join(image_folder_path, filename)
        img = cv2.imread(img_path)
        
        # Ensure the image is not None
        if img is not None:
            if enlarge:
                #for the first image in 107
                if img.shape[0] < 100: 
                    img = cv2.copyMakeBorder(img, 100-img.shape[0], 0, 0, 0, cv2.BORDER_CONSTANT, value=[0,0,0])
                if img.shape[1] < 100:
                    img = cv2.copyMakeBorder(img, 0, 0, 100-img.shape[1], 0, cv2.BORDER_CONSTANT, value=[0,0,0])
                img = cv2.resize(img, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            images.append(img_rgb)

    # Use imageio to create the GIF
    imageio.mimsave(output_gif_path, images, duration=0.1, loop=0)

if __name__ == "__main__":
    import argparse
    from tqdm import tqdm
    parser=argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=os.getcwd().replace('/3dsp_utils','')+ '/3dsp_pre_process/train')
    parser.add_argument('--root', type=str, default=os.getcwd().replace('3dsp_utils','3dsp')+ '/train')
    parser.add_argument('--folder', type=str, default=False,help="folder name to visualize. If not specified, visualize all folders in the root.")
    parser.add_argument('--save_gif', type=bool, default=True)
    parser.add_argument('--create_2d_gif', type=bool, default=True)
    parser.add_argument('--video_length', type=int, default=20)
    parser.add_argument('--only_2d', action='store_true', help="if specified, only visualize 2d pose", default=True)
    args=parser.parse_args()

    if args.folder:
        folder_list = ["00107"]
    else:
        folder_list = os.listdir(args.root)
        
    for folder in tqdm(folder_list):
        #continue if the folder in save_path already exists
        # if os.path.exists(os.path.join(args.save_path,folder)):
        #     continue
        
        #load the 3d keypoints
        keypoints_3d=get_keypoints_3d(folder)

        #create gif of original 3d keypoints
        if args.save_gif:
            plot_3d_keypoints(keypoints_3d, folder)
            output_dir = os.path.join(args.save_path,folder)
            os.makedirs(output_dir, exist_ok=True)
            os.makedirs(output_dir+"/gif", exist_ok=True)
            create_gif(output_dir+"/pose3D", output_dir+f"/gif/3d.gif")

        #normalize
        keypoints_3d=normalize_keypoints_3d(keypoints_3d)
        #rotation
        keypoints_3d=rotate_3d_keypoints(keypoints_3d)
        #smoothing
        keypoints_3d = smoothing(keypoints_3d,5,1,5,3)

        #update the json files
        for i in range(len(keypoints_3d)):
            json_path = os.path.join(args.root, folder, "posture", f'{str(i+1).zfill(3)}.json')
            new_json_path = os.path.join(args.save_path, folder, "posture")
            os.makedirs(new_json_path, exist_ok=True)
            new_json_path = os.path.join(new_json_path, f'{str(i+1).zfill(3)}.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
                for k in range(len(data['keypont_3d'])):
                    data['keypont_3d'][str(k)]["x"] = keypoints_3d[i][k][0]
                    data['keypont_3d'][str(k)]["y"] = keypoints_3d[i][k][1]
                    data['keypont_3d'][str(k)]["z"] = keypoints_3d[i][k][2]
                json_string = json.dumps(data, indent=4)
                with open(new_json_path, "w") as json_file:
                    json_file.write(json_string)
                f.close()

        #3d pose visualization
        plot_3d_keypoints(keypoints_3d, folder)

        #save as gif
        if args.save_gif:
            create_gif(output_dir+"/pose3D", output_dir+f"/gif/pre_processed_3d.gif")

        #create gif of 2d keypoints
        if args.create_2d_gif:
            from vis_post import vis_pose
            args.folder=folder
            vis_pose(args)
            create_gif(os.path.join(args.save_path,folder,"pose2D"), output_dir+f"/gif/2d.gif",True)

