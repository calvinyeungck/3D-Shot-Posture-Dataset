#ref the code of MotionAGFormer https://github.com/TaatiTeam/MotionAGFormer

#2d pose visualization

import os
import cv2
import numpy as np
import json
import copy
from tqdm import tqdm
from MotionAGFormer.demo.vis_sn import show2Dpose, show3Dpose, showimage
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pdb

def vis_pose(args):
    #get all folders in the root
    root = args.root
    output_dir = args.save_path
    video_length = args.video_length

    if args.folder is not None:
        folders = [args.folder]
    else:
        folders = os.listdir(root)

    for folder in folders:
        video_path=os.path.join(root, folder, 'img')

        #read the json file
        keypoints_2d = []
        keypoints_3d = []
        for i in range(video_length):
            json_path = os.path.join(root, folder, 'posture', f'{str(i+1).zfill(3)}.json')
            with open(json_path, 'r') as f:
                data = json.load(f)
                data_2d = data['keypont_2d']
                data_3d = data['keypont_3d']
                data_2d_list = []
                data_3d_list = []
                for k,v in data_2d.items():
                    data_2d_list.append([v["x"],v["y"],1])
                for k,v in data_3d.items():
                    data_3d_list.append([v["x"],v["y"],v["z"]])
                keypoints_2d.append(data_2d_list)
                keypoints_3d.append(data_3d_list)
                f.close()
        
        keypoints_2d = np.array(keypoints_2d)
        keypoints_3d = np.array(keypoints_3d)

        for i in range(video_length):
            img = cv2.imread(video_path + "/"+str(i+1).zfill(3) + '.jpg')
            img_size = img.shape

            input_2D = keypoints_2d[i]

            image = show2Dpose(input_2D, copy.deepcopy(img))

            output_dir_2D = os.path.join(output_dir,folder,'pose2D')
            os.makedirs(output_dir_2D, exist_ok=True)
            cv2.imwrite(os.path.join(output_dir_2D, str(('%03d'% (i+1))) + '.png'), image)

        if args.only_2d:
            continue
        # pdb.set_trace()

        #3d pose visualization
        joints_left =  [4, 5, 6, 11, 12, 13]
        joints_right = [1, 2, 3, 14, 15, 16]
        print('\nGenerating 3D pose...')
        for idx in range(video_length):
        # for idx, clip in enumerate(clips):
            # input_2D = normalize_screen_coordinates(clip, w=img_size[1], h=img_size[0]) 

            # input_2D_aug = copy.deepcopy(input_2D)
            # input_2D_aug[..., 0] *= -1
            # input_2D_aug[..., joints_left + joints_right, :] = input_2D_aug[..., joints_right + joints_left, :]
            
            # input_2D = torch.from_numpy(input_2D.astype('float32')).cuda()
            # input_2D_aug = torch.from_numpy(input_2D_aug.astype('float32')).cuda()

            # output_3D_non_flip = model(input_2D) 
            # output_3D_flip = model(input_2D)
            # output_3D = (output_3D_non_flip + output_3D_flip) / 2

            # if idx == len(clips) - 1:
            #     output_3D = output_3D[:, downsample]

            # output_3D[:, :, 0, :] = 0
            # post_out_all = output_3D[0].cpu().detach().numpy()
            post_out_all = keypoints_3d
            for j, post_out in enumerate(post_out_all):
                # rot =  [0.1407056450843811, -0.1500701755285263, -0.755240797996521, 0.6223280429840088]
                # rot = np.array(rot, dtype='float32')
                # post_out = camera_to_world(post_out, R=rot, t=0)
                # post_out[:, 2] -= np.min(post_out[:, 2])
                # max_value = np.max(post_out)
                # post_out /= max_value

                fig = plt.figure(figsize=(9.6, 5.4))
                gs = gridspec.GridSpec(1, 1)
                gs.update(wspace=-0.00, hspace=0.05) 
                ax = plt.subplot(gs[0], projection='3d')
                pdb.set_trace()
                show3Dpose(post_out, ax)
                
                #update post_out_all
                post_out_all[j]=post_out

                output_dir_3D = os.path.join(output_dir,folder,'pose3D')
                os.makedirs(output_dir_3D, exist_ok=True)
                str(('%04d'% (idx * 243 + j)))
                plt.savefig(os.path.join(output_dir_3D, str(('%03d'% (j+1))) + '.png'), dpi=200, format='png', bbox_inches='tight')
                plt.close(fig)

        # all
        image_2d_dir = os.listdir(output_dir_2D)
        image_3d_dir = os.listdir(output_dir_3D)

        print('\nGenerating demo...')
        # for i in tqdm(range(len(image_2d_dir))):
        for i in range(len(image_2d_dir)):
            image_2d = plt.imread(os.path.join(output_dir_2D,image_2d_dir[i]))
            image_3d = plt.imread(os.path.join(output_dir_3D,image_3d_dir[i]))

            ## crop
            edge = (image_2d.shape[1] - 100) // 2
            edge = 0 if edge < 0 else edge
            image_2d = image_2d[:, edge:image_2d.shape[1] - edge]

            edge = 130
            image_3d = image_3d[edge:image_3d.shape[0] - edge, edge:image_3d.shape[1] - edge]

            ## show
            font_size = 12
            fig = plt.figure(figsize=(15.0, 5.4))
            ax = plt.subplot(121)
            showimage(ax, image_2d)
            ax.set_title("Input", fontsize = font_size)

            ax = plt.subplot(122)
            showimage(ax, image_3d)
            ax.set_title("Reconstruction", fontsize = font_size)

            ## save
            output_dir_pose = os.path.join(output_dir,folder,'pose')
            os.makedirs(output_dir_pose, exist_ok=True)
            plt.subplots_adjust(top=1, bottom=0, right=1, left=0, hspace=0, wspace=0)
            plt.margins(0, 0)
            plt.savefig(os.path.join(output_dir_pose, str(('%03d'% (i+1))) + '.png'), dpi=200, bbox_inches = 'tight')
            plt.close(fig)

if __name__ == "__main__":
    import argparse
    parser=argparse.ArgumentParser()
    parser.add_argument('--save_path', type=str, default=os.getcwd()+'/3dsp_addition/vis_3dsp/')
    parser.add_argument('--root', type=str, default=os.getcwd().replace('3dsp_utils','3dsp')+ '/train')
    parser.add_argument('--folder', type=str, default="00001",help="folder name to visualize. If not specified, visualize all folders in the root.")
    parser.add_argument('--video_length', type=int, default=20)
    parser.add_argument('--only_2d', action='store_true', help="if specified, only visualize 2d pose", default=False)
    args=parser.parse_args()
    
    #plot 2d and 3d pose
    vis_pose(args)