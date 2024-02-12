import json
import os
import pandas as pd
import numpy as np
import pdb

def load_json(dataset_path):
    folder_list=os.listdir(dataset_path)
    df_per_folder={}
    df_all=[]

    for folder in folder_list:
        folder_path=os.path.join(dataset_path,folder,"posture")
        json_list=os.listdir(folder_path)
        df_folder=[]
        for i in range(20):
            json_path=os.path.join(folder_path,f'{str(i+1).zfill(3)}.json')
            #read json
            # pdb.set_trace()
            with open(json_path, 'r') as file:
                # Load the JSON data from the file
                data = json.load(file)
                pose_3d=data['keypont_3d']
                temp_dict={}
                for k,v in pose_3d.items():
                    temp_dict[f"{k}_x"]=v["x"]
                    temp_dict[f"{k}_y"]=v["y"]
                    temp_dict[f"{k}_z"]=v["z"]
                    
                df_frame = pd.DataFrame.from_dict(temp_dict, orient='index').T
                df_folder.append(df_frame)
        
        df_folder=pd.concat(df_folder).reset_index(drop=True)
        df_per_folder[folder]=df_folder
        df_all.append(df_folder)
    df_all=pd.concat(df_all).reset_index(drop=True)
    
    return df_per_folder,df_all

def calculate_angle(A, B, C):
    '''
    # Example coordinates (replace with your actual coordinates)
    A = [1, 2, 3]  # Point A
    B = [4, 5, 6]  # Point B the middle point
    C = [7, 8, 9]  # Point C
    '''
    # Calculate vectors AB and BC
    AB = np.array(B) - np.array(A)
    BC = np.array(C) - np.array(B)
    
    # Calculate dot product
    dot_product = np.dot(AB, BC)
    
    # Calculate magnitudes
    magnitude_AB = np.linalg.norm(AB)
    magnitude_BC = np.linalg.norm(BC)
    
    # Calculate angle in radians
    angle_radians = np.arccos(dot_product / (magnitude_AB * magnitude_BC))
    
    # Convert angle to degrees
    angle_degrees = np.degrees(angle_radians)
    
    return angle_degrees

def joint_features(data_dict):
    joint_feautures=[]
    joint_name="ankle"
    # get ankle features () 3 Left Ankle 6 Right Ankle,  2 Left Knee, 5 Right Knee, 12 Right Elbow, 15 Left Elbow
    # use 0 Center of Hips and 8 Center of Shoulder to aviod error in left and right switch
    left_id=3
    right_id=6
    hip_id=0
    left_knee_id=2
    right_knee_id=5
    shoulder_id=8
    left_elbow_id=15
    right_elbow_id=12

    for k,v in data_dict.items():
        dist_x_left,dist_y_left,dist_z_left,dist_xy_left,dist_xz_left, dist_yz_left, dist_all_left=[],[],[],[],[],[],[]
        dist_x_right,dist_y_right,dist_z_right,dist_xy_right,dist_xz_right, dist_yz_right, dist_all_right=[],[],[],[],[],[],[]
        #calculate the left and right Knee Angle, Hip Angle, Shoulder Angle
        knee_angle_left,knee_angle_right,hip_angle_left,hip_angle_right,shoulder_angle_left,shoulder_angle_right=[],[],[],[],[],[]
        for i in range(19):
            dist_x_left.append(abs(v.iloc[i+1][f'{left_id}_x']-v.iloc[i][f'{left_id}_x']))
            dist_y_left.append(abs(v.iloc[i+1][f'{left_id}_y']-v.iloc[i][f'{left_id}_y']))
            dist_z_left.append(abs(v.iloc[i+1][f'{left_id}_z']-v.iloc[i][f'{left_id}_z']))
            dist_x_right.append(abs(v.iloc[i+1][f'{right_id}_x']-v.iloc[i][f'{right_id}_x']))
            dist_y_right.append(abs(v.iloc[i+1][f'{right_id}_y']-v.iloc[i][f'{right_id}_y']))
            dist_z_right.append(abs(v.iloc[i+1][f'{right_id}_z']-v.iloc[i][f'{right_id}_z']))
            dist_xy_left.append((dist_x_left[i]**2+dist_y_left[i]**2)**0.5)
            dist_xz_left.append((dist_x_left[i]**2+dist_z_left[i]**2)**0.5)
            dist_yz_left.append((dist_y_left[i]**2+dist_z_left[i]**2)**0.5)
            dist_all_left.append((dist_x_left[i]**2+dist_y_left[i]**2+dist_z_left[i]**2)**0.5)
            dist_xy_right.append((dist_x_right[i]**2+dist_y_right[i]**2)**0.5)
            dist_xz_right.append((dist_x_right[i]**2+dist_z_right[i]**2)**0.5)
            dist_yz_right.append((dist_y_right[i]**2+dist_z_right[i]**2)**0.5)
            dist_all_right.append((dist_x_right[i]**2+dist_y_right[i]**2+dist_z_right[i]**2)**0.5)
            #calculate the left and right Knee Angle, Hip Angle, Shoulder Angle
            knee_angle_left.append(calculate_angle([v.iloc[i][f'{hip_id}_x'],v.iloc[i][f'{hip_id}_y'],v.iloc[i][f'{hip_id}_z']],
                                                   [v.iloc[i][f'{left_knee_id}_x'],v.iloc[i][f'{left_knee_id}_y'],v.iloc[i][f'{left_knee_id}_z']],
                                                   [v.iloc[i][f'{left_id}_x'],v.iloc[i][f'{left_id}_y'],v.iloc[i][f'{left_id}_z']]))
            knee_angle_right.append(calculate_angle([v.iloc[i][f'{hip_id}_x'],v.iloc[i][f'{hip_id}_y'],v.iloc[i][f'{hip_id}_z']],
                                                    [v.iloc[i][f'{right_knee_id}_x'],v.iloc[i][f'{right_knee_id}_y'],v.iloc[i][f'{right_knee_id}_z']],
                                                    [v.iloc[i][f'{right_id}_x'],v.iloc[i][f'{right_id}_y'],v.iloc[i][f'{right_id}_z']]))
            hip_angle_left.append(calculate_angle([v.iloc[i][f'{shoulder_id}_x'],v.iloc[i][f'{shoulder_id}_y'],v.iloc[i][f'{shoulder_id}_z']],
                                                    [v.iloc[i][f'{hip_id}_x'],v.iloc[i][f'{hip_id}_y'],v.iloc[i][f'{hip_id}_z']],
                                                    [v.iloc[i][f'{left_knee_id}_x'],v.iloc[i][f'{left_knee_id}_y'],v.iloc[i][f'{left_knee_id}_z']]))
            hip_angle_right.append(calculate_angle([v.iloc[i][f'{shoulder_id}_x'],v.iloc[i][f'{shoulder_id}_y'],v.iloc[i][f'{shoulder_id}_z']],
                                                    [v.iloc[i][f'{hip_id}_x'],v.iloc[i][f'{hip_id}_y'],v.iloc[i][f'{hip_id}_z']],
                                                    [v.iloc[i][f'{right_knee_id}_x'],v.iloc[i][f'{right_knee_id}_y'],v.iloc[i][f'{right_knee_id}_z']]))
            shoulder_angle_left.append(calculate_angle([v.iloc[i][f'{hip_id}_x'],v.iloc[i][f'{hip_id}_y'],v.iloc[i][f'{hip_id}_z']],
                                                    [v.iloc[i][f'{shoulder_id}_x'],v.iloc[i][f'{shoulder_id}_y'],v.iloc[i][f'{shoulder_id}_z']],
                                                    [v.iloc[i][f'{left_elbow_id}_x'],v.iloc[i][f'{left_elbow_id}_y'],v.iloc[i][f'{left_elbow_id}_z']]))
            shoulder_angle_right.append(calculate_angle([v.iloc[i][f'{hip_id}_x'],v.iloc[i][f'{hip_id}_y'],v.iloc[i][f'{hip_id}_z']],
                                                    [v.iloc[i][f'{shoulder_id}_x'],v.iloc[i][f'{shoulder_id}_y'],v.iloc[i][f'{shoulder_id}_z']],
                                                    [v.iloc[i][f'{right_elbow_id}_x'],v.iloc[i][f'{right_elbow_id}_y'],v.iloc[i][f'{right_elbow_id}_z']]))
        # pdb.set_trace()
        #define the shooting leg with distance
        if sum(dist_x_left)>sum(dist_x_right):
            joint_feautures.append([sum(dist_x_left),sum(dist_y_left),sum(dist_z_left),sum(dist_xy_left),sum(dist_xz_left),sum(dist_yz_left),sum(dist_all_left),
                                max(v.iloc[:,left_id*3]),max(v.iloc[:,left_id*3+1]),max(v.iloc[:,left_id*3+2]),
                                min(v.iloc[:,left_id*3]),min(v.iloc[:,left_id*3+1]),min(v.iloc[:,left_id*3+2]),
                                sum(knee_angle_left)/len(knee_angle_left),sum(hip_angle_left)/len(hip_angle_left),sum(shoulder_angle_left)/len(shoulder_angle_left),sum(shoulder_angle_right)/len(shoulder_angle_right),
                                max(knee_angle_left),max(hip_angle_left),max(shoulder_angle_left),max(shoulder_angle_right),
                                min(knee_angle_left),min(hip_angle_left),min(shoulder_angle_left),min(shoulder_angle_right)
                                ])

        else:
            joint_feautures.append([sum(dist_x_right),sum(dist_y_right),sum(dist_z_right),sum(dist_xy_right),sum(dist_xz_right),sum(dist_yz_right),sum(dist_all_right),
                                max(v.iloc[:,right_id*3]),max(v.iloc[:,right_id*3+1]),max(v.iloc[:,right_id*3+2]),
                                min(v.iloc[:,right_id*3]),min(v.iloc[:,right_id*3+1]),min(v.iloc[:,right_id*3+2]),
                                sum(knee_angle_right)/len(knee_angle_right),sum(hip_angle_right)/len(hip_angle_right),sum(shoulder_angle_left)/len(shoulder_angle_left),sum(shoulder_angle_right)/len(shoulder_angle_right),
                                max(knee_angle_right),max(hip_angle_right),max(shoulder_angle_left),max(shoulder_angle_right),
                                min(knee_angle_right),min(hip_angle_right),min(shoulder_angle_left),min(shoulder_angle_right)
                                ])
    # convert to dataframe
    df_joint_feautures=pd.DataFrame(joint_feautures,columns=[f"{joint_name}_dist_x",f"{joint_name}_dist_y",
                                                             f"{joint_name}_dist_z",f"{joint_name}_dist_xy",
                                                             f"{joint_name}_dist_xz",f"{joint_name}_dist_yz",
                                                             f"{joint_name}_dist_all",f"{joint_name}_max_x",
                                                             f"{joint_name}_max_y",f"{joint_name}_max_z",
                                                             f"{joint_name}_min_x",f"{joint_name}_min_y",
                                                             f"{joint_name}_min_z","knee_angle_mean",
                                                            "hip_angle_mean","shoulder_angle_left_mean","shoulder_angle_right_mean",
                                                            "knee_angle_max","hip_angle_max","shoulder_angle_left_max","shoulder_angle_right_max",
                                                            "knee_angle_min","hip_angle_min","shoulder_angle_left_min","shoulder_angle_right_min"
                                                             ])
    # df_joint_feautures=pd.DataFrame(joint_feautures,columns=[f"{joint_name}_dist_x","dist_y","dist_z","dist_xy","dist_xz","dist_yz","dist_all","max_x","max_y","max_z","min_x","min_y","min_z"])
    return df_joint_feautures
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='analysis')
    parser.add_argument("--data_path", type=str, default=os.getcwd().replace("analysis","3dsp")+"/train")
    parser.add_argument("--label_path", type=str, default=os.getcwd()+"/deep_learning+kmean_result/t-SNE.csv")
    parser.add_argument("--out_path", type=str, default=os.getcwd()+"deep_learning+kmean_result")
    args=parser.parse_args()



    data_path= args.data_path
    label_path= args.label_path
    out_path= args.out_path

    data_dict,df_all=load_json(data_path)

    max_x,max_y,max_z,min_x,min_y,min_z=[],[],[],[],[],[]
    for i in range(17):
        max_x.append(df_all.iloc[:,i*3].max())
        max_y.append(df_all.iloc[:,i*3+1].max())
        max_z.append(df_all.iloc[:,i*3+2].max())
        min_x.append(df_all.iloc[:,i*3].min())
        min_y.append(df_all.iloc[:,i*3+1].min())
        min_z.append(df_all.iloc[:,i*3+2].min())
    max_x,max_y,max_z,min_x,min_y,min_z=max(max_x),max(max_y),max(max_z),min(min_x),min(min_y),min(min_z)
    print("max_x",max_x,"min_x",min_x)
    print("max_y",max_y,"min_y",min_y)
    print("max_z",max_z,"min_z",min_z)

    df_joint_feautures=joint_features(data_dict)

    df_label=pd.read_csv(label_path)

    df_analysis= df_label["label"].copy()

    #summarze the ankle features for each cluster
    df_analysis=pd.concat([df_analysis,df_joint_feautures],axis=1)
    summary=df_analysis.groupby('label').mean()
    # pdb.set_trace()
    #add row for differnce and percentage change
    summary=summary.append(abs(summary.iloc[0]-summary.iloc[1]),ignore_index=True)
    summary=summary.append(abs(summary.iloc[0]-summary.iloc[2]),ignore_index=True)
    summary=summary.append(abs(summary.iloc[1]-summary.iloc[2]),ignore_index=True)
    summary = summary.append(abs(summary.iloc[0] - summary.iloc[1]) / ((summary.iloc[0] + summary.iloc[1]) / 2), ignore_index=True)
    summary = summary.append(abs(summary.iloc[0] - summary.iloc[2]) / ((summary.iloc[1] + summary.iloc[2]) / 2), ignore_index=True)
    summary = summary.append(abs(summary.iloc[1] - summary.iloc[2]) / ((summary.iloc[0] + summary.iloc[2]) / 2), ignore_index=True)

    #set the index name
    summary.index=["cluster_1","cluster_2","cluster_3","diff_1-2","diff_1-3","diff_2-3","per_1-2","per_1-3","per_2-3"]

    #round up to 6 dp
    summary=summary.round(6)

    #save the result and summary
    df_analysis.to_csv(out_path+"/joint_feautures.csv")
    summary.to_csv(out_path+"/joint_feautures_summary.csv")




