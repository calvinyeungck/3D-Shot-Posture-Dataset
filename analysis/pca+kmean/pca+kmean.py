#kmean++ and t-SNE from Wear et al. 

from sklearn.cluster import KMeans
from sklearn.manifold import TSNE
import numpy as np
import os
import json
import matplotlib.pyplot as plt
import pandas as pd
import pdb
from yellowbrick.cluster import KElbowVisualizer
from yellowbrick.cluster import SilhouetteVisualizer
from sklearn.decomposition import PCA
from sklearn.metrics import pairwise_distances_argmin_min
import configparser

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

def load_stats(info_path):
    folder_list=os.listdir(info_path)
    info=[]
    config = configparser.ConfigParser()
    for folder in folder_list:
        ini_path=os.path.join(info_path,folder,"info.ini")
        config.read(ini_path)
        info.append(config['info']['label'])    
    return info

def perform_pca(df_folder,df_all,outpath):
    pca = PCA()
    pca.fit(df_all)

    explained_variance_ratio_cumsum = np.cumsum(pca.explained_variance_ratio_)
    plt.plot(range(1, len(explained_variance_ratio_cumsum) + 1), explained_variance_ratio_cumsum, color='#1f77b4')
    plt.xlabel('Number of Principal Components')
    plt.ylabel('Cumulative Explained Variance Ratio')
    plt.title('Explained Variance Ratio')

    n_components_80 = np.argmax(explained_variance_ratio_cumsum >= 0.80) + 1

    plt.axvline(x=n_components_80, color='#ff7f0e', linestyle='--', label=f'{n_components_80} Components (80% Explained Variance)')

    plt.legend()
    plt.grid(True)
    plt.savefig(outpath+'/pca.png')
    plt.close()

    df_kmean=[]
    for folder in df_folder.keys():
        df_folder[folder]=pca.fit_transform(df_folder[folder])[:, :7]
        #convert the 2d array to 1d array
        df_folder[folder]=df_folder[folder].flatten()
        df_kmean.append(df_folder[folder])


    df_kmean=pd.DataFrame(df_kmean).reset_index(drop=True)
    return df_kmean

def det_cluster_num(df_kmean,outpath):
    #mute the warning
    import warnings
    warnings.filterwarnings("ignore")

    kmeans = KMeans( init='k-means++',random_state=42)
    vis = KElbowVisualizer(kmeans, k=(1,11))

    vis.fit(df_kmean)
    vis.show(outpath+'/elbow_method.png')
    plt.close()


    vis2 = SilhouetteVisualizer(kmeans, colors='yellowbrick')
    vis2.fit(df_kmean)
    vis2.show(outpath+'/silhouette_method.png')
    plt.close()

def analysis_kmean(df_kmean,outpath,dataset_path,info_path,num_cluster):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
    kmeans = KMeans(n_clusters=num_cluster, init='k-means++', random_state=42)
    kmeans.fit(df_kmean)

    # Initialize t-SNE with desired parameters
    tsne = TSNE(n_components=2, perplexity=30, n_iter=1000, random_state=42)

    # Perform dimensionality reduction
    len_df_kmean=len(df_kmean)
    df_centers = pd.DataFrame(kmeans.cluster_centers_)
    df_kmean_embedded = pd.concat([df_kmean, df_centers])
    df_kmean_embedded = df_kmean_embedded.reset_index(drop=True)

    X_embedded = tsne.fit_transform(df_kmean_embedded)

    df_kmean_embedded=X_embedded[:len_df_kmean,:]

    df_center_embedded=X_embedded[len_df_kmean:,:]
    #calculate the distance between the center and the cluster
    labels=kmeans.labels_

    df_out=pd.DataFrame(df_kmean_embedded,columns=["x","y"])
    df_out["label"]=labels+1
    df_out.to_csv(outpath+"/t-SNE.csv")

    # Visualize the lower-dimensional embedding
    for i in range(num_cluster):
        plt.scatter(df_kmean_embedded[labels == i, 0], df_kmean_embedded[labels == i, 1], label=f'Cluster {i+1}',cmap='viridis')
    plt.scatter(df_center_embedded[:, 0], df_center_embedded[:, 1], c='#f0e442', s=100, label='Centroids')

    plt.title('t-SNE Visualization')
    plt.xlabel('Dimension 1')
    plt.ylabel('Dimension 2')
    plt.legend()    
    plt.savefig(outpath+'/t-SNE.png')  # Save the plot as a PNG file
    plt.close()

    # set index as column "clips_id"
    df_kmean['clips_id'] = df_kmean.index
 
    #get the closest point to the center for each cluster
    #calculate the distance between the center and the cluster
    for i in range(5):
        # Calculate closest points and distances
        closest_points_idx, _ = pairwise_distances_argmin_min(df_centers.values, df_kmean.loc[:, ~df_kmean.columns.isin(['clips_id'])].values)
        # print(closest_points_idx)
        print(f"{i+1} closest points to centroids:")
        for idx, point in enumerate(closest_points_idx):
            print(f"Cluster {idx+1} centroid: {df_centers.index[idx]}, Closest point: {df_kmean['clips_id'][point]+1}")

        # Drop the closest points to prepare for the next iteration
        df_kmean = df_kmean.drop(closest_points_idx)
        df_kmean = df_kmean.reset_index(drop=True)

    info=load_stats(info_path)
    labels=kmeans.labels_

    #creat a df to store the info and the label
    df_info=pd.DataFrame(info,columns=["label"])
    #add 1 to all the label
    labels=labels+1
    df_info["cluster"]=labels

    #print the stats of the cluster
    print(df_info.groupby('cluster').describe())


#read data
import argparse
parser = argparse.ArgumentParser(description='Process some integers.')
parser.add_argument('--dataset_path', type=str, default=os.getcwd().replace("analysis","3dsp")+"/train", help='dataset path')
parser.add_argument('--info_path', type=str, default=os.getcwd().replace("analysis","3dsp")+"/train", help='info path')
parser.add_argument('--outpath', type=str, default=os.getcwd()+"/pca+kmean_result", help='output path')
args = parser.parse_args()

dataset_path=args.dataset_path
info_path=args.info_path
outpath=args.outpath
if not os.path.exists(outpath):
    os.makedirs(outpath)

#load the data
df_folder,df_all=load_json(dataset_path)

# Perform PCA
df_kmean=perform_pca(df_folder,df_all,outpath)

#determine the number of cluster
det_cluster_num(df_kmean,outpath)

#set kmean to 3
analysis_kmean(df_kmean,outpath,dataset_path,info_path,3)



