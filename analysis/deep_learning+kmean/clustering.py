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
from model import GCNLSTMAutoencoder
from train import CustomDataset, create_adj_matrix
from torch.utils.data import DataLoader
import torch

def det_cluster_num(df_kmean,outpath):
    if not os.path.exists(outpath):
        os.makedirs(outpath)
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
    df_centers.to_csv(outpath+"/cluster_center.csv")
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

def load_stats(info_path):
    folder_list=os.listdir(info_path)
    info=[]
    config = configparser.ConfigParser()
    for folder in folder_list:
        ini_path=os.path.join(info_path,folder,"info.ini")
        config.read(ini_path)
        info.append(config['info']['label'])    
    return info

import argparse
parser = argparse.ArgumentParser(description='clustering')
parser.add_argument("--model_path", type=str, default=os.getcwd()+"/deep_learning+kmean/params/best_model.pth")
parser.add_argument("--out_path", type=str, default=os.getcwd()+"/deep_learning+kmean_result")
parser.add_argument("--data_path", type=str, default=os.getcwd().replace("analysis","3dsp_pre_process")+"/train")
parser.add_argument("--dataset_path", type=str, default=os.getcwd().replace("analysis","3dsp")+"/train")
parser.add_argument("--info_path", type=str, default=os.getcwd().replace("analysis","3dsp")+"/train")         
args=parser.parse_args()

model_path= args.model_path
out_path =  args.out_path
data_path = args.data_path
dataset_path = args.dataset_path
info_path = args.info_path

if not os.path.exists(out_path):
    os.makedirs(out_path)


batch_size = 10

#model parameters
seq_len = 20 #fixed
num_nodes = 17 #fixed
gcn_input_size = 3 #fixed
gcn_hidden_size = 8
gcn_output_size = 4
lstm_hidden_size = 64
lstm_num_layer = 3
model = GCNLSTMAutoencoder(gcn_input_size, gcn_hidden_size, gcn_output_size,lstm_hidden_size, lstm_num_layer,batch_size,seq_len, num_nodes) 
model.load_state_dict(torch.load(model_path))

#load data
dataset = CustomDataset(data_path)
data_loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,num_workers=4)
adj_matrix = create_adj_matrix()

#extract latent space
model.eval()
    
latent_variables = []
with torch.no_grad():
    for data in data_loader:
        gcn_encoded = model.gcn_encoder(data, adj_matrix)
        lstm_encoded = model.lstm_encoder(gcn_encoded.view(batch_size, seq_len, -1))
        #convert to numpy
        lstm_encoded = lstm_encoded.cpu().numpy()
        latent_variables.append(lstm_encoded)

latent_variables = np.array(latent_variables)
latent_variables = latent_variables.reshape(-1, lstm_hidden_size)
# pdb.set_trace()

#convert to dataframe
df_kmean=pd.DataFrame(latent_variables)

#determine the number of cluster
# det_cluster_num(df_kmean,out_path)

#set kmean number
analysis_kmean(df_kmean,out_path,dataset_path,info_path,num_cluster=3)