import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from sklearn.model_selection import train_test_split
from model import GCNLSTMAutoencoder
import numpy as np
import os
import json
import pdb

class CustomDataset(Dataset):
    def __init__(self, data_path):
        self.data_path = data_path
        self.data_files = os.listdir(data_path)

    def __len__(self):
        return len(self.data_files)

    def __getitem__(self, index):
        posture_path = os.path.join(self.data_path, self.data_files[index],"posture")
        batch=[]
        #loop through 20 frames
        for i in range(20):
            json_path=os.path.join(posture_path,f'{str(i+1).zfill(3)}.json')
            #read json
            with open(json_path, 'r') as file:
                # Load the JSON data from the file
                data = json.load(file)
                pose_3d=data['keypont_3d']
                coord=[]
                #loop through 17 keypoints
                for k,v in pose_3d.items():
                    coord.append([v["x"],v["y"],v["z"]])
            batch.append(coord)
        
        #convert to tensor
        data_item = torch.tensor(batch, dtype=torch.float32)
        return data_item

def create_adj_matrix():
    I = np.array( [0, 0, 1, 4, 2, 5, 0, 7,  8,  8, 14, 15, 11, 12, 8,  9])
    J = np.array( [1, 4, 2, 5, 3, 6, 7, 8, 14, 11, 15, 16, 12, 13, 9, 10])

    num_nodes = max(max(I), max(J)) + 1
    adj_matrix = np.zeros((num_nodes, num_nodes))

    # Set edge weights
    for i, j in zip(I, J):
        adj_matrix[i, j] = 1  # Set edge from node i to node j
        adj_matrix[j, i] = 1  # Set edge from node j to node i

    # Add autoconnections (self-loops)
    np.fill_diagonal(adj_matrix, 1)

    # Convert to PyTorch tensor
    adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

    return adj_matrix
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='train the gcn lstm ae model')
    parser.add_argument("--out_path", type=str,default=os.getcwd()+"/deep_learning+kmean_result")
    parser.add_argument("--data_path", type=str, default=os.getcwd().replace("analysis","3dsp_pre_process")+"/train")
    args=parser.parse_args()
    # Create adjacency matrix
    adj_matrix = create_adj_matrix()

    # Define paths
    out_path =  args.out_path
    data_path = args.data_path
    if not os.path.exists(out_path):
        os.makedirs(out_path)

    #learning parameters
    batch_size = 10
    learning_rate = 1e-3
    num_epochs = 1000
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    num_workers = 4

    #model parameters
    seq_len = 20 #fixed
    num_nodes = 17 #fixed
    gcn_input_size = 3 #fixed
    gcn_hidden_size = 8
    gcn_output_size = 4
    lstm_hidden_size = 64
    lstm_num_layer = 3
    model = GCNLSTMAutoencoder(gcn_input_size, gcn_hidden_size, gcn_output_size,lstm_hidden_size, lstm_num_layer,batch_size,seq_len, num_nodes) 
    model.to(device)

    # Split dataset into training and validation sets
    dataset = CustomDataset(data_path)
    train_data, val_data = train_test_split(dataset, test_size=0.2, random_state=42)

    # Define data loaders
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True,num_workers=num_workers)
    val_loader = DataLoader(val_data, batch_size=batch_size, shuffle=False,num_workers=num_workers)

    # Define optimizer and loss function
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    loss_function = torch.nn.MSELoss()

    # Initialize variables for early stopping
    best_val_loss = np.inf
    patience = 10
    counter = 0

    # Training loop
    for epoch in range(num_epochs):
        model.train()
        train_loss = 0.0
        adj_matrix = adj_matrix.to(device)
        # Iterate over batches
        for batch_idx, data in enumerate(train_loader):
            optimizer.zero_grad()

            data = data.to(device)
            # Forward pass
            recon_data = model(data, adj_matrix)

            # Compute reconstruction loss
            recon_loss = loss_function(recon_data, data)

            # Total loss 
            loss = recon_loss 

            # Backward pass
            loss.backward()
            optimizer.step()

            train_loss += loss.item()

        # Compute average training loss
        train_loss /= len(train_loader)

        # Validation
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for data in val_loader:
                data = data.to(device)
                recon_data = model(data, adj_matrix)
                recon_loss = loss_function(recon_data, data)
                loss = recon_loss 
                val_loss += loss.item()
            val_loss /= len(val_loader)

        # Print training and validation loss with to cpu    
        print(f'Epoch [{epoch + 1}/{num_epochs}], Train Loss: {train_loss:.4f}, Val Loss: {val_loss:.4f}')
        

        # Early stopping
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            counter = 0
            # Save the model
            torch.save(model.state_dict(), out_path+'/best_model.pth')
        else:
            counter += 1
            if counter >= patience:
                print("Early stopping...")
                break
