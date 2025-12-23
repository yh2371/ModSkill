import torch
import torch.nn as nn
import torch.nn.functional as F
#['Pelvis', 'L_Hip', 'L_Knee', 'L_Ankle', 'L_Toe', 'R_Hip', 'R_Knee', 'R_Ankle', 'R_Toe', 'Torso', 'Spine', 'Chest', 'Neck', 'Head', 'L_Thorax', 'L_Shoulder', 'L_Elbow', 'L_Wrist', 'L_Hand', 'R_Thorax', 'R_Shoulder', 'R_Elbow', 'R_Wrist', 'R_Hand']
total = []
import joblib
class PartAttentionLayer(nn.Module):   
    def __init__(self, input_dim1=[4*24, 4*24, 6*24, 5*24, 5*24], input_dim2=[4*15, 4*15, 1+12+5*15, 5*15, 5*15], latent_dim=64):
        super(PartAttentionLayer, self).__init__()
        self.input_dims = [input_dim1[i] + input_dim2[i] for i in range(len(input_dim1))]
        self.input_dim1 = input_dim1
        self.input_dim2 = input_dim2
        self.J = 24
        self.num = len(self.input_dims)
        self.key = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dims[i], 256), nn.Linear(256,latent_dim)) for i in range(self.num)])
        self.value = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dims[i], 256), nn.Linear(256,latent_dim))for i in range(self.num)])
        self.query = nn.ModuleList([nn.Sequential(nn.Linear(self.input_dims[i], 256), nn.Linear(256,latent_dim)) for i in range(self.num)])
    
    def forward(self, x):
        keys = []
        values = []
        queries = []
        selfobs = torch.cat([x[:,13:13+8*15],x[:,:13], x[:,13+8*15:13+23*15]], dim=-1)
        taskobs = x[:,358:]
        taskobs = torch.cat([taskobs[:,1*24:9*24], taskobs[:,:1*24], taskobs[:,9*24:24*24]], dim=-1)
        for i in range(len(self.input_dims)):
            if i == 0:
                start1 = 0
                start2 = 0
            else:
                start1 = sum(self.input_dim1[:i])
                start2 = sum(self.input_dim2[:i])
            x_in = torch.cat([selfobs[:,start2:start2+self.input_dim2[i]], taskobs[:,start1:start1+self.input_dim1[i]]], dim=-1)
            keys.append(self.key[i](x_in).unsqueeze(2))
            values.append(self.value[i](x_in).unsqueeze(1))
            queries.append(self.query[i](x_in).unsqueeze(1))

        keys = torch.cat(keys, dim=2)  # Shape (batch_size, latent_dim, 6)
        values = torch.cat(values, dim=1)  # Shape (batch_size, 6, latent_dim)
        queries = torch.cat(queries, dim=1)  # Shape (batch_size, 5, latent_dim)
        attn_scores = torch.bmm(queries, keys)  # Shape (batch_size, 5, 6) 
        #print(attn_scores.shape)
        attn_weights = F.softmax(attn_scores, dim=2)  # Normalize scores #(batch_size, 5, 6)
        #print(attn_weights.shape)
        # total.append(F.softmax(attn_scores, dim=2).cpu().numpy())
        # joblib.dump(total, "attn_crouch.pkl")
        
        attended_features = torch.bmm(attn_weights, values).squeeze(1)  # Shape (batch_size, 5, latent_dim)
        
        return attended_features
        #print(attended_features.shape)
        # Generate output for each head
        #outputs = [head(attended_features) for head in self.head_outputs]
        
        #return torch.stack(outputs, dim=1)  # Shape (batch_size, 5, latent_dim)

if __name__ == "__main__":
    batch_size = 16
    input_dim = 358+24+24*23  # 1 common info + 5 body parts (assuming each is a scalar)
    latent_dim = 64  # Dimension of the latents for each head

    layer = PartAttentionLayer()#input_dim, latent_dim)
    input_tensor = torch.randn(batch_size, input_dim)
    output_tensor = layer(input_tensor)

    #print("Output shape:", output_tensor.shape)