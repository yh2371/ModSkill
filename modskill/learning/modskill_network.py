import torch
import torch.nn as nn

from phc.learning.network_builder import NetworkBuilder
from phc.learning.attention import PartAttentionLayer
from rl_games.algos_torch import torch_ext


class ModSkillNetwork(NetworkBuilder.BaseNetwork):

    def __init__(self, mlp_args, output_size=69, partitioning=[4, 4, 5, 5, 5], task=False, checkpoint_path=None):
        super().__init__()

        self.partitioning = partitioning
        self.task = task
        self.checkpoint_path = checkpoint_path
        
        self._build_extractor(self.partitioning, self.task, mlp_args)
        self._build_actors(mlp_args)
        if self.task:
            self.load_base_net(self.checkpoint_path, 5)
            self.freeze_pnn(5)
            

    def freeze_pnn(self, idx):
        for param in self.actors[:idx].parameters():
            param.requires_grad = False

    def load_base_net(self, model_path, actors=1):
        checkpoint = torch_ext.load_checkpoint(model_path)
        for idx in range(actors):
            self.load_actor2(checkpoint, idx)

    def load_actor2(self, checkpoint, idx=0):
        state_dict = self.actors[idx].state_dict()
        state_dict['0.weight'].copy_(checkpoint['model'][f'a2c_network.pnn.actors.{idx}.0.weight'])
        state_dict['0.bias'].copy_(checkpoint['model'][f'a2c_network.pnn.actors.{idx}.0.bias'])
        state_dict['2.weight'].copy_(checkpoint['model'][f'a2c_network.pnn.actors.{idx}.2.weight'])
        state_dict['2.bias'].copy_(checkpoint['model'][f'a2c_network.pnn.actors.{idx}.2.bias'])
        state_dict['4.weight'].copy_(checkpoint['model'][f'a2c_network.pnn.actors.{idx}.4.weight'])
        state_dict['4.bias'].copy_(checkpoint['model'][f'a2c_network.pnn.actors.{idx}.4.bias'])
        state_dict['6.weight'].copy_(checkpoint['model'][f'a2c_network.pnn.actors.{idx}.6.weight'])
        state_dict['6.bias'].copy_(checkpoint['model'][f'a2c_network.pnn.actors.{idx}.6.bias'])
        #print("state_dict", state_dict)

        

    def _build_extractor(self, partitioning, task, mlp_args):
        if self.task: 
            self.extractor = self._build_sequential_mlp(64 * 5, **mlp_args)   
        else:     
            obs_size1 = {'L_Hip':24, 'L_Knee':24, 'L_Ankle':24, 'L_Toe':24, 'R_Hip':24, 'R_Knee':24, 'R_Ankle':24, 'R_Toe':24, 'Torso':48, 'Spine':24, 'Chest':24, 'Neck':24, 'Head':24, 'L_Thorax':24, 'L_Shoulder':24, 'L_Elbow':24, 'L_Wrist':24, 'L_Hand':24, 'R_Thorax':24, 'R_Shoulder':24, 'R_Elbow':24, 'R_Wrist':24, 'R_Hand':24}
            obs_size2 = {'L_Hip':15, 'L_Knee':15, 'L_Ankle':15, 'L_Toe':15, 'R_Hip':15, 'R_Knee':15, 'R_Ankle':15, 'R_Toe':15, 'Torso':1+12+15, 'Spine':15, 'Chest':15, 'Neck':15, 'Head':15, 'L_Thorax':15, 'L_Shoulder':15, 'L_Elbow':15, 'L_Wrist':15, 'L_Hand':15, 'R_Thorax':15, 'R_Shoulder':15, 'R_Elbow':15, 'R_Wrist':15, 'R_Hand':15}

            start = 0
            input_dim1 = []
            input_dim2 = []
            for i in range(len(partitioning)):
                end = start + partitioning[i]
                input_dim1.append(sum(list(obs_size1.values())[start:end]))
                input_dim2.append(sum(list(obs_size2.values())[start:end]))
                start = end
            self.extractor = PartAttentionLayer(input_dim1, input_dim2)

    def _build_actors(self, mlp_args):
        self.actors = nn.ModuleList()
        self.dof = {'L_Hip':3, 'L_Knee':3, 'L_Ankle':3, 'L_Toe':3, 'R_Hip':3, 'R_Knee':3, 'R_Ankle':3, 'R_Toe':3, 'Torso':3, 'Spine':3, 'Chest':3, 'Neck':3, 'Head':3, 'L_Thorax':3, 'L_Shoulder':3, 'L_Elbow':3, 'L_Wrist':3, 'L_Hand':3, 'R_Thorax':3, 'R_Shoulder':3, 'R_Elbow':3, 'R_Wrist':3, 'R_Hand':3}
        start = 0
        output_sizes = []
        for i, part in enumerate(self.partitioning):
            end = start + part
            output_sizes.append(sum(list(self.dof.values())[start:end]))
            self.actors.append(self._build_sequential_mlp(output_sizes[i], latent_size=64, **mlp_args))
            start = end
        print(output_sizes)

    def _build_sequential_mlp(
        self,
        actions_num,
        units,
        activation,
        dense_func,
        input_size=64,
        latent_size=None,
        norm_only_first_layer=False,
        norm_func_name=None,
        need_norm=True,
    ):
        layers = []
        if latent_size is not None:
            in_size = latent_size
        else:
            in_size = input_size
        for unit in units:
            layers.append(dense_func(in_size, unit))
            layers.append(self.activations_factory.create(activation))

            if need_norm:
                if norm_only_first_layer and norm_func_name is not None:
                    need_norm = False
                if norm_func_name == "layer_norm":
                    layers.append(nn.LayerNorm(unit))
                elif norm_func_name == "batch_norm":
                    layers.append(nn.BatchNorm1d(unit))

            in_size = unit

        layers.append(nn.Linear(units[-1], actions_num))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = self.extractor(x)  # (bs, parts=5, latent=64)
        features = torch.nn.functional.normalize(features, dim=-1)

        actions = [self.actors[i](features[:, i, :]) for i in range(len(self.partitioning))]
        actions = torch.cat(actions, dim=1)
        return actions, [actions]
