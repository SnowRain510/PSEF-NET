import torch
import torch.nn.functional as F
from torch import nn, optim
from src import UNet



import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision.transforms import transforms
from torchmeta.datasets import MiniImagenet
from torchmeta.utils.data import BatchMetaDataLoader
from torchmeta.transforms import Categorical, ClassSplitter
#from torchvision.models import UNet


# 定义UNet分割网络
class SegmentationUNet(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(SegmentationUNet, self).__init__()
        self.unet = UNet(in_channels,num_classes=5)

    def forward(self, x):
        return self.unet(x)


# 定义元学习网络（MAML）
class MetaLearner(nn.Module):
    def __init__(self, in_channels, out_channels):
        super(MetaLearner, self).__init__()
        self.unet = SegmentationUNet(in_channels, out_channels)

    def forward(self, x):
        return self.unet(x)


# 训练MAML和UNet分割网络
def train_metaseg(train_loader, model, meta_optimizer, seg_optimizer, device):
    model.train()
    for batch in train_loader:
        x, y = batch["input"], batch["target"]
        x = x.to(device)
        y = y.to(device)

        meta_optimizer.zero_grad()
        seg_optimizer.zero_grad()

        # 使用MAML进行元学习初始化
        meta_model = MetaLearner(in_channels, out_channels).to(device)
        meta_model.load_state_dict(model.state_dict())

        # 进行元学习微调
        for _ in range(num_inner_steps):
            logits = meta_model(x)
            loss = compute_loss(logits, y)
            gradients = torch.autograd.grad(loss, meta_model.parameters(), create_graph=True)
            meta_model.update_params(learning_rate, gradients)

        # 计算微调后的损失
        seg_logits = meta_model.unet(x)
！！！！！！！

class MetaUNet(nn.Module):
    def __init__(self, in_channels, out_channels, hidden_size=64,num_classes=5):
        super(MetaUNet, self).__init__()
        self.unet = UNet(in_channels,num_classes=5)
        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x, weights=None):
        if weights is None:
            weights = self.unet.parameters()

        out = self.unet(x, weights)
        return out

    def compute_loss(self, x, y, weights=None):
        pred = self.forward(x, weights)
        loss = self.loss_fn(pred, y)
        return loss

    def update_weights(self, loss, optimizer, lr):
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    def adapt(self, support_x, support_y, query_x, query_y, inner_lr=0.01, num_adapt_steps=5):
        """
        Perform inner loop adaptation on support and query set
        """
        # Create optimizer for inner loop adaptation
        optimizer = optim.SGD(self.unet.parameters(), lr=inner_lr)

        for step in range(num_adapt_steps):
            # Forward pass on support set
            support_loss = self.compute_loss(support_x, support_y)

            # Update weights with respect to support set
            self.update_weights(support_loss, optimizer, lr=inner_lr)

        # Compute loss on query set using adapted weights
        query_loss = self.compute_loss(query_x, query_y)

        return query_loss.item()

    def meta_update(self, support_x, support_y, query_x, query_y, meta_lr=0.001, num_meta_steps=5):
        """
        Perform outer loop meta-update on support and query set
        """
        # Create optimizer for outer loop meta-update
        meta_optimizer = optim.Adam(self.parameters(), lr=meta_lr)

        # Create a copy of the initial weights
        initial_weights = self.unet.parameters()

        for step in range(num_meta_steps):
            # Clone the model with the initial weights for a new meta-update iteration
            model_copy = MetaUNet(self.unet.in_channels, self.unet.out_channels, self.unet.hidden_size)

            # Compute loss on query set with initial weights
            query_loss = self.compute_loss(query_x, query_y, weights=initial_weights)

            # Compute gradients for the meta-update
            query_loss.backward()

            # Update the meta-parameters with the gradients
            for p, p_copy in zip(self.parameters(), model_copy.parameters()):
                if p.grad is not None:
                    p_copy.grad = p.grad.clone()

            # Perform a meta-update step on the model parameters
            meta_optimizer.step()

            # Reset the gradients for the next iteration
            meta_optimizer.zero_grad()

        # Compute loss on query set with updated weights
        final_query_loss = self.compute_loss(query_x, query_y)

        return final_query_loss.item()
