import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.cuda.amp import autocast as autocast

class FlattenAndLinear(nn.Module):
    def __init__(self,in_channels=128):
        super(FlattenAndLinear, self).__init__()
        # 假设输入数据的维度为 [batch_size, 10, 128, 256]
        # 我们首先通过一个卷积层来提取特征，这里使用一个简单的一维卷积
        self.conv1 = nn.Conv1d(in_channels=in_channels, out_channels=64, kernel_size=3, stride=1, padding=1)
        # 接着使用一个最大池化层来降低维度
        self.pool = nn.MaxPool1d(kernel_size=2, stride=2)
        # 然后通过一个全连接层将卷积层的输出映射到目标维度
        self.fc = nn.Linear(64 * 128, 256)

    def forward(self, x):
        # 通过卷积层和激活函数
        x = self.pool(F.relu(self.conv1(x)))
        # 将卷积层输出的维度从 [batch_size, 64, 64] 调整为 [batch_size, 64 * 64]
        x = x.view(x.size(0), -1)  # Flatten the tensor
        # 通过全连接层和激活函数
        x = F.relu(self.fc(x))
        return x


class ImprovedCrossAttentionModel(nn.Module):
    def __init__(self):
        super(ImprovedCrossAttentionModel, self).__init__()      
        # Positional Encoding for video features
        self.positional_encoding_video = nn.Parameter(torch.randn((128, 8, 6), dtype=torch.float32))
        self.positional_encoding_audio = nn.Parameter(torch.randn((128, 2), dtype=torch.float32))
        
        # Attention layers
        self.query_proj = nn.Linear(2, 256)  # Project audio features to match the dimensionality
        self.key_proj = nn.Linear(8 * 6, 256)    # Project video features after convolution and reshaping
        self.value_proj = nn.Linear(8 * 6, 256)  # Same as key projection
        
        # Output layers to ensure output dimension is 5*batchsize, 256
        self.final_linear = nn.Linear(256, 256)
        self.output_reshape = FlattenAndLinear(128)

    def forward(self, video_features, audio_features):
        with autocast():
            # video_features: (5*batch_size, 128, 8, 6)
            # audio_features: (5*batch_size, 128, 2)
            
            # Applying convolutional layer to video features
            video_features = video_features.view(-1, 128, 8, 6)  # Reshape to handle each frame independently
            # video_features = self.conv_layer(batched_frames)

            # print("video_features:",video_features.shape) # 10 128 8 6
            # print("self.positional_encoding:",self.positional_encoding_video.shape) # 128 8 6
            video_features = video_features + self.positional_encoding_video.expand(video_features.size(0), -1, -1, -1)  # Adding positional encoding

            # Reshape and maintain spatial structure for attention
            video_features = video_features.view(-1, 128, 8*6)  # Keep spatial dimensions together # 10 128 48
            # video_features = video_features.permute(0, 2, 1).contiguous().view(-1, 10*8*6)  # Reshape for key/value projection
            
            audio_features = audio_features.view(-1, 128, 2)  # Just ensure dimensions are correct # 10 128 2
            # print("audio_features:",audio_features.shape)
            audio_features = audio_features + self.positional_encoding_audio.expand(audio_features.size(0), -1, -1)  # Adding positional encoding
            
            # Project features for attention
            query = self.query_proj(audio_features) # 10 128 256
            key = self.key_proj(video_features)
            value = self.value_proj(video_features)

            # Scaled dot-product attention
            # print("query:",query.shape) # query: torch.Size([10, 128, 256])
            # print("key.shape:",key.shape) # key.shape: torch.Size([10, 256])
            # print("key.transpose(1, 2).shape:",key.transpose(1, 2).shape)

            attention_scores = torch.bmm(query, key.transpose(1, 2)) / (256 ** 0.5)
            attention = F.softmax(attention_scores, dim=-1)
            attended_features = torch.bmm(attention, value)
            
            # Post-attention processing to match required output dimensions
            attended_features = self.final_linear(attended_features)
            # print("attended_features:",attended_features.shape)

            output = self.output_reshape(attended_features)
            
            return output


if __name__ == "__main__":
    improved_model = ImprovedCrossAttentionModel()
    video_features = torch.randn(5*2, 128, 8, 6)
    audio_features = torch.randn(5*2, 128, 2)
    output = improved_model(video_features, audio_features)
    # print(output.shape)  # Expected shape: (5*batch_size, 256)
