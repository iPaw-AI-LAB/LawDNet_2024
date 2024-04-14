class ModifiedCrossAttention(nn.Module):
    def __init__(self, feature_dim, num_heads=1):
        super(ModifiedCrossAttention, self).__init__()
        self.num_heads = num_heads
        self.feature_dim_per_head = feature_dim // num_heads

        self.query_projection = nn.Linear(feature_dim, feature_dim)
        self.key_projection = nn.Linear(feature_dim, feature_dim)
        self.value_projection = nn.Linear(feature_dim, feature_dim)

        self.softmax = nn.Softmax(dim=-1)

    def forward(self, queries, keys, values):
        batch_size = queries.size(0)

        # Project queries, keys, and values
        queries = self.query_projection(queries).view(batch_size, -1, self.num_heads, self.feature_dim_per_head).transpose(1, 2)
        keys = self.key_projection(keys).view(batch_size, -1, self.num_heads, self.feature_dim_per_head).transpose(1, 2)
        values = self.value_projection(values).view(batch_size, -1, self.num_heads, self.feature_dim_per_head).transpose(1, 2)

        # Calculate attention scores and apply softmax
        attention_scores = torch.matmul(queries, keys.transpose(-2, -1)) / math.sqrt(self.feature_dim_per_head)
        attention_weights = self.softmax(attention_scores)

        # Apply attention weights to values
        weighted_values = torch.matmul(attention_weights, values)

        # Reshape weighted values back to original shape
        weighted_values = weighted_values.transpose(1, 2).contiguous().view(batch_size, -1, self.feature_dim_per_head * self.num_heads)

        return weighted_values

# 假设 img_para 和 audio_para 是未经全局平均池化的特征，具有更高的维度
# modified_cross_attention = ModifiedCrossAttention(feature_dim=你的特征维度, num_heads=你选择的头数)

# 用法示例，需要根据实际情况调整特征的形状
# img_para_flattened = img_para.view(batch_size, -1, feature_dim)  # 假设 img_para 是 [batch_size, C, H, W]
# audio_para_flattened = audio_para.view(batch_size, -1, feature_dim)  # 假设 audio_para 是 [batch_size, C, L]
# trans_para = modified_cross_attention(queries=img_para_flattened, keys=audio_para_flattened, values=img_para_flattened)
