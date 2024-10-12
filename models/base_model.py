import torch
import torch.nn as nn
import torch.nn.functional as F

# (3) class GLU(nn.Module): Gated unit: Function: 1. Sequence depth modeling 2. Reduce gradient dispersion and accelerate convergence
class GLU(nn.Module):
    def __init__(self, input_channel, output_channel):
        super(GLU, self).__init__()
        self.linear_left = nn.Linear(input_channel, output_channel)
        self.linear_right = nn.Linear(input_channel, output_channel)

    def forward(self, x):
        return torch.mul(self.linear_left(x), torch.sigmoid(self.linear_right(x)))

# (2) class StockBlockLayer(nn.Module):
class StockBlockLayer(nn.Module):
    def __init__(self, time_step, unit, multi_layer, stack_cnt=0):
        super(StockBlockLayer, self).__init__()
        self.time_step = time_step
        self.unit = unit
        self.stack_cnt = stack_cnt
        self.multi = multi_layer
        self.weight = nn.Parameter(
            torch.Tensor(1, 3 + 1, 1, self.time_step * self.multi, 
                         self.multi * self.time_step))  # [K+1, 1, in_c, out_c] [1, 4, 1, 12*5, 5*12]
        nn.init.xavier_normal_(self.weight) # normal初始化

        self.forecast = nn.Linear(self.time_step * self.multi, self.time_step * self.multi) # 输入 12*5 输出 12*5
        self.forecast_result = nn.Linear(self.time_step * self.multi, self.time_step) # 输入 12*5 输出 12
        if self.stack_cnt == 0:
            self.backcast = nn.Linear(self.time_step * self.multi, self.time_step) # 一个全链接层将 12*5 转换成 12 
        self.backcast_short_cut = nn.Linear(self.time_step, self.time_step) # 数据原特征表达
        self.relu = nn.ReLU()
        self.GLUs = nn.ModuleList()
        self.output_channel = 4 * self.multi
        for i in range(3):
            if i == 0:
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * 4, self.time_step * self.output_channel))
            elif i == 1:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
            else:
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                self.GLUs.append(GLU(self.time_step * self.output_channel, self.time_step * self.output_channel))
                # self.time_step * self.output_channel = 12 * (4 * 5) = 240

    def spe_seq_cell(self, input):
        # input shape: [32,4,1,140,12]
        batch_size, k, input_channel, node_cnt, time_step = input.size()
        input = input.view(batch_size, -1, node_cnt, time_step)
        # fft: 快速离散傅立叶变换， rfft：去除那些共轭对称的值，减小存储
        ffted = torch.rfft(input, 1, onesided=False)
        # ffted.shape:[32,4,140,12,2] 2:实，虚
        real = ffted[..., 0].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1) #[32,140,4*12]
        img = ffted[..., 1].permute(0, 2, 1, 3).contiguous().reshape(batch_size, node_cnt, -1)

        # GLU
        for i in range(3):
            real = self.GLUs[i * 2](real) # 0，2，4
            img = self.GLUs[2 * i + 1](img) # 1，3，5
        # real shape: [32,140,240]
        real = real.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        img = img.reshape(batch_size, node_cnt, 4, -1).permute(0, 2, 1, 3).contiguous()
        # real shape: [32,4,140,60]
        time_step_as_inner = torch.cat([real.unsqueeze(-1), img.unsqueeze(-1)], dim=-1)
        # real shape: [32,4,140,60,2]

        # IDFT
        iffted = torch.irfft(time_step_as_inner, 1, onesided=False)
        # [32,4,140,60]
        return iffted

    def forward(self, x, mul_L):
        mul_L = mul_L.unsqueeze(1)
        # mul_L.shape:[4,1,140,140]
        x = x.unsqueeze(1)
        # x.shape:[32, 1, 1, 140, 12]

        # Learning latent representaations of multipul time-series in the spetral
        gfted = torch.matmul(mul_L, x) # 输入的X和mul_L相乘 torch.matmul支持广播机制
        # gfted.shape:[32,4,1,140,12]

        # Capture the repeated patterns inthe periodic data, or the auto-correlation features aong different timestamps
        gconv_input = self.spe_seq_cell(gfted).unsqueeze(2)
        # gconv_input shape: [32,4,1,140,60]

        # GConv + IGFT
        # weight: torch.Size([1, 3 + 1, 1, self.time_step * self.multi, self.multi * self.time_step])
        # weight: [1, 4, 1, 12*5, 5*12]
        igfted = torch.matmul(gconv_input, self.weight)
        # igfted shape: [32,4,1,140,60]
        igfted = torch.sum(igfted, dim=1)
        # igfted shape: [32,1,140,60]

        # Forecast
        forecast_source = torch.sigmoid(self.forecast(igfted).squeeze(1))
        forecast = self.forecast_result(forecast_source)
        # forcast_source shape [32,140,60] and forcast shape [32,140,12]

        # Backcast
        if self.stack_cnt == 0:
            # x shape: [32,1,1,140,12]
            backcast_short = self.backcast_short_cut(x).squeeze(1)
            # backcast_short shape: [32,1,140,12]
            backcast_source = torch.sigmoid(self.backcast(igfted) - backcast_short)
        else:
            backcast_source = None
            # backcast_source shape: [32,1,140,12] or None
        return forecast, backcast_source

# (1) class Model(nn.Module): Initialization section
class Model(nn.Module):
    def __init__(self, units, stack_cnt, time_step, multi_layer, horizon=1, dropout_rate=0.5, leaky_rate=0.2,
                 device='cpu'):
        super(Model, self).__init__()
        # the dimension of the features
        self.unit = units # The input dimension, The output dimension is also 140, because we need to calculate a 140*140 matrix
        self.stack_cnt = stack_cnt # Number of blocks for StemGNNBlock (2)
        self.unit = units
        self.alpha = leaky_rate
        self.time_step = time_step # Reference window size, is the length of input sequence
        self.horizon = horizon # Prediction window size, H
        # Self-attention
        # For the last hidden state r of gru, use the Self-attention method to calculate the adjacency matrix
        self.weight_key = nn.Parameter(torch.zeros(size=(self.unit, 1))) # K: 140*1
        nn.init.xavier_uniform_(self.weight_key.data, gain=1.414)
        self.weight_query = nn.Parameter(torch.zeros(size=(self.unit, 1))) # Q: 140*1 #############
        nn.init.xavier_uniform_(self.weight_query.data, gain=1.414)
        # Call the GRU model directly
        self.GRU = nn.GRU(self.time_step, self.unit) # self.time_step Input dimension 12 * self.unit Output dimension 140
        self.multi_layer = multi_layer
        self.stock_block = nn.ModuleList()
        # Initialize two StemGNNBlocks
        self.stock_block.extend(
            [StockBlockLayer(self.time_step, self.unit, self.multi_layer, stack_cnt=i) for i in range(self.stack_cnt)]) #################
        self.fc = nn.Sequential(
            nn.Linear(int(self.time_step), int(self.time_step)), # time_step 12
            nn.LeakyReLU(),
            nn.Linear(int(self.time_step), self.horizon), # horizon 3 
        )
        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout = nn.Dropout(p=dropout_rate)
        self.to(device)

    def get_laplacian(self, graph, normalize):
        """
        return the laplacian of the graph.
        :param graph: the graph structure without self loop, [N, N].
        :param normalize: whether to used the normalized laplacian.
        :return: graph laplacian.
        """
        if normalize:
            D = torch.diag(torch.sum(graph, dim=-1) ** (-1 / 2))
            L = torch.eye(graph.size(0), device=graph.device, dtype=graph.dtype) - torch.mm(torch.mm(D, graph), D)
        else:
            D = torch.diag(torch.sum(graph, dim=-1))
            L = D - graph
        return L
    
    # Returns the Chebyshev polynomial
    def cheb_polynomial(self, laplacian):
        """
        Compute the Chebyshev Polynomial, according to the graph laplacian.
        :param laplacian: the graph laplacian, [N, N].
        :return: the multi order Chebyshev laplacian, [K, N, N].
        """
        N = laplacian.size(0)  # [N, N]
        laplacian = laplacian.unsqueeze(0)
        first_laplacian = torch.zeros([1, N, N], device=laplacian.device, dtype=torch.float)
        second_laplacian = laplacian
        third_laplacian = (2 * torch.matmul(laplacian, second_laplacian)) - first_laplacian
        forth_laplacian = 2 * torch.matmul(laplacian, third_laplacian) - second_laplacian
        multi_order_laplacian = torch.cat([first_laplacian, second_laplacian, third_laplacian, forth_laplacian], dim=0)
        return multi_order_laplacian
    
    # The following part processes the data, then uses GRU for temporal learning, and then obtains attention through the self-attention mechanism (self_graph_attention)
    def latent_correlation_layer(self, x):
        # The original input of [32, 12, 140] becomes [140, 32, 12] data [sequence, batch, features]
        # Because we want to use GRU to learn time series, the final dimension should be the time series
        input, _ = self.GRU(x.permute(2, 0, 1).contiguous()) 
        # Output Matrix torch.Size([140, 32, 140]) [sequence, batch, D*Hout(output_size)]
        input = input.permute(1, 0, 2).contiguous()
        # Next call this model function to get attention
        attention = self.self_graph_attention(input)
        ###### input: attention.shape:[32,140,140]
        attention = torch.mean(attention, dim=0)
        # attention.shape:[140,140]
        degree = torch.sum(attention, dim=1)
        # degree.shape: torch.Size([140])
        # laplacian is sym or not
        attention = 0.5 * (attention + attention.T) # Turn it into a symmetric lap matrix
        degree_l = torch.diag(degree)
        # degree.shape:[140]
        # laplacian is sym or not
        # Next, generate a symmetric attention
        diagonal_degree_hat = torch.diag(1 / (torch.sqrt(degree) + 1e-7))
        # Returns a two-dimensional matrix with the inverse of degree as the diagonal, torch.Size([140，140])
        laplacian = torch.matmul(diagonal_degree_hat,
                                 torch.matmul(degree_l - attention, diagonal_degree_hat))
        # Take square root of the inverse matrix of degree
        mul_L = self.cheb_polynomial(laplacian)
        # Use Chebyshev's formula to get the fourth-order Laplace matrix
        ###### The dimension of mul_L is [4, 140, 140]
        # Return to the forward function
        return mul_L, attention

    def self_graph_attention(self, input):
        # The original input of [batch, sequence, output_size(features)] becomes [batch, output_size, sequence]
        input = input.permute(0, 2, 1).contiguous()
        bat, N, fea = input.size() 
        ###### 32,140,140
        key = torch.matmul(input, self.weight_key) 
        ###### input 32,140,140 * self.weight_key 140*1 = 32,140,1
        # self.weight_key.shape:[140,1]
        # key.shape:[32,140,1]
        query = torch.matmul(input, self.weight_query)
        # Repeat is to multiply the length of each corresponding dimension by a multiple. The following is to multiply the third dimension by N
        data = key.repeat(1, 1, N).view(bat, N * N, 1) + query.repeat(1, N, 1)
        # key:  It becomes the dimension of [32,140,140]
        # view: data.shape:[32,140*140,1]
        # query:  It becomes the dimension of [32,140*140,1]
        data = data.squeeze(2)
        data = data.view(bat, N, -1)
        data = self.leakyrelu(data)
        # data.shape:[32,140,140]
        attention = F.softmax(data, dim=2)
        attention = self.dropout(attention)
        ###### attention.shape:[32,140,140]
        # Finally, we get attention: a matrix representing the relationship between features i and j
        return attention

    def graph_fft(self, input, eigenvectors):
        return torch.matmul(eigenvectors, input)

    def forward(self, x):
        # Part 1
        mul_L, attention = self.latent_correlation_layer(x) # input x
        X = x.unsqueeze(1).permute(0, 1, 3, 2).contiguous()
        # x: [batch, sequence, features] == X: [batch, 1, features, sequence]

        # Part 2
        result = []
        # Start entering the two-layer StockBlockLayer and pass in X and the fourth-order laplacian matrix
        for stack_i in range(self.stack_cnt): # 0；1
            # X Shape: torch.Size([32, 1, 140, 12])
            # Mul_L Shape: torch.Size([4, 140, 140])
            forecast, X = self.stock_block[stack_i](X, mul_L) # G = (X, Mul_L) in stemgnn
            # output forecast: predicted values == [32, 1, 140, 12]
            # output X: processed backcast == [32, 1, 140, 12]
            result.append(forecast)
        forecast = result[0] + result[1] # [32, 1, 140, 12]
        forecast = self.fc(forecast) # [32, 140, 12]
        if forecast.size()[-1] == 1: # Horizon = 1
            return forecast.unsqueeze(1).squeeze(-1), attention
            # forecast shape: [32, 140, 1] == [32, 1, 140]
        else: # Horizon = 3
            return forecast.permute(0, 2, 1).contiguous(), attention
            # forecast shape: [32, 140, 3] == [32, 3, 140]
