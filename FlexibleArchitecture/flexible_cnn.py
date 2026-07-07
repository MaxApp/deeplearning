import torch
import torch.nn as nn

# Base CNN block build with dynamic parameters.
# The block composed of Conv2d, BatchNorm2d, ReLU, MaxPool layers
class ConvBlock(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, padding, pool_kernel_size, pool_stride):
        super().__init__()
        
        self.conv_layer = nn.Sequential(
            nn.Conv2d(in_channels,out_channels,kernel_size=kernel_size, padding=padding, stride=1),
            # Batch normalization
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.MaxPool2d(pool_kernel_size, pool_stride)
        )

    def forward(self, x):
        return self.conv_layer(x)
    
# The fc layers block
class FCBlock(nn.Module):

    def __init__(self, dropout_rate, in_features, out_features) -> None:
        super().__init__()
        # dropout + fc layer
        self.fc_layer = nn.Sequential(
            nn.Dropout(dropout_rate),
            nn.Linear(in_features, out_features),
            nn.BatchNorm1d(out_features),
            nn.ReLU()
        )

    def forward(self, x):
        return self.fc_layer(x)

    
# Flexible CNN composed of dynamic blocks
class FlexibleCNN(nn.Module):

    def __init__(self, num_of_fiters_per_conv_layer:list[int], kernel_size_per_layer:list[int], 
                 paddings_per_layer:list[int], max_pool_size_per_layer:list[int], max_pool_stride_per_layer:list[int],
                 dropout_rate:list[float], num_of_fc_neurous:list[int], num_of_classes:int) -> None:
        super().__init__()
        
        # TODO: check the size of the associated parameters lists are identical.

        self.dropout_rate = dropout_rate
        self.num_of_fc_neurous = num_of_fc_neurous
        self.num_of_classes = num_of_classes

        # feature layers
        self.conv_layers = []
        
        # number of conv layers
        num_of_conv_layers = len(num_of_fiters_per_conv_layer)

        # input layer is original image with RGB
        in_channels = 3

        for i in range(num_of_conv_layers):
            conv_layer = ConvBlock(in_channels, 
                                   num_of_fiters_per_conv_layer[i], 
                                   kernel_size=kernel_size_per_layer[i], 
                                   padding=paddings_per_layer[i],
                                   pool_kernel_size=max_pool_size_per_layer[i],
                                   pool_stride=max_pool_stride_per_layer[i])
            self.conv_layers.append(conv_layer)

            # set the next conv input_channels
            in_channels = num_of_fiters_per_conv_layer[i]

    def forward(self, x):
        for i in self.conv_layers:
            x = self.conv_layers[i](x)
        
        flattened_x = torch.flatten(x, 1) # dim=1

        # construct fc layers dynamically
        # Caveat: ONLY create structures for the first time
        if self.fc_layers is None:
            # number of fc layers
            num_of_fc_layers = len(self.num_of_fc_neurous)
            fc_in_channels = flattened_x.size(1)
            self.fc_layers = []
            for i in range(num_of_fc_layers):
                fc_out_channels = self.num_of_fc_neurous[i]
                fc_layer = FCBlock(self.dropout_rate[i], fc_in_channels, fc_out_channels)
                self.fc_layers.append(fc_layer)
                fc_in_channels = fc_out_channels
            
            # init final output layer
            self.final_layer = nn.Sequential(
                nn.Dropout(p=0.5),
                nn.Linear(fc_in_channels, self.num_of_classes)
            )
        
        for fc_layer in self.fc_layers:
            flattened_x = fc_layer(flattened_x)

        # last classification layer
        return self.final_layer(flattened_x)



if "__main__" == __name__:
    num_of_fiters_per_conv_layer = [6,3]
    kernel_size_per_layer = [5, 3]
    paddings_per_layer = [2, 1]
    max_pool_size_per_layer = [2,2]
    max_pool_stride_per_layer = [1,1]
    dropout_rate = [0.6, 0.5]
    num_of_fc_neurous = [15, 10]
    num_of_classes = 5
    model = FlexibleCNN(num_of_fiters_per_conv_layer,
                        kernel_size_per_layer,
                        paddings_per_layer,
                        max_pool_size_per_layer,
                        max_pool_stride_per_layer,
                        dropout_rate,
                        num_of_fc_neurous,
                        num_of_classes)
    
    print(model)