import torch.nn as nn
import torch
import torch.nn.functional as F


class ConvLSTMCell(nn.Module):

    def __init__(self, input_dim, hidden_dim, kernel_size, bias):
        """
        Initialize ConvLSTM cell.

        Parameters
        ----------
        input_dim: int
            Number of channels of input tensor.
        hidden_dim: int
            Number of channels of hidden state.
        kernel_size: (int, int)
            Size of the convolutional kernel.
        bias: bool
            Whether to add the bias.
        """

        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        height, width = image_size
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device))


class ConvLSTM(nn.Module):
    """

    Parameters:
        input_dim: Number of channels in input
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.

    Input:
        A tensor of size B, T, C, H, W or T, B, C, H, W
    Output:
        A tuple of two lists of length num_layers (or length 1 if return_all_layers is False).
            0 - layer_output_list is the list of lists of length T of each output
            1 - last_state_list is the list of last states
                    each element of the list is a tuple (h, c) for hidden state and memory
    Example:
        >> x = torch.rand((32, 10, 64, 128, 128))
        >> convlstm = ConvLSTM(64, 16, 3, 1, True, True, False)
        >> _, last_states = convlstm(x)
        >> h = last_states[0][0]  # 0 for layer index, 0 for h index
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, num_layers,
                 new_channel, new_seq_len,
                 batch_first=False, bias=True, return_all_layers=False):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)
        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.kernel_size = kernel_size
        self.num_layers = num_layers
        self.batch_first = batch_first
        self.bias = bias
        self.return_all_layers = return_all_layers

        cell_list = []
        for i in range(0, self.num_layers):
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]

            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

        self.mod_layer = ResizeAndSelectLastK(k=new_seq_len)
        self.final_conv = nn.Conv2d(in_channels=self.hidden_dim[0],
                                    out_channels=new_channel,
                                    kernel_size=(1, 1),
                                    bias=self.bias)

    def forward(self, input_tensor, hidden_state=None):
        """

        Parameters
        ----------
        input_tensor: todo
            5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful

        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        b, _, _, h, w = input_tensor.size()

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            hidden_state = self._init_hidden(batch_size=b,
                                             image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :],
                                                 cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            # Apply final_conv only after the last layer
            if layer_idx == self.num_layers - 1:
                # Before applying self.final_conv
                batch_size, seq_len, channels, height, width = layer_output.size()
                layer_output = layer_output.view(batch_size * seq_len, channels, height, width)
                # Apply the convolution
                layer_output = self.final_conv(layer_output)
                layer_output = torch.sigmoid(layer_output)
                # Reshape back if needed
                layer_output = layer_output.view(batch_size, seq_len, -1, height,
                                                 width)  # adjust -1 according to your new number of channels

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

        if not self.return_all_layers:
            layer_output_list = layer_output_list[-1:]
            last_state_list = last_state_list[-1:]

        resized_and_selected_output = self.mod_layer(layer_output_list)

        return resized_and_selected_output[-1]

    def _init_hidden(self, batch_size, image_size):
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


class ResizeAndSelectLastK(nn.Module):
    def __init__(self, k):
        super(ResizeAndSelectLastK, self).__init__()
        self.k = k

    def forward(self, layer_output_list):
        # Process each tensor in the list
        processed_outputs = []
        for output in layer_output_list:
            pooled_output = output[:, -self.k:,:,:,:]
            # B, T, C, H, W = output.shape
            # pooled_output = torch.zeros(B, self.k, C, H, W, device=output.device)
            #
            # # Calculate the number of original time steps to combine for one output time step
            # step = T / self.k
            # # Average the time steps for each output time step
            # for i in range(self.k):
            #     start_idx = int(i * step)
            #     end_idx = int((i + 1) * step)
            #     pooled_output[:, i, :, :, :] = output[:, start_idx:end_idx, :, :, :].mean(dim=1)

            # output 0~1
            processed_outputs.append(pooled_output)

        return processed_outputs


# ------------------ 3D U-Net ----------------------

class Conv3DBlock(nn.Module):
    """
    The basic block for double 3x3x3 convolutions in the analysis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> desired number of output channels
    :param bottleneck -> specifies the bottlneck block
    -- forward()
    :param input -> input Tensor to be convolved
    :return -> Tensor
    """

    def __init__(self, in_channels, out_channels, bottleneck=False) -> None:
        super(Conv3DBlock, self).__init__()
        self.conv1 = nn.Conv3d(in_channels=in_channels, out_channels=out_channels // 2, kernel_size=(3, 3, 3),
                               padding=1)
        self.bn1 = nn.BatchNorm3d(num_features=out_channels // 2)
        self.conv2 = nn.Conv3d(in_channels=out_channels // 2, out_channels=out_channels, kernel_size=(3, 3, 3),
                               padding=1)
        self.bn2 = nn.BatchNorm3d(num_features=out_channels)
        self.relu = nn.ReLU()
        self.bottleneck = bottleneck
        if not bottleneck:
            self.pooling = nn.MaxPool3d(kernel_size=(2, 2, 2), stride=2)

    def forward(self, input):
        res = self.relu(self.bn1(self.conv1(input)))
        res = self.relu(self.bn2(self.conv2(res)))
        None
        if not self.bottleneck:
            out = self.pooling(res)
        else:
            out = res
        return out, res


class UpConv3DBlock(nn.Module):
    """
    The basic block for upsampling followed by double 3x3x3 convolutions in the synthesis path
    -- __init__()
    :param in_channels -> number of input channels
    :param out_channels -> number of residual connections' channels to be concatenated
    :param last_layer -> specifies the last output layer
    :param num_classes -> specifies the number of output channels for dispirate classes
    -- forward()
    :param input -> input Tensor
    :param residual -> residual connection to be concatenated with input
    :return -> Tensor
    """

    def __init__(self, in_channels, res_channels=0, last_layer=False, num_classes=None) -> None:
        super(UpConv3DBlock, self).__init__()
        assert (last_layer == False and num_classes == None) or (
                last_layer == True and num_classes != None), 'Invalid arguments'
        self.upconv1 = nn.ConvTranspose3d(in_channels=in_channels, out_channels=in_channels, kernel_size=(2, 2, 2),
                                          stride=2)
        self.relu = nn.ReLU()
        self.bn = nn.BatchNorm3d(num_features=in_channels // 2)
        self.conv1 = nn.Conv3d(in_channels=in_channels + res_channels, out_channels=in_channels // 2,
                               kernel_size=(3, 3, 3), padding=(1, 1, 1))
        self.conv2 = nn.Conv3d(in_channels=in_channels // 2, out_channels=in_channels // 2, kernel_size=(3, 3, 3),
                               padding=(1, 1, 1))
        self.last_layer = last_layer
        if last_layer:
            self.conv3 = nn.Conv3d(in_channels=in_channels // 2, out_channels=num_classes, kernel_size=(1, 1, 1))

    def forward(self, input, residual=None):
        out = self.upconv1(input)
        if residual != None: out = torch.cat((out, residual), 1)
        out = self.relu(self.bn(self.conv1(out)))
        out = self.relu(self.bn(self.conv2(out)))
        if self.last_layer: out = self.conv3(out)
        return out


class UNet3D(nn.Module):
    """
    The 3D UNet model
    -- __init__()
    :param in_channels -> number of input channels
    :param num_classes -> specifies the number of output channels or masks for different classes
    :param level_channels -> the number of channels at each level (count top-down)
    :param bottleneck_channel -> the number of bottleneck channels
    :param device -> the device on which to run the model
    -- forward()
    :param input -> input Tensor
    :return -> Tensor
    """

    def __init__(self, in_channels, num_classes, level_channels=[64, 128, 256], bottleneck_channel=512) -> None:
        super(UNet3D, self).__init__()
        level_1_chnls, level_2_chnls, level_3_chnls = level_channels[0], level_channels[1], level_channels[2]
        self.a_block1 = Conv3DBlock(in_channels=in_channels, out_channels=level_1_chnls)
        self.a_block2 = Conv3DBlock(in_channels=level_1_chnls, out_channels=level_2_chnls)
        self.a_block3 = Conv3DBlock(in_channels=level_2_chnls, out_channels=level_3_chnls)
        self.bottleNeck = Conv3DBlock(in_channels=level_3_chnls, out_channels=bottleneck_channel, bottleneck=True)
        self.s_block3 = UpConv3DBlock(in_channels=bottleneck_channel, res_channels=level_3_chnls)
        self.s_block2 = UpConv3DBlock(in_channels=level_3_chnls, res_channels=level_2_chnls)
        self.s_block1 = UpConv3DBlock(in_channels=level_2_chnls, res_channels=level_1_chnls, num_classes=num_classes,
                                      last_layer=True)

    def forward(self, input):
        # change dim
        input = input.permute([0, 2, 1, 3, 4])  # -> [b, c, t, w, h]
        # Analysis path forward feed
        out, residual_level1 = self.a_block1(input)
        out, residual_level2 = self.a_block2(out)
        out, residual_level3 = self.a_block3(out)
        out, _ = self.bottleNeck(out)

        # Synthesis path forward feed
        out = self.s_block3(out, residual_level3)
        out = self.s_block2(out, residual_level2)
        out = self.s_block1(out, residual_level1)

        # # average to reduce the 3rd dimension to 1
        # out = out.mean(dim=2, keepdim=True)  # Averages across the depth dimension
        # out = out.squeeze(dim=2)

        out = torch.sigmoid(out)

        return out


# ------------------

def test_model_output(model, input_tensor, ground_truth):
    """
    Test the model output and ground truth for size, value range, and class validity.
    :param model: The ConvLSTM model instance.
    :param input_tensor: A sample input tensor for the model.
    :param ground_truth: The ground truth data for comparison.
    """
    # Forward pass
    output, _ = model(input_tensor)

    # Check the size of the output
    print("Model Output Size:", output.shape)

    # Applying softmax to convert logits to probabilities
    probabilities = F.softmax(output, dim=2)

    # Check value range in model output
    min_prob, max_prob = probabilities.min(), probabilities.max()
    print("Minimum Probability in Model Output:", min_prob.item())
    print("Maximum Probability in Model Output:", max_prob.item())

    # Check if the probabilities are between 0 and 1
    if 0 <= min_prob.item() <= 1 and 0 <= max_prob.item() <= 1:
        print("Value range in Model Output is valid.")
    else:
        print("Value range in Model Output is invalid.")

    # Check for class validity in model output (assuming 3 classes)
    _, predicted_classes = probabilities.max(2)
    unique_classes_model = torch.unique(predicted_classes)
    print("Unique predicted classes in Model Output:", unique_classes_model)
    if all(0 <= uc < 3 for uc in unique_classes_model):
        print("Class check in Model Output passed.")
    else:
        print("Invalid classes found in Model Output.")

    # Check Ground Truth
    print("Ground Truth Size:", ground_truth.shape)
    unique_classes_gt = torch.unique(ground_truth)
    print("Unique classes in Ground Truth:", unique_classes_gt)
    if all(0 <= uc < 3 for uc in unique_classes_gt):
        print("Class check in Ground Truth passed.")
    else:
        print("Invalid classes found in Ground Truth.")

    return output


if __name__ == "__main__":
    # Define the input dimensions
    batch_size = 4
    time_seq = 256
    num_channels = 1
    height = 16
    width = 16

    # Number of classes for output (modify as needed)
    n_classes = 10

    # Create a dummy input tensor with the specified dimensions
    x = torch.randn(batch_size, time_seq, num_channels, height, width)

    input_dim = 1
    hidden_dim = 64
    output_dim = 2
    output_tl = 1
    kernel_size = (3, 3)
    num_layers = 2
    height = 16  # Example height, adjust as needed
    width = 16  # Example width, adjust as needed
    bias = True

    # Initialize the UNet model
    model = ConvLSTM(input_dim=input_dim,
                     hidden_dim=hidden_dim,
                     new_channel=output_dim,
                     new_seq_len=output_tl,
                     kernel_size=kernel_size,
                     num_layers=num_layers,
                     batch_first=True)

    print("input shape:", x.shape)
    # Pass the input tensor through the model
    output = model(x)

    # Print the output shape
    print("Output shape:", output.shape)
    print("Output max value:", output.max().item())
    print("Output min value:", output.min().item())
