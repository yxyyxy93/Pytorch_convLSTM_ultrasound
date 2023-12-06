import torch.nn as nn
import torch
import torch.nn.functional as F

__all__ = [
    "ConvLSTM"
]


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
        # Add batch normalization layer
        self.batch_norm = nn.BatchNorm2d(4 * self.hidden_dim)

    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state
        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis
        combined_conv = self.conv(combined)

        # Apply batch normalization
        combined_conv = self.batch_norm(combined_conv)

        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next

    def init_hidden(self, batch_size, image_size):
        """
        :param batch_size:
        :param image_size:
        :return:
        """
        height, width = image_size
        init_h = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        init_c = torch.zeros(batch_size, self.hidden_dim, height, width, device=self.conv.weight.device)
        return init_h, init_c


class ConvLSTM(nn.Module):
    """
    Parameters:
        input_dim: Number of channels in input#
        hidden_dim: Number of hidden channels
        kernel_size: Size of kernel in convolutions
        num_layers: Number of LSTM layers stacked on each other
        batch_first: Whether dimension 0 is the batch or not
        bias: Bias or no bias in Convolution
        return_all_layers: Return the list of computations for all layers
        Note: Will do same padding.
    Input:
        A tensor of size [B, T, C, H, W] or [T, B, C, H, W]#
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

    def __init__(self, input_dim, hidden_dim, output_dim, output_tl, kernel_size=(3, 3), num_layers=2,
                 batch_first=True, bias=True):
        super(ConvLSTM, self).__init__()

        self._check_kernel_size_consistency(kernel_size)

        # Make sure that both `kernel_size` and `hidden_dim` are lists having len == num_layers
        kernel_size = self._extend_for_multilayer(kernel_size, num_layers)
        hidden_dim = self._extend_for_multilayer(hidden_dim, num_layers)
        if not len(kernel_size) == len(hidden_dim) == num_layers:
            raise ValueError('Inconsistent list length.')

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim
        self.output_dim = output_dim
        self.output_tl = output_tl
        self.kernel_size = kernel_size
        self.num_layers = num_layers  # Define transposed convolution layers for spatial upsampling
        # # Upsample layer to increase dimension from 21 to 42
        # self.upsample_conv1 = nn.ConvTranspose2d(in_channels=hidden_dim[0],
        #                                          out_channels=hidden_dim[0],
        #                                          kernel_size=4, stride=2, padding=1)
        # # Upsampling to exact 50x50 dimensions
        # self.upsample_to_50 = nn.Upsample(size=(50, 50), mode='bilinear', align_corners=False)

        # Update the final convolution layer for 3-class output
        self.final_conv = nn.Conv2d(in_channels=hidden_dim[0], out_channels=output_dim, kernel_size=1)

        self.batch_first = batch_first
        self.bias = bias

        cell_list = []
        for i in range(0, self.num_layers):
            # if i==0:
            #     cur_input_dim = self.input_dim
            # else:
            #     cur_input_dim = self.hidden_dim[i - 1]
            cur_input_dim = self.input_dim if i == 0 else self.hidden_dim[i - 1]
            cell_list.append(ConvLSTMCell(input_dim=cur_input_dim,
                                          hidden_dim=self.hidden_dim[i],
                                          kernel_size=self.kernel_size[i],
                                          bias=self.bias))

        self.cell_list = nn.ModuleList(cell_list)

    def forward(self, input_tensor, hidden_state=None):
        """
        Parameters
        ----------
        input_tensor: 5-D Tensor either of shape (t, b, c, h, w) or (b, t, c, h, w)
        hidden_state: todo
            None. todo implement stateful
        Returns
        -------
        last_state_list, layer_output
        """
        if not self.batch_first:
            # (t, b, c, h, w) -> (b, t, c, h, w)
            input_tensor = input_tensor.permute(1, 0, 2, 3, 4)

        # Implement stateful ConvLSTM
        if hidden_state is not None:
            raise NotImplementedError()
        else:
            # Since the init is done in forward. Can send image size here
            b, _, _, h, w = input_tensor.size()
            hidden_state = self._init_hidden(batch_size=b, image_size=(h, w))

        layer_output_list = []
        last_state_list = []

        seq_len = input_tensor.size(1)
        cur_layer_input = input_tensor

        for layer_idx in range(self.num_layers):

            h, c = hidden_state[layer_idx]
            output_inner = []
            for t in range(seq_len):
                h, c = self.cell_list[layer_idx](input_tensor=cur_layer_input[:, t, :, :, :], cur_state=[h, c])
                output_inner.append(h)

            layer_output = torch.stack(output_inner, dim=1)
            cur_layer_input = layer_output

            layer_output_list.append(layer_output)
            last_state_list.append([h, c])

            last_state_list = last_state_list[-1:]

        # Process each layer's output
        final_outputs = []
        for layer_idx, layer_output in enumerate(layer_output_list):
            # layer_output has shape [B, T, C, H, W]
            # Downsample the time dimension  to output_tl
            total_time_steps = layer_output.size(1)
            selected_indices = torch.linspace(0, total_time_steps - 1, self.output_tl).long()
            sampled_output = layer_output[:, selected_indices, ...]

            if layer_idx == self.num_layers - 1:
                # For the last layer, reduce the channel dimension to output_dim
                B, T, C, H, W = sampled_output.shape
                sampled_output = sampled_output.view(B * T, C, H,
                                                     W)  # Flatten temporal dimension for batch-wise processing
                sampled_output = self.final_conv(sampled_output)  # Apply 1x1 convolution
                sampled_output = sampled_output.view(B, T, self.output_dim, H,
                                                     W)  # Reshape back to original format with reduced channels

            final_outputs.append(sampled_output)

        # Use the output of the last layer as the final output
        final_output = final_outputs[-1]

        return last_state_list, final_output

    def _init_hidden(self, batch_size, image_size):
        """
        :param batch_size:
        :param image_size:
        :return:
        """
        init_states = []
        for i in range(self.num_layers):
            init_states.append(self.cell_list[i].init_hidden(batch_size, image_size))
        return init_states

    @staticmethod
    def _check_kernel_size_consistency(kernel_size):
        """
        :param kernel_size:
        :return:
        """
        if not (isinstance(kernel_size, tuple) or
                (isinstance(kernel_size, list) and all([isinstance(elem, tuple) for elem in kernel_size]))):
            raise ValueError('`kernel_size` must be tuple or list of tuples')

    @staticmethod
    def _extend_for_multilayer(param, num_layers):
        """
        :param param:
        :param num_layers:
        :return:
        """
        if not isinstance(param, list):
            param = [param] * num_layers
        return param


def test_model_output(model, input_tensor, ground_truth):
    """
    Test the model output and ground truth for size, value range, and class validity.
    :param model: The ConvLSTM model instance.
    :param input_tensor: A sample input tensor for the model.
    :param ground_truth: The ground truth data for comparison.
    """
    # Forward pass
    _, output = model(input_tensor)

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


# Example usage
if __name__ == "__main__":
    import test  # for debug
    import visualization
    import torch.onnx

    input_dim = 1
    hidden_dim = 64
    output_dim = 2
    output_tl = 168
    kernel_size = (3, 3)
    num_layers = 2
    height = 21  # Example height, adjust as needed
    width = 21  # Example width, adjust as needed
    bias = True

    # Initialize model using the configuration
    convLSTM_model = ConvLSTM(input_dim=input_dim,
                              hidden_dim=hidden_dim,
                              output_dim=output_dim,
                              output_tl=output_tl,
                              kernel_size=kernel_size,
                              num_layers=num_layers)

    # Prepare test dataset
    test_loader = test.load_test_dataset()
    for data in test_loader:
        input_data = data['lr']
        gt = data['gt']

        output = test_model_output(convLSTM_model, input_data, gt)

        gt = gt.squeeze(dim=0)
        output = output.squeeze(dim=0)
        sr_3d = torch.argmax(output, dim=1)
        print(gt.shape)
        print(sr_3d.shape)
        visualization.visualize_sample(gt, sr_3d, title="Ground Truth vs Output", slice_idx=(84, 10, 10))

        # Instantiate the submodel using the configuration
        submodel = ConvLSTMCell(input_dim=input_dim,
                                hidden_dim=hidden_dim,
                                kernel_size=kernel_size,
                                bias=bias)

        # Create a dummy input and initial state using the configuration
        dummy_input = torch.randn(1, input_dim, height, width)
        dummy_state = (torch.zeros(1, hidden_dim, height, width),
                       torch.zeros(1, hidden_dim, height, width))
        # Export to ONNX
        torch.onnx.export(submodel, (dummy_input, dummy_state), "submodel_convlstm_cell.onnx")

        break
