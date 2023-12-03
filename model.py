from convLSTM import ConvLSTMCell  # Import ConvLSTMCell from your file
import torch
import torch.nn as nn

__all__ = [
    "ConvLSTM3DClassifier",
    "ConvLSTM3DTransposedConv"
]


class ConvLSTM3DClassifier(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, output_size):
        super(ConvLSTM3DClassifier, self).__init__()
        self.hidden_dim = hidden_dim  # Add this line
        self.conv_lstm = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias=True)

        # Assuming the output size is also a 3D dataset but with different dimensions
        self.output_height, self.output_width, self.output_time = output_size

        # Additional layers to adjust spatial dimensions
        self.conv_resize = nn.Conv2d(hidden_dim, hidden_dim, kernel_size=3, padding=1)
        self.upsample = nn.Upsample(size=(self.output_height, self.output_width))

        # Final layer to adjust time dimension
        self.time_adjust = nn.Conv2d(hidden_dim, self.output_time, kernel_size=1)

    def forward(self, x):
        device = x.device
        b, t, _, h, w = x.size()  # batch size, time steps, height, width
        h_t, c_t = self.init_hidden_state(b, h, w, device)  # Initialize hidden state

        # Process each time step
        for time_step in range(t):
            h_t, c_t = self.conv_lstm(x[:, time_step, :, :], (h_t, c_t))

        # Resize spatial dimensions
        h_t = self.conv_resize(h_t)
        h_t = self.upsample(h_t)

        # Adjust time dimension
        out = self.time_adjust(h_t)

        return out

    def init_hidden_state(self, batch_size, height, width, device):
        # Initialize hidden and cell states on the specified device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))


class ConvLSTM3DTransposedConv(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size, output_size):
        super(ConvLSTM3DTransposedConv, self).__init__()
        self.hidden_dim = hidden_dim  # Add this line
        self.conv_lstm = ConvLSTMCell(input_dim, hidden_dim, kernel_size, bias=True)

        # Assuming the output size is also a 3D dataset but with different dimensions
        self.output_height, self.output_width, self.output_time = output_size

        # Transposed convolution layer
        self.transposed_conv = nn.ConvTranspose2d(hidden_dim, hidden_dim, kernel_size=3, stride=2, padding=1,
                                                  output_padding=1)

        # Final layer to adjust time dimension
        self.time_adjust = nn.Conv2d(hidden_dim, self.output_time, kernel_size=1)

    def forward(self, x):
        device = x.device
        b, t, _, h, w = x.size()  # batch size, time steps, height, width
        h_t, c_t = self.init_hidden_state(b, h, w, device)  # Initialize hidden state

        # Process each time step
        for time_step in range(t):
            h_t, c_t = self.conv_lstm(x[:, time_step, :, :], (h_t, c_t))

        # Upsample using transposed convolution
        h_t = self.transposed_conv(h_t)

        # Fine-tune the size using interpolation
        h_t = nn.functional.interpolate(h_t, size=(self.output_height, self.output_width), mode='bilinear',
                                        align_corners=False)

        # Adjust time dimension
        out = self.time_adjust(h_t)

        return out

    def init_hidden_state(self, batch_size, height, width, device):
        # Initialize hidden and cell states on the specified device
        return (torch.zeros(batch_size, self.hidden_dim, height, width, device=device),
                torch.zeros(batch_size, self.hidden_dim, height, width, device=device))
