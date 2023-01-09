import torch
import torch.nn as nn

class Transformer(nn.Module):
    def __init__(self, input_dim, output_dim, d_model, n_heads, dropout=0.1):
        super().__init__()

        # Create the encoder layers
        self.encoder_layers = nn.TransformerEncoderLayer(d_model, n_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layers, num_layers=6)

        # Create the decoder layers
        self.decoder_layers = nn.TransformerDecoderLayer(d_model, n_heads, dim_feedforward=2048, dropout=dropout)
        self.transformer_decoder = nn.TransformerDecoder(self.decoder_layers, num_layers=6)

        # Create the final linear layer
        self.linear = nn.Linear(d_model, output_dim)

        # Initialize the weights
        self.init_weights()

    def forward(self, src, trg, src_mask=None, trg_mask=None):
        # Pass the input through the encoder layers
        encoder_output = self.transformer_encoder(src, src_mask)

        # Pass the encoder output and the target through the decoder layers
        decoder_output = self.transformer_decoder(trg, encoder_output, src_mask, trg_mask)

        # Final linear layer
        output = self.linear(decoder_output)

        return output

    def init_weights(self):
        # Initialize the weights
        for p in self.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)