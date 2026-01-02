"""
Unit test for the Encoder module
"""
import torch
from scripts.decoder import Decoder

def test_decoder():
    """
    Unit test for the decoder
    """
    # 1. Create the decoder
    decoder = Decoder()
    # 2. Create dummy z_t: (batch=4, 32, 32)
    z_t = torch.randn((4, 32, 32))
    # 3. Create dummy h_t: (batch=4, 512)
    h_t = torch.randn((4, 512))
    # 4. Forward pass
    mean, std = decoder(z_t, h_t)
    # 5. Assert mean.shape == (4, 1, 84, 84)
    assert mean.shape == (4, 1, 84, 84)
    # 6. Assert std.shape == (4, 1, 84, 84)
    assert std.shape == (4, 1, 84, 84)
    # 3. Assert: 0 <= mean <= 1 (sigmoid output)
    assert torch.all((mean >= 0) & (mean <= 1))
    # 4. Assert: std > 0 (softplus + epsilon)
    assert torch.all(std > 0)

def test_encoder_gradient_flow():
    """Test that gradients flow through encoder"""
    decoder = Decoder()
    z_t = torch.randn((4, 32, 32), requires_grad=True)
    h_t = torch.randn((4, 512), requires_grad=True)
    mean, std = decoder(z_t, h_t)
    loss = mean.sum() + std.sum()
    loss.backward()
    # Check gradients exist
    assert z_t.grad is not None
    assert h_t.grad is not None
    assert decoder.conv_tr1.weight.grad is not None
