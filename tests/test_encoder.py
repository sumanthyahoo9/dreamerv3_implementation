"""
Unit test for the Encoder module
"""
import torch
from scripts.encoder import Encoder

def test_encoder_shape():
    """
    Test the encoder shape without proprioceptive input
    """
    encoder = Encoder()
    random_inputs = torch.randn((4, 1, 84, 84))
    random_outs = encoder(random_inputs)
    assert random_outs.shape == (4, 32, 32)

def test_encoder_with_proprio():
    """
    Test the encoder with proprioceptive input
    """
    # 1. Create encoder with proprio_dim=10
    encoder = Encoder(proprio_dim=10)
    # 2. Create image: (batch=4, 1, 84, 84)
    random_image_inputs = torch.randn((4, 1, 84, 84))
    # 3. Create proprio: (batch=4, 10)
    random_proprio_input = torch.randn((4, 10))
    # 4. Forward pass with both
    random_outs = encoder(random_image_inputs, random_proprio_input)
    # 5. Assert shape still (4, 32, 32)
    assert random_outs.shape == (4, 32, 32)

def test_encoder_sampling():
    """Test that sampling produces valid one-hot vectors"""
    # 1. Create encoder
    encoder = Encoder(proprio_dim=10)
    random_image_inputs = torch.randn((4, 1, 84, 84))
    random_proprio_input = torch.randn((4, 10))
    # 2. Get logits from forward pass
    logits = encoder(random_image_inputs, random_proprio_input)
    # 3. Sample z_t using encoder.sample(logits)
    z_t = encoder.sample(logits)
    # 4. Assert:
    #    - Shape is (batch, 32, 32)
    #    - Values are 0 or 1 (one-hot)
    #    - Each category has exactly one 1 (sum along dim=-1 equals 1)
    assert z_t.shape == (4, 32, 32)
    assert torch.all((z_t == 0) | (z_t == 1))
    assert torch.allclose(z_t.sum(dim=-1), torch.ones(4, 32))

def test_encoder_gradient_flow():
    """Test that gradients flow through encoder"""
    encoder = Encoder()
    x = torch.randn(2, 1, 84, 84, requires_grad=True)
    logits = encoder(x)
    loss = logits.sum()  # Dummy loss
    loss.backward()
    
    # Check gradients exist
    assert x.grad is not None
    assert encoder.conv1.weight.grad is not None

     

