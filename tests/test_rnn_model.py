"""
Unit test for the RNN model
"""
import torch
import torch.nn as nn
from src.rnn_model import RecurrentModel

def test_rnn_model():
    """
    test the RNN model
    """
    rnn = RecurrentModel()
    z_prev = torch.randn((4, 32, 32), requires_grad=True)
    a_prev = torch.randn((4, 18), requires_grad=True)
    h_prev = torch.randn((4, 512), requires_grad=True)
    h_next = rnn(z_prev, a_prev, h_prev)
    assert h_next.shape == h_prev.shape
    loss = h_next.sum()
    loss.backward()
    assert z_prev.grad is not None
    assert a_prev.grad is not None
    assert h_prev.grad is not None


