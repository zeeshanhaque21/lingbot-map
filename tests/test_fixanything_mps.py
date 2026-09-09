import torch

from lingbot_map.reconstruction.fixanything_mps import rotary_float32


def test_real_rotary_matches_original_double_precision_calculation():
    generator = torch.Generator().manual_seed(7)
    x = torch.randn(1, 31, 512, generator=generator)
    phase = torch.randn(31, 1, 64, generator=generator, dtype=torch.float64)
    frequencies = torch.polar(torch.ones_like(phase), phase)
    reference = (
        torch.view_as_real(
            torch.view_as_complex(x.double().reshape(1, 31, 4, 64, 2)) * frequencies
        )
        .flatten(2)
        .float()
    )
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    actual = rotary_float32(
        x.to(device), frequencies.to(torch.complex64).to(device), 4
    ).cpu()
    torch.testing.assert_close(actual, reference, atol=5e-7, rtol=1e-6)
