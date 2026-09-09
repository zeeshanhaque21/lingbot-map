"""Explicit MPS adaptations for the pinned FixAnything/DiffSynth runtime."""

import torch


def rotary_float32(x, freqs, num_heads):
    """Real-valued rotary multiplication avoids unsupported MPS float64."""
    pairs = x.float().reshape(*x.shape[:2], num_heads, -1, 2)
    real, imaginary = pairs[..., 0], pairs[..., 1]
    cosine, sine = freqs.real.float(), freqs.imag.float()
    return (
        torch.stack(
            [real * cosine - imaginary * sine, real * sine + imaginary * cosine], dim=-1
        )
        .flatten(2)
        .to(x.dtype)
    )


def install_mps_compatibility():
    import diffsynth.models.wan_video_dit as dit
    import fixanything.pipelines.wan_video_v2v_mask_dynamic as pipeline

    original_frequency = dit.precompute_freqs_cis
    original_embedding = dit.sinusoidal_embedding_1d
    original_rotary = dit.rope_apply

    def frequencies(*args, **kwargs):
        # Compute original double precision phases on CPU, then store complex64.
        return original_frequency(*args, **kwargs).to(torch.complex64)

    def embedding(dim, position):
        # Tiny timestep tensors can retain the original CPU float64 calculation.
        return original_embedding(dim, position.cpu()).to(position.device)

    def rotary(x, freqs, num_heads):
        return (
            rotary_float32(x, freqs, num_heads)
            if x.device.type == "mps"
            else original_rotary(x, freqs, num_heads)
        )

    dit.precompute_freqs_cis = frequencies
    dit.sinusoidal_embedding_1d = embedding
    dit.rope_apply = rotary
    pipeline.sinusoidal_embedding_1d = embedding
    return {
        "rotary": "float32 real multiplication on MPS",
        "frequency": "original CPU float64 phases stored as complex64",
        "timestep": "original CPU float64 calculation copied to MPS",
        "offload": "explicit persistent parameter budget avoids CUDA memory queries",
    }
