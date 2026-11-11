import torch
from transforms import LEVELS, perturb, color_temp, saturation
from enhance import METHODS, enhance

torch.manual_seed(0)
IMG = torch.rand(3, 32, 32)
GRAY = torch.full((3, 8, 8), 0.5)


def test_reference_level_is_identity():
    for factor, levels in LEVELS.items():
        assert len(levels) == 7
        out = perturb(IMG, factor, levels[3])
        assert torch.allclose(out, IMG, atol=1e-6), factor


def test_gray_stays_gray_at_6500k():
    assert torch.allclose(color_temp(GRAY, 6500), GRAY)


def test_warm_and_cool_white_point():
    warm = color_temp(GRAY, 3000)
    cool = color_temp(GRAY, 12000)
    assert warm[0].mean() > warm[2].mean()  # R > B
    assert cool[2].mean() > cool[0].mean()  # B > R


def test_zero_saturation_equal_channels():
    out = saturation(IMG, 0.0)
    assert torch.allclose(out[0], out[1]) and torch.allclose(out[1], out[2])


def _low_contrast_gray(n=1):
    # 0.4~0.6 사이에 몰린 회색 이미지 (대비가 낮은 화면)
    return (0.4 + 0.2 * torch.rand(n, 1, 64, 64)).expand(-1, 3, -1, -1).contiguous()


def test_enhancement_widens_low_contrast():
    x = _low_contrast_gray()
    for m in ["stretch", "equalize", "clahe"]:
        out = enhance(x, m)
        assert out.std() > 1.5 * x.std(), m


def test_stretch_and_equalize_use_full_range():
    x = _low_contrast_gray()
    for m in ["stretch", "equalize"]:
        out = enhance(x, m)
        assert out.min() < 0.05 and out.max() > 0.95, m


def test_enhancement_keeps_gray_gray():
    x = _low_contrast_gray()
    for m in ["stretch", "equalize", "clahe"]:
        out = enhance(x, m)
        assert (out[:, 0] - out[:, 2]).abs().max() <= 2 / 255, m


def test_none_is_identity_and_shapes_preserved():
    batch = torch.rand(4, 3, 16, 16)
    assert torch.equal(enhance(batch, "none"), batch)
    for m in METHODS:
        assert enhance(batch, m).shape == batch.shape
    for factor, levels in LEVELS.items():
        assert perturb(batch, factor, levels[0]).shape == batch.shape
