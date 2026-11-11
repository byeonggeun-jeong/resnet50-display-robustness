"""화면 조건 변환 5종.

모든 함수는 [0,1] 범위의 RGB 텐서 (..., 3, H, W)를 받아 같은 모양을 돌려준다.
크기 조정 후, 정규화 전에 적용한다.
"""
import math
import torch

# 각 요인 7단계, 가운데(index 3)가 원본
LEVELS = {
    "brightness": [0.3, 0.5333, 0.7667, 1.0, 1.2333, 1.4667, 1.7],
    "gamma": [1.6, 1.8, 2.0, 2.2, 2.5, 2.8, 3.0],
    "color_temp": [3000, 4000, 5000, 6500, 8000, 10000, 12000],
    "contrast": [0.3, 0.5333, 0.7667, 1.0, 1.2333, 1.4667, 1.7],
    "saturation": [0.0, 0.3333, 0.6667, 1.0, 1.3333, 1.6667, 2.0],
}
RANGES = {k: (v[0], v[-1]) for k, v in LEVELS.items()}

LUMA = torch.tensor([0.2126, 0.7152, 0.0722])  # Rec.709 휘도 가중치


def luminance(x):
    w = LUMA.to(x.device, x.dtype).view(3, 1, 1)
    return (x * w).sum(dim=-3, keepdim=True)


def brightness(x, k):
    """패널 휘도 배율."""
    return (x * k).clamp(0, 1)


def gamma(x, g, ref=2.2):
    """기준 감마 2.2로 인코딩된 영상을 감마 g 패널로 보여 준 결과."""
    return x.clamp(0, 1) ** (g / ref)


def kelvin_to_rgb(t):
    """흑체 색온도 -> RGB (Tanner Helland 근사, 0~1)."""
    t = t / 100.0
    if t <= 66:
        r = 255.0
        g = 99.4708025861 * math.log(t) - 161.1195681661
    else:
        r = 329.698727446 * (t - 60) ** -0.1332047592
        g = 288.1221695283 * (t - 60) ** -0.0755148492
    if t >= 66:
        b = 255.0
    elif t <= 19:
        b = 0.0
    else:
        b = 138.5177312231 * math.log(t - 10) - 305.0447927307
    return [min(max(c, 0.0), 255.0) / 255.0 for c in (r, g, b)]


def color_temp(x, kelvin, ref=6500):
    """화이트 포인트를 kelvin으로 옮긴다. 6500K에서 항등, 최대 채널 배율은 1."""
    if kelvin == ref:
        return x
    gain = torch.tensor(kelvin_to_rgb(kelvin)) / torch.tensor(kelvin_to_rgb(ref))
    gain = gain / gain.max()
    return (x * gain.to(x.device, x.dtype).view(3, 1, 1)).clamp(0, 1)


def contrast(x, k):
    """이미지 평균 휘도를 축으로 명암 폭을 k배."""
    m = luminance(x).mean(dim=(-3, -2, -1), keepdim=True)
    return ((x - m) * k + m).clamp(0, 1)


def saturation(x, s):
    """휘도는 두고 색 성분만 s배. s=0이면 흑백."""
    y = luminance(x)
    return (y + (x - y) * s).clamp(0, 1)


APPLY = {
    "brightness": brightness,
    "gamma": gamma,
    "color_temp": color_temp,
    "contrast": contrast,
    "saturation": saturation,
}


def perturb(x, factor, level):
    return APPLY[factor](x, level)
