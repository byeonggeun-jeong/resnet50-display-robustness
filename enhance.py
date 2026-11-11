"""입력 이미지에 쓰는 일반적인 히스토그램 기반 화질 개선 3종.

모두 OpenCV 방식대로 YCrCb의 밝기(Y) 채널만 바꾸고 색(Cr, Cb)은 그대로 둔다.
입력과 출력은 [0,1] 범위의 RGB 텐서 (N, 3, H, W).
"""
import cv2
import numpy as np
import torch

CLAHE = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))


def stretch_y(y, lo_pct=1, hi_pct=99):
    """히스토그램 스트레칭: 하위 1%~상위 99% 밝기를 0~255로 넓힌다."""
    lo, hi = np.percentile(y, [lo_pct, hi_pct])
    if hi - lo < 1:
        return y
    return np.clip((y.astype(np.float32) - lo) * 255.0 / (hi - lo), 0, 255).astype(np.uint8)


Y_FUNCS = {
    "stretch": stretch_y,           # 히스토그램 스트레칭
    "equalize": cv2.equalizeHist,   # 히스토그램 평탄화
    "clahe": CLAHE.apply,           # 적응형 히스토그램 평탄화 (CLAHE)
}
METHODS = ["none", *Y_FUNCS]


def enhance(x, method):
    if method == "none":
        return x
    f = Y_FUNCS[method]
    imgs = (x.permute(0, 2, 3, 1).cpu().numpy() * 255 + 0.5).astype(np.uint8)
    out = np.empty_like(imgs)
    for i, img in enumerate(imgs):
        ycc = cv2.cvtColor(img, cv2.COLOR_RGB2YCrCb)
        ycc[..., 0] = f(np.ascontiguousarray(ycc[..., 0]))
        out[i] = cv2.cvtColor(ycc, cv2.COLOR_YCrCb2RGB)
    return torch.from_numpy(out).permute(0, 3, 1, 2).to(x.device, x.dtype) / 255
