"""ResNet-50 V1·V2 가중치의 화면 조건별 Top-1 정확도와 정답 확률 측정.

python evaluate.py
  - 개선 기법(없음 / 스트레칭 / 평탄화 / CLAHE) × 원본 + 5요인 × 7단계를 V1·V2 모두 측정
  - results/: results.json, 정확도 곡선, 히트맵, 개선 기법 비교, 모델별 오분류 예시
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision import transforms as T
from torchvision.models import resnet50, ResNet50_Weights

from enhance import METHODS, enhance
from transforms import LEVELS, perturb

ROOT = Path(__file__).parent
DATA = ROOT / "data" / "imagenette2-320"
RESULTS = ROOT / "results"
DEVICE = "cuda"

# Imagenette 폴더(WordNet ID) -> ImageNet-1k 클래스 번호
WNID_TO_IDX = {
    "n01440764": 0,    # tench
    "n02102040": 217,  # English springer
    "n02979186": 482,  # cassette player
    "n03000684": 491,  # chain saw
    "n03028079": 497,  # church
    "n03394916": 566,  # French horn
    "n03417042": 569,  # garbage truck
    "n03425413": 571,  # gas pump
    "n03445777": 574,  # golf ball
    "n03888257": 701,  # parachute
}
CATEGORIES = ResNet50_Weights.IMAGENET1K_V1.meta["categories"]
WEIGHTS = {"V1": ResNet50_Weights.IMAGENET1K_V1, "V2": ResNet50_Weights.IMAGENET1K_V2}
MEAN = torch.tensor([0.485, 0.456, 0.406], device=DEVICE).view(1, 3, 1, 1)
STD = torch.tensor([0.229, 0.224, 0.225], device=DEVICE).view(1, 3, 1, 1)
METHOD_LABELS = {"none": "no enhancement", "stretch": "histogram stretch",
                 "equalize": "histogram equalization", "clahe": "CLAHE"}


def load_val():
    """검증 이미지를 Resize(256) + CenterCrop(224) 한 uint8 텐서로 캐시."""
    cache = ROOT / "data" / "val_224.pt"
    if cache.exists():
        return torch.load(cache)
    tf = T.Compose([T.Resize(256), T.CenterCrop(224), T.PILToTensor()])
    items = [(p, idx) for wnid, idx in sorted(WNID_TO_IDX.items())
             for p in sorted((DATA / "val" / wnid).iterdir())]
    x = torch.stack([tf(Image.open(p).convert("RGB")) for p, _ in items])
    y = torch.tensor([c for _, c in items])
    torch.save((x, y), cache)
    return x, y


def build_model(weights):
    return resnet50(weights=weights).to(DEVICE).eval().to(memory_format=torch.channels_last)


@torch.no_grad()
def predict(models, x_uint8, y, factor=None, level=None, method="none", bs=250):
    """한 조건에서 모든 모델의 예측과 정답 확률. 변환·개선은 모델 간에 한 번만 계산."""
    preds = {k: [] for k in models}
    probs = {k: [] for k in models}
    for i in range(0, len(x_uint8), bs):
        x = x_uint8[i:i + bs].to(DEVICE).float() / 255
        if factor is not None:
            x = perturb(x, factor, level)
        x = enhance(x, method)
        x = ((x - MEAN) / STD).contiguous(memory_format=torch.channels_last)
        yb = y[i:i + bs].to(DEVICE)
        for k, model in models.items():
            with torch.autocast("cuda", dtype=torch.bfloat16):
                p = F.softmax(model(x).float(), dim=1)
            preds[k].append(p.argmax(1).cpu())
            probs[k].append(p[torch.arange(len(p)), yb].cpu())
    return {k: (torch.cat(preds[k]), torch.cat(probs[k])) for k in models}


def summary(pred, prob, y):
    return {"acc": (pred == y).float().mean().item(), "prob": prob.mean().item()}


def class_drops(y, clean_pred, preds, rows):
    """요인별 최악 단계(원본 단계 제외)에서 클래스별 정확도 하락폭."""
    res = {}
    for factor, levels in LEVELS.items():
        worst = min((r for r in rows[factor] if r["level"] != levels[3]), key=lambda r: r["acc"])["level"]
        p = preds[(factor, worst)]
        out = []
        for idx in WNID_TO_IDX.values():
            m = y == idx
            a0 = (clean_pred[m] == y[m]).float().mean().item()
            a1 = (p[m] == y[m]).float().mean().item()
            out.append({"class": CATEGORIES[idx], "clean_acc": a0, "worst_acc": a1, "drop": a0 - a1})
        res[factor] = {"worst_level": worst, "classes": sorted(out, key=lambda r: -r["drop"])}
    return res


def save_examples(x, y, clean_pred, pred, model_name, factor, level, n=8):
    """원본은 맞혔는데 변환 후 틀린 예시 n장."""
    idx = torch.nonzero((clean_pred == y) & (pred != y)).flatten()
    if len(idx) == 0:
        return
    idx = idx[torch.linspace(0, len(idx) - 1, min(n, len(idx))).long()]
    fig, axes = plt.subplots(2, len(idx), figsize=(2.2 * len(idx), 5), squeeze=False)
    for j, i in enumerate(idx):
        img = x[i:i + 1].float() / 255
        for r, im in enumerate([img, perturb(img, factor, level)]):
            axes[r, j].imshow(im[0].permute(1, 2, 0).numpy())
            axes[r, j].set_xticks([]); axes[r, j].set_yticks([])
        axes[0, j].set_title(f"true: {CATEGORIES[y[i]]}", fontsize=8)
        axes[1, j].set_title(f"pred: {CATEGORIES[pred[i]]}", fontsize=8, color="red")
    axes[0, 0].set_ylabel("original")
    axes[1, 0].set_ylabel(f"{factor}={level}")
    fig.suptitle(f"ResNet-50 {model_name}: correct on original, wrong after {factor}={level}")
    fig.tight_layout()
    fig.savefig(RESULTS / f"misclassified_{model_name}_{factor}.png", dpi=110)
    plt.close(fig)


def plot_curves(series, fname, title):
    """series: {라벨: {factor: rows}}"""
    fig, axes = plt.subplots(1, 5, figsize=(20, 4), sharey=True)
    for ax, factor in zip(axes, LEVELS):
        for label, factors in series.items():
            rows = factors[factor]
            ax.plot([r["level"] for r in rows], [r["acc"] * 100 for r in rows], "o-", label=label)
        ax.axvline(LEVELS[factor][3], color="gray", ls=":", lw=1)
        ax.set_title(factor); ax.set_xlabel("level"); ax.grid(alpha=0.3)
    axes[0].set_ylabel("Top-1 accuracy (%)")
    axes[0].legend()
    fig.suptitle(title)
    fig.tight_layout()
    fig.savefig(RESULTS / fname, dpi=110)
    plt.close(fig)


def plot_heatmap(results):
    """요인(행) × 단계(열) 정확도 히트맵, V1·V2 나란히. 색은 두 모델 공통 범위."""
    factors = list(LEVELS)
    mats = {m: np.array([[r["acc"] * 100 for r in results[m]["none"]["factors"][f]] for f in factors])
            for m in WEIGHTS}
    vmin = min(a.min() for a in mats.values())
    vmax = max(a.max() for a in mats.values())
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.6))
    for ax, (m, mat) in zip(axes, mats.items()):
        im = ax.imshow(mat, cmap="RdYlGn", vmin=vmin, vmax=vmax, aspect="auto")
        for r, f in enumerate(factors):
            for c, lv in enumerate(LEVELS[f]):
                lv_txt = f"{round(lv, 2):g}" if f != "color_temp" else f"{lv}K"
                ax.text(c, r, f"{mat[r, c]:.1f}\n({lv_txt})", ha="center", va="center", fontsize=8,
                        fontweight="bold" if c == 3 else "normal")
        ax.set_xticks(range(7), ["1", "2", "3", "4\n(original)", "5", "6", "7"])
        ax.set_yticks(range(len(factors)), factors)
        ax.set_title(f"ResNet-50 {m}  (clean {results[m]['none']['clean']['acc'] * 100:.1f}%)")
    fig.colorbar(im, ax=axes, label="Top-1 accuracy (%)", shrink=0.9)
    fig.suptitle("Accuracy by display condition (level index 1-7, value in parentheses)")
    fig.savefig(RESULTS / "heatmap_accuracy.png", dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    RESULTS.mkdir(exist_ok=True)
    x, y = load_val()
    print(f"val images: {len(x)}", flush=True)
    models = {k: build_model(w) for k, w in WEIGHTS.items()}
    results = {k: {} for k in models}
    none_preds = {k: {} for k in models}

    for method in METHODS:
        out = predict(models, x, y, method=method)
        for k in models:
            results[k][method] = {"clean": summary(*out[k], y), "factors": {}}
            if method == "none":
                none_preds[k]["clean"] = out[k][0]
        for factor, levels in LEVELS.items():
            rows = {k: [] for k in models}
            for lv in levels:
                out = predict(models, x, y, factor, lv, method)
                for k in models:
                    rows[k].append({"level": lv, **summary(*out[k], y)})
                    if method == "none":
                        none_preds[k][(factor, lv)] = out[k][0]
            for k in models:
                results[k][method]["factors"][factor] = rows[k]
        for k in models:
            print(f"[{k} / {method}] clean {results[k][method]['clean']['acc'] * 100:.2f}%", flush=True)
            for factor, rows in results[k][method]["factors"].items():
                print(f"  {factor:11s} " + " ".join(f"{r['acc'] * 100:5.1f}" for r in rows), flush=True)

    for k in models:
        clean_pred = none_preds[k]["clean"]
        drops = class_drops(y, clean_pred, none_preds[k], results[k]["none"]["factors"])
        results[k]["none"]["class_drops"] = drops
        for factor, d in drops.items():
            save_examples(x, y, clean_pred, none_preds[k][(factor, d["worst_level"])], k, factor, d["worst_level"])

    with open(RESULTS / "results.json", "w", encoding="utf-8") as f:
        json.dump(results, f, indent=1, ensure_ascii=False)

    plot_curves({k: results[k]["none"]["factors"] for k in models}, "accuracy_curves.png",
                "ResNet-50 V1 vs V2 on Imagenette val: accuracy vs display condition")
    plot_heatmap(results)
    for k in models:
        plot_curves({METHOD_LABELS[m]: results[k][m]["factors"] for m in METHODS}, f"enhancement_{k}.png",
                    f"ResNet-50 {k}: accuracy with histogram-based enhancement applied to the input")


if __name__ == "__main__":
    main()
