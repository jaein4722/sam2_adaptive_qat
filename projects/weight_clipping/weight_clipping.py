import argparse
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import torch
import torch.nn as nn
import numpy as np

# import sys
# from PTQ import (
#     MinMaxActWrapper, MinMaxActFakeQuant, MinMaxActObserver,
#     StaticQuantWrapper, FakeQuantize, MinMaxObserver
# )

# sys.modules['__main__'].MinMaxActWrapper = MinMaxActWrapper
# sys.modules['__main__'].MinMaxActFakeQuant = MinMaxActFakeQuant
# sys.modules['__main__'].MinMaxActObserver = MinMaxActObserver
# sys.modules['__main__'].StaticQuantWrapper = StaticQuantWrapper
# sys.modules['__main__'].FakeQuantize = FakeQuantize
# sys.modules['__main__'].MinMaxObserver = MinMaxObserver


# -----------------------------
# SAM2 전용 모델 로더
# -----------------------------
def load_sam2_model(
    sam2_cfg: str,
    ckpt_path: str,
    device: str = "cpu",
) -> nn.Module:
    """
    SAM2 구성(config)과 체크포인트를 사용해 모델을 로드한다.
    """
    dev = torch.device(device)
    try:
        from sam2.build_sam import build_sam2  # type: ignore
    except Exception as e:
        raise ImportError(f"SAM2 로더 임포트 실패: {e}")
    model = build_sam2(config_file=sam2_cfg, ckpt_path=ckpt_path, device=device, mode="eval")
    model.to(dev).eval()
    return model


# -----------------------------
# 가중치 클리핑 유틸리티
# -----------------------------
@torch.no_grad()
def iter_target_params(
    model: nn.Module,
    modules: Optional[List[str]] = None,
    include_bias: bool = False,
) -> Iterable[Tuple[str, torch.Tensor]]:
    # 기본 대상: Linear/Conv2d weight만
    INCLUDE_TYPES = (nn.Linear, nn.Conv2d)
    # 절대/강력 제외: 정규화/임베딩/포지셔널/rotary/scale 등
    EXCLUDE_TYPES = (
        nn.LayerNorm, nn.BatchNorm1d, nn.BatchNorm2d, nn.BatchNorm3d,
        nn.GroupNorm, nn.SyncBatchNorm, nn.Embedding
    )
    EXCLUDE_NAME_KEYS = (
        "pos_embed", "position", "relative_position", "rel_pos",
        "embedding", "rope", "rotary", "norm", "rmsnorm", "bn",
        "qk_scale", "logit_scale"
    )
    # 모듈 경로 기반으로도 추가 제외 (필요시 확장)
    EXCLUDE_MODULE_PREFIXES = (
        "image_encoder.patch_embed",            # stem은 초기에 제외 권장
        # SAM2 대응 프리픽스
        "sam_prompt_encoder.pe_layer",
        "sam_prompt_encoder.point_embeddings",
        "sam_prompt_encoder.box_embeddings",
        "sam_mask_decoder.iou_prediction_head",
        "sam_mask_decoder.output_hypernetworks_mlps",
    )

    if modules is None:
        modules = [n for n, _ in model.named_children()]

    # 선택된 top-level 하위모듈만 순회
    selected: List[Tuple[str, nn.Module]] = []
    for n, m in model.named_children():
        if n in modules:
            selected.append((n, m))

    def _name_has_any(name_lower: str, keys: Tuple[str, ...]) -> bool:
        return any(k in name_lower for k in keys)

    yielded = set()
    for root_name, sub in selected:
        for mod_name, mod in sub.named_modules():
            full_mod_name = f"{root_name}.{mod_name}" if mod_name else root_name
            lname = full_mod_name.lower()
            # 1) 모듈 경로 프리픽스 제외
            if lname.startswith(EXCLUDE_MODULE_PREFIXES):
                continue
            # 2) 타입 제외
            if isinstance(mod, EXCLUDE_TYPES):
                continue
            # 3) 이름 키워드 제외
            if _name_has_any(lname, EXCLUDE_NAME_KEYS):
                continue
            # 4) 포함 타입 필터 (Linear/Conv2d만 대상으로)
            if not isinstance(mod, INCLUDE_TYPES):
                continue

            # 하위 파라미터는 현재 모듈 기준으로만 (recurse=False) 꺼내 중복방지
            for pname, param in mod.named_parameters(recurse=False):
                if param is None or param.ndim == 0:
                    continue
                if pname == "weight" or (include_bias and pname == "bias"):
                    fqname = f"{full_mod_name}.{pname}"
                    if fqname not in yielded:
                        yielded.add(fqname)
                        yield fqname, param


@torch.no_grad()
def compute_global_threshold_percentile(
    model: nn.Module,
    percentile: float = 99.9,
    modules: Optional[List[str]] = None,
    include_bias: bool = False,
    bins: int = 16384,
) -> float:
    """
    매우 큰 텐서에서도 동작하도록, 전체 |w| 분포를 스트리밍 히스토그램으로 근사하여
    퍼센타일 임계값을 계산한다. (메모리-안전)
    """
    # 1) 전역 최대 절대값 추정
    max_abs = 0.0
    any_param = False
    for _, p in iter_target_params(model, modules=modules, include_bias=include_bias):
        any_param = True
        val = float(p.detach().abs().max().item())
        if val > max_abs:
            max_abs = val
    if (not any_param) or max_abs <= 0.0:
        return 0.0

    # 2) 고정 bin 경계 생성 및 누적 히스토그램
    #    [0, max_abs] 구간을 균등 분할 (bins+1 edges)
    bin_edges = np.linspace(0.0, max_abs, num=bins + 1, dtype=np.float64)
    counts = np.zeros(bins, dtype=np.int64)

    for _, p in iter_target_params(model, modules=modules, include_bias=include_bias):
        arr = p.detach().abs().flatten().cpu().numpy().astype(np.float64, copy=False)
        if arr.size == 0:
            continue
        h, _ = np.histogram(arr, bins=bin_edges)
        counts += h

    total = int(counts.sum())
    if total == 0:
        return 0.0
    target = total * (percentile / 100.0)
    cdf = np.cumsum(counts)
    idx = int(np.searchsorted(cdf, target, side="left"))
    if idx >= bins:
        idx = bins - 1
    # 임계값은 해당 bin의 상한값으로 사용
    thr = float(bin_edges[idx + 1])
    return thr


@torch.no_grad()
def clip_model_weights(
    model: nn.Module,
    mode: str = "global_percentile",  # 'global_percentile' | 'per_layer_percentile' | 'abs'
    value: float = 0.1,  # mode='abs'일 때 사용 ([-value, value])
    percentile: float = 99.9,  # mode='*_percentile'일 때 사용
    modules: Optional[List[str]] = None,
    include_bias: bool = False,
) -> Dict[str, float]:
    """
    단순 가중치 클리핑.
    반환값은 적용 임계값의 요약 정보.
    """
    summary: Dict[str, float] = {}
    if mode == "global_percentile":
        thr = compute_global_threshold_percentile(
            model, percentile=percentile, modules=modules, include_bias=include_bias
        )
        for _, p in iter_target_params(model, modules=modules, include_bias=include_bias):
            p.clamp_(min=-thr, max=thr)
        summary["global_threshold"] = float(thr)
    elif mode == "per_layer_percentile":
        for pname, p in iter_target_params(model, modules=modules, include_bias=include_bias):
            abs_w = p.detach().abs().flatten().cpu()
            thr = float(torch.quantile(abs_w, percentile / 100.0).item())
            p.clamp_(min=-thr, max=thr)
            summary[pname] = float(thr)
    elif mode == "abs":
        thr = float(value)
        for _, p in iter_target_params(model, modules=modules, include_bias=include_bias):
            p.clamp_(min=-thr, max=thr)
        summary["abs_threshold"] = float(thr)
    else:
        raise ValueError("mode must be one of {'global_percentile','per_layer_percentile','abs'}")
    for n,m in model.named_modules():
        if isinstance(m, torch.nn.LayerNorm):
            print(n, m.weight.abs().min().item(), m.weight.abs().mean().item())
    return summary


# -----------------------------
# CLI
# -----------------------------
def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser(description="SAM2 weight clipping")
    # SAM2 전용 인자
    ap.add_argument("--sam2_cfg", type=str, required=True, help="SAM2 구성 파일 경로 (e.g., configs/sam2.1/sam2.1_hiera_b+.yaml)")
    ap.add_argument("--ckpt", type=str, required=True, help="SAM2 체크포인트 경로 (*.pt)")
    ap.add_argument("--device", type=str, default="cpu", choices=["cpu", "cuda"], help="로드/적용 장치")

    # 대상 모듈 선택 (SAM2 탑레벨 명칭 기준)
    ap.add_argument("--no_enc", action="store_true", help="image_encoder 제외")
    ap.add_argument("--no_prompt", action="store_true", help="sam_prompt_encoder 제외")
    ap.add_argument("--no_dec", action="store_true", help="sam_mask_decoder 제외")
    ap.add_argument("--include_bias", action="store_true", help="bias 포함 여부")

    # 클리핑 설정
    ap.add_argument("--mode", type=str, default="global_percentile", choices=["global_percentile", "per_layer_percentile", "abs"], help="클리핑 모드")
    ap.add_argument("--percentile", type=float, default=99.9, help="percentile 모드에서 사용")
    ap.add_argument("--abs_value", type=float, default=0.1, help="abs 모드에서 사용")

    # 저장
    ap.add_argument("--output", type=str, required=True, help="클리핑 적용 후 저장 경로 (*.pth)")
    return ap.parse_args()


def main():
    args = parse_args()
    model = load_sam2_model(
        sam2_cfg=args.sam2_cfg,
        ckpt_path=args.ckpt,
        device=args.device,
    )

    targets: List[Tuple[str, bool]] = [
        ("image_encoder", not args.no_enc),
        ("sam_prompt_encoder", not args.no_prompt),
        ("sam_mask_decoder", not args.no_dec),
    ]
    selected_modules = [name for name, enable in targets if enable and hasattr(model, name)]

    summary = clip_model_weights(
        model,
        mode=args.mode,
        value=args.abs_value,
        percentile=args.percentile,
        modules=selected_modules,
        include_bias=args.include_bias,
    )

    out_path = Path(args.output)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"model": model.state_dict()}, str(out_path))

    print("Clipping summary:")
    for k, v in summary.items():
        print(f"  {k}: {v}")
    print(f"Saved clipped model to: {out_path}")


if __name__ == "__main__":
    main()


