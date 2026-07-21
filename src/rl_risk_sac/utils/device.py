from __future__ import annotations

from typing import Any

import torch


def resolve_device(config: dict[str, Any]) -> str:
    requested = str(config.get("device", "cpu"))
    selection_cfg = config.get("device_selection", {})
    auto_values = {"cuda", "cuda:auto", "auto"}

    if requested not in auto_values:
        if requested.startswith("cuda") and not torch.cuda.is_available():
            if bool(selection_cfg.get("fallback_to_cpu", True)):
                return "cpu"
            raise RuntimeError(f"Requested device {requested!r}, but CUDA is not available")
        return requested

    if not torch.cuda.is_available():
        if bool(selection_cfg.get("fallback_to_cpu", True)):
            return "cpu"
        raise RuntimeError("Auto CUDA device selection requested, but CUDA is not available")

    device_count = torch.cuda.device_count()
    candidates = selection_cfg.get("candidate_ids")
    if candidates is None:
        candidate_ids = list(range(device_count))
    else:
        candidate_ids = [int(device_id) for device_id in candidates]

    if not candidate_ids:
        raise ValueError("device_selection.candidate_ids must not be empty")

    invalid = [device_id for device_id in candidate_ids if device_id < 0 or device_id >= device_count]
    if invalid:
        raise ValueError(f"Invalid CUDA device ids {invalid}; available ids are 0..{device_count - 1}")

    min_free_memory_gb = float(selection_cfg.get("min_free_memory_gb", 0.0))
    memory_weight = float(selection_cfg.get("memory_weight", 0.7))
    compute_weight = float(selection_cfg.get("compute_weight", 0.3))

    summaries = []
    for device_id in candidate_ids:
        free_bytes, total_bytes = torch.cuda.mem_get_info(device_id)
        props = torch.cuda.get_device_properties(device_id)
        free_gb = free_bytes / 1024**3
        free_ratio = free_bytes / max(total_bytes, 1)
        compute_score = props.multi_processor_count * (props.major + props.minor / 10.0)
        summaries.append(
            {
                "id": device_id,
                "name": props.name,
                "free_gb": free_gb,
                "free_ratio": free_ratio,
                "compute_score": compute_score,
                "score": 0.0,
            }
        )

    max_compute_score = max(item["compute_score"] for item in summaries)
    best: tuple[float, int] | None = None
    for item in summaries:
        compute_ratio = item["compute_score"] / max(max_compute_score, 1e-6)
        score = memory_weight * item["free_ratio"] + compute_weight * compute_ratio
        item["compute_ratio"] = compute_ratio
        item["score"] = score
        if item["free_gb"] < min_free_memory_gb:
            continue
        if best is None or item["score"] > best[0]:
            best = (item["score"], item["id"])

    if best is None:
        if bool(selection_cfg.get("fallback_to_cpu", True)):
            print(f"CUDA auto selection found no GPU with >= {min_free_memory_gb:.2f} GB free memory; using cpu")
            return "cpu"
        raise RuntimeError(f"No CUDA device satisfies min_free_memory_gb={min_free_memory_gb}")

    selected = best[1]
    if bool(selection_cfg.get("print_summary", True)):
        for item in summaries:
            marker = "*" if item["id"] == selected else " "
            print(
                f"{marker} cuda:{item['id']} {item['name']} "
                f"free={item['free_gb']:.2f}GB free_ratio={item['free_ratio']:.3f} "
                f"compute_ratio={item['compute_ratio']:.3f} score={item['score']:.3f}"
            )
        print(f"selected device: cuda:{selected}")
    return f"cuda:{selected}"
