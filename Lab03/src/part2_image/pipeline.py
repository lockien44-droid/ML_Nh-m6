"""Run our NumPy K-Means on one or more images."""
from pathlib import Path
from hashlib import sha256
import re
import numpy as np
import pandas as pd
from PIL import Image
from time import perf_counter
from .segment import segment_full_resolution
from .preprocessing import prepare_image
from .evaluate import evaluate_segmentation
from .visualization import plot_comparison_grid, plot_inertia


def run_part(run, settings, log):
    colorspace = settings.get("colorspace", "RGB")
    values = settings["k_values"]
    if not values or any(not isinstance(k, int) or k < 1 for k in values):
        raise ValueError("Image k_values must be positive integers.")
    values = sorted(set(values))
    figures = run / "figures"
    figures.mkdir()
    segmented = run / "segmented"
    segmented.mkdir()
    sources = settings["paths"]
    stems = [re.sub(r"[^\w-]+", "_", Path(path).stem).strip("_") or "image" for path in sources]
    rows, metadata = [], []
    report = ["# Phân đoạn ảnh bằng K-Means", "",
        "Phân đoạn ảnh bằng K-Means NumPy tự cài.",
        "Thời gian gồm fit và tái tạo ảnh thu nhỏ.",
        "Inertia đo sai số màu trong không gian màu đã chuẩn hóa [0,1]; không phải độ chính xác nhận diện vật thể.", ""]
    for source, stem in zip(sources, stems):
        source = Path(source)
        if stems.count(stem) > 1:
            stem += "_" + sha256(str(source.resolve()).encode()).hexdigest()[:8]
        X, array, original = prepare_image(source, settings["max_size"], colorspace=colorspace)
        distinct = len(np.unique(X, axis=0))
        if any(k > distinct for k in values):
            raise ValueError(f"{source.name}: K exceeds {distinct} distinct resized colors.")
        scratch = {}
        for k in values:
            log(f"{stem}_k{k}")
            scores, scratch[k], model = evaluate_segmentation(X, array, k, settings)
            started = perf_counter()
            full = segment_full_resolution(original, model, colorspace)
            scores["time_full_resolution"] = perf_counter() - started
            Image.fromarray(full).save(segmented / f"{stem}_k{k}.png")
            scores.pop("iterations", None)
            rows.append(dict(image=stem, source=str(source), colorspace=colorspace, **scores))
        filename = stem + "_k" + "_".join(str(k) for k in values) + ".png"
        plot_comparison_grid(array, scratch, values, figures / filename)
        report += [f"## {source.name}", "", f"![So sánh các K](figures/{filename})", ""]
        report += ["Ảnh đúng kích thước gốc, gán màu bằng tâm cụm học từ ảnh thu nhỏ:", "",
                   *[f"- [K = {k}](segmented/{stem}_k{k}.png)" for k in values], ""]
        metadata.append(dict(source=str(source.resolve()), dataset_sha256=sha256(source.read_bytes()).hexdigest(),
                             original_size=list(original.size), working_size=[array.shape[1], array.shape[0]],
                             pixel_samples=len(X), image_key=stem))
    metrics = pd.DataFrame(rows)
    metrics.to_csv(run / "metrics.csv", index=False)
    plot_inertia(metrics, figures / "inertia_vs_k.png")
    report += ["![Inertia theo K](figures/inertia_vs_k.png)", ""]
    (run / "report.md").write_text("\n".join(report), encoding="utf-8")
    return dict(images=metadata, colorspace=colorspace, k_values=values)


def main():
    from ..pipeline import main as run
    run(default_part="2")


if __name__ == "__main__":
    main()
