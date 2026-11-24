"""Export plots and a report from actual experimental outputs."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
import argparse
import json
import re
from html import escape
from pathlib import Path
from shutil import copy2
from .preprocessing import LAB



def generate(run, results, selection):
    summary = json.loads((run / "preprocessing/summary.json").read_text(encoding="utf-8"))
    weekly = summary.get("frequency") == "weekly"
    fmcg = summary.get("revenue_definition", "").startswith("price_unit")
    unit = "Revenue" if fmcg else "Sales revenue" if weekly else "Units"
    for name in results.model:
        data = pd.read_csv(run / "evaluation" / f"{name}.csv").rename(columns={"sku": "StockCode", "Revenue": "sales"})
        fig, axes = plt.subplots(1, 2, figsize=(11, 4))
        axes[0].scatter(data.sales, data.predicted, alpha=0.25, s=8)
        limit = max(data.sales.max(), data.predicted.max())
        axes[0].plot([0, limit], [0, limit], "--")
        axes[0].set(xlabel=f"Actual {unit}", ylabel=f"Predicted {unit}", title=name)
        axes[1].hist(data.residual, bins=40)
        axes[1].set(xlabel="Actual - predicted", ylabel="Count", title="Residual distribution")
        fig.tight_layout()
        fig.savefig(run / "figures" / f"{name}.png")
        plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    results.plot.bar(x="model", y=["mae", "rmse"], ax=axes[0], rot=35)
    results.plot.bar(x="model", y="fit_seconds", ax=axes[1], rot=35)
    fig.tight_layout()
    fig.savefig(run / "figures/metrics_comparison.png")
    plt.close(fig)
    headers = results.columns.tolist()
    table = ["| " + " | ".join(headers) + " |",
             "| " + " | ".join(["---"] * len(headers)) + " |"]
    for row in results.itertuples(index=False, name=None):
        table.append("| " + " | ".join(f"{value:.4f}" if isinstance(value, float) else str(value) for value in row) + " |")
    lines = ["# Lab02 — Kết quả thực nghiệm", "",
             f"Dự đoán số lượng bán dương theo {summary.get('entity', 'product')}/ngày. Chọn mô hình bằng RMSE trung bình của ba cửa sổ mở rộng trên train và validation cuối.",
             "", "## Kết quả test", "", "\n".join(table), "",
             "## Phạm vi đánh giá", "",
             "Đánh giá từng ngày với lịch sử bán thực tế đến hết ngày trước đó. "
             "Các ngày không ghi nhận giao dịch được xem là 0. "
             f"Đối tượng dự báo là {summary.get('entity', 'product')}, gồm {len(summary['products'])} nhóm được chọn từ train. "
             "Mô hình chỉ áp dụng cho nhóm sản phẩm này và dữ liệu lịch sử cùng nguồn.",
             "", "MAE là sai số tuyệt đối trung bình. MSE là sai số bình phương trung bình. "
             "RMSE là căn MSE, cùng đơn vị số lượng. R² so sánh sai số với dự đoán trung bình của tập đánh giá.",
             "", "## Biểu đồ", ""]
    for name in results.model:
        lines.extend([f"![{name}](../results/runs/{run.name}/figures/{name}.png)", ""])
    lines.extend(["", "## Phân tích sai số", "",
                  "Các bảng *_largest_errors.csv lưu 10 lỗi lớn nhất. Các bảng *_StockCode_metrics.csv và *_weekday_metrics.csv phân tích theo sản phẩm và ngày trong tuần.",
                  "Residual dương là dự báo thiếu, residual âm là dự báo dư.",
                  f"![So sánh](../results/runs/{run.name}/figures/metrics_comparison.png)"])
    trials = pd.read_csv(run / "tuning/validation_metrics.csv")
    winner = trials.loc[trials.selection_rmse.idxmin(), "model"]
    chosen = results.set_index("model").loc[winner]
    baseline = results.set_index("model").loc["mean_baseline"]
    improvement = 100 * (1 - chosen.rmse / baseline.rmse)
    rolling = results.set_index("model").loc["rolling_28_baseline"] if "rolling_28_baseline" in set(results.model) else None
    product_scores = pd.read_csv(run / "evaluation/product_baseline_comparison.csv") if (run / "evaluation/product_baseline_comparison.csv").exists() else pd.DataFrame()
    errors = pd.read_csv(run / "evaluation" / f"{winner}_largest_errors.csv").rename(columns={"sku": "StockCode", "Revenue": "sales"})
    lines.extend(["", "## Kết luận và giới hạn", "",
                  f"Mô hình được chọn bằng dữ liệu phát triển là **{winner}**. RMSE test {chosen.rmse:.4f}, R² {chosen.r2:.4f}. Mức giảm RMSE so với baseline trung bình là {improvement:.2f}%.",
                  f"R² test của mô hình được chọn là {chosen.r2:.4f}. Kết quả phản ánh một lần chia theo thời gian. Không dùng bảng test để chọn lại cấu hình.",
                  "Top sản phẩm được cố định từ toàn bộ train trước walk-forward. Các fold kiểm chứng dự báo trong phạm vi sản phẩm này, chưa mô phỏng việc chọn lại danh mục tại từng mốc quá khứ.",
                  "Không có giao dịch được quy ước là 0, nhưng dữ liệu không phân biệt hết hàng, đóng cửa và không có nhu cầu. " + ("Dữ liệu nhóm hàng giữ nguyên hai ngày biên." if summary.get('entity') == 'product_category' else "Hai ngày biên bị loại theo giả định dữ liệu có thể chưa đủ ngày."),
                  "Các đặc trưng robust gồm trung vị, độ lệch chuẩn 7/28 ngày, số ngày doanh số 0, trung bình cùng thứ bốn tuần và cuối tuần. Tất cả chỉ dùng lịch sử trước ngày dự báo.",
                  "", "## Các lỗi lớn nhất của mô hình được chọn", "",
                  "| Sản phẩm | Ngày | Thực tế | Dự đoán | Sai số tuyệt đối |",
                  "| --- | --- | --- | --- | --- |"])
    for error in errors.head(5).itertuples():
        lines.append(f"| {error.StockCode} | {error.date} | {error.sales:.2f} | {error.predicted:.2f} | {error.absolute_error:.2f} |")
    if not product_scores.empty:
        wins = int((product_scores.mean_baseline_improvement_pct > 0).sum())
        lines.extend(["", f"Mô hình cải thiện RMSE so với baseline trung bình trên {wins}/{len(product_scores)} sản phẩm. Xem evaluation/product_baseline_comparison.csv để đối chiếu từng sản phẩm với các baseline lịch sử."])
    if rolling is not None:
        lines.extend(["", f"Baseline trung bình 28 ngày đạt RMSE {rolling.rmse:.4f}, trong khi mô hình được chọn đạt {chosen.rmse:.4f}. Mức giảm RMSE so với baseline này là {100 * (1 - chosen.rmse / rolling.rmse):.2f}%."])
    if (run / "evaluation/before_after.csv").exists():
        lines.extend(["", "So sánh trước/sau được lưu trong evaluation/before_after.csv. Đây là đối chiếu mô tả trên cùng test, không phải bằng chứng cải thiện trên một tập test mới độc lập."])
    if weekly:
        lines.extend(["", "## Định nghĩa mục tiêu và dữ liệu", "", f"Đơn vị {summary.get('unit', 'source currency')}. Công thức {summary.get('revenue_definition', 'source Sales')}. Số dòng nguồn {summary['raw_rows']}, số dòng hợp lệ {summary['clean_transaction_rows']}, loại {summary.get('invalid_rows_removed', 0)} dòng không hợp lệ sau loại trùng."])
        lines = [line.replace("số lượng bán dương", "doanh thu Sales")
                 .replace("/ngày", "/tuần").replace("từng ngày", "từng tuần")
                 .replace("ngày trước đó", "tuần trước đó").replace("Các ngày không", "Các tuần không")
                 .replace("cùng đơn vị số lượng", "cùng đơn vị doanh thu nguồn")
                 .replace("Dữ liệu nhóm hàng giữ nguyên hai ngày biên.", "Chỉ giữ tuần đầy đủ từ thứ Hai đến Chủ nhật.")
                 .replace("Hai ngày biên bị loại theo giả định dữ liệu có thể chưa đủ ngày.", "Chỉ giữ tuần đầy đủ từ thứ Hai đến Chủ nhật.")
                 .replace("ngày doanh số 0", "tuần doanh thu 0")
                 .replace("7/28 ngày", "7/28 tuần").replace("28 ngày", "28 tuần")
                 .replace("bốn tuần và cuối tuần", "bốn mốc cách 7 tuần và cờ cuối tuần") for line in lines]
        lines[2:2] = [f"Nguồn {summary.get('source', 'CSV')}. Tuần bắt đầu thứ Hai. Chọn tối đa 20 sản phẩm theo doanh thu train với ít nhất 6 tuần có giao dịch trong train. Tuần thiếu giao dịch được quy ước Sales bằng 0. Các đặc trưng nội bộ lag_7 và mean_28 lần lượt là 7 tuần trước và trung bình 28 tuần trước. Đầu vào không dùng Profit, Ship Date hoặc số lượng/giảm giá của tuần dự báo.", ""]
    split_lines = []
    for split in ("train", "validation", "test"):
        dates = pd.read_csv(run / "preprocessing" / f"{split}.csv", usecols=["date"])["date"]
        start = pd.Timestamp(dates.min())
        end = pd.Timestamp(dates.max()) + pd.Timedelta(days=6 if weekly else 0)
        split_lines.append(f"{split}: {start:%d/%m/%Y} đến {end:%d/%m/%Y}, {len(dates)} mẫu")
    context = ". ".join(split_lines) + "."
    lines[4:4] = ["## Khoảng thời gian dữ liệu", "",
                  "Ngày kết thúc dưới đây bao gồm Chủ nhật cuối tuần." if weekly else "Mốc đầu và cuối của từng tập dữ liệu.",
                  "", *["- " + line for line in split_lines], ""]
    if fmcg:
        lines = [line.replace("doanh thu Sales", "doanh thu Revenue")
                 .replace("theo product/tuần", "theo sku/tuần")
                 .replace("Đối tượng dự báo là product, gồm", "Đối tượng dự báo là sku, gồm")
                 .replace("nhóm được chọn từ train", "sku được chọn từ train")
                 .replace("Sales bằng 0", "Revenue bằng 0")
                 .replace("*_StockCode_metrics.csv", "*_sku_metrics.csv")
                 .replace("| Sản phẩm | Ngày | Thực tế |", "| sku | Tuần bắt đầu | Revenue thực tế |")
                 for line in lines]
        for index, line in enumerate(lines):
            if line.startswith("Nguồn "):
                lines[index] = (f"Nguồn {summary['source']}. Revenue được tính bằng price_unit × units_sold "
                    "cho mỗi dòng hợp lệ, rồi cộng gộp theo sku × tuần qua tất cả channel, region và pack_type. "
                    "Tuần từ thứ Hai đến Chủ nhật. Chọn tối đa 20 sku theo Revenue train, có doanh thu dương "
                    "ở ít nhất 6 tuần train. Tuần thiếu giao dịch được quy ước Revenue bằng 0. "
                    "Đầu vào dự báo chỉ dùng lịch sử Revenue và lịch của tuần cần dự báo.")
        lines += ["", "## Lựa chọn bằng validation và kết quả test", "",
                  "Điểm chọn cấu hình là selection_rmse, trung bình RMSE của ba cửa sổ train mở rộng và validation cuối. "
                  "Cấu hình được chốt trước khi đánh giá test.",
                  f"**{results.loc[results.rmse.idxmin(), 'model']}** có RMSE thấp nhất trên test trong bảng này. "
                  f"**{winner}** là mô hình được chọn từ validation và dùng mặc định để dự đoán. "
                  "Thứ hạng test được dùng để mô tả kết quả, không dùng để thay đổi mô hình đã chọn."]
        context = ("Revenue = price_unit × units_sold, cộng theo sku × tuần qua channel, region và pack_type. "
                   "Tuần bắt đầu thứ Hai. " + context)
    markdown = "\n".join(lines)
    publish_reports(run, results, selection, markdown, context)
    return markdown


def update_history(path, data, run_id):
    if data.empty:
        return
    data = data.assign(Run_ID=run_id)
    if path.exists():
        previous = pd.read_csv(path)
        data = pd.concat([previous.loc[previous.Run_ID != run_id], data], ignore_index=True)
    temporary = path.with_suffix(".tmp")
    data.to_csv(temporary, index=False)
    temporary.replace(path)


def publish_reports(run, results, selection, markdown, context=""):
    """Publish portable MD/HTML/PDF, figures, histories and archive older reports."""
    from reportlab.lib import colors
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.pdfbase import pdfmetrics
    from reportlab.pdfbase.ttfonts import TTFont
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image, PageBreak
    from PIL import Image as PILImage
    from matplotlib import get_data_path

    reports = LAB / "reports"
    reports.mkdir(exist_ok=True)
    stem = "report_" + run.name
    figure_root = reports / "figures" / run.name
    categories = {"daily_sales": "eda", "metrics_comparison": "training"}
    images = []
    for path in sorted((run / "figures").glob("*.png")):
        stage = categories.get(path.stem, "evaluation")
        destination = figure_root / stage / path.name
        destination.parent.mkdir(parents=True, exist_ok=True)
        copy2(path, destination)
        images.append(destination)
        markdown = markdown.replace(f"../results/runs/{run.name}/figures/{path.name}",
                                    destination.relative_to(reports).as_posix())
    markdown += "\n\n## Cấu hình được chọn\n\n"
    for name, parameters in selection.items():
        markdown += f"- **{name}** — {json.dumps(parameters, ensure_ascii=False)}\n"
    eda_image = figure_root / "eda/daily_sales.png"
    if eda_image.exists():
        markdown += f"\n## EDA\n\n![Doanh số theo kỳ]({eda_image.relative_to(reports).as_posix()})\n"
    (reports / f"{stem}.md").write_text(markdown, encoding="utf-8")
    title = "Lab02 - Product Revenue Prediction - " + run.name if "sku_metrics.csv" in markdown else "Lab02 - Product Sales Prediction - " + run.name
    summary = json.loads((run / "preprocessing/summary.json").read_text(encoding="utf-8"))
    note = (f"{summary.get('frequency', 'daily')} target: {summary.get('target', 'sales')}. Configuration selected "
            "using expanding-window and held-out validation RMSE. Final fit uses train plus "
            f"validation. Test predicts one {'week' if summary.get('frequency') == 'weekly' else 'day'} ahead with observed past revenue/sales. " + context)
    analysis = markdown.split("## Kết luận và giới hạn")[-1].split("## Cấu hình được chọn")[0]
    paragraphs = [re.sub(r"\*\*(.*?)\*\*", r"\1", line) for line in analysis.splitlines()
                  if line.strip() and not line.startswith(("|", "##", "!"))]
    winner = pd.read_csv(run / "tuning/validation_metrics.csv").sort_values("selection_rmse", kind="stable").iloc[0]["model"]
    errors = pd.read_csv(run / "evaluation" / f"{winner}_largest_errors.csv").rename(columns={"sku": "StockCode", "Revenue": "sales"}).head(5)
    error_columns = ["StockCode", "date", "sales", "predicted", "absolute_error"]
    if summary.get("revenue_definition", "").startswith("price_unit"):
        errors = errors.rename(columns={"StockCode": "sku", "sales": "Revenue"})
        error_columns = ["sku", "date", "Revenue", "predicted", "absolute_error"]
    analysis_html = "".join("<p>" + escape(line) + "</p>" for line in paragraphs)
    analysis_html += errors[error_columns].to_html(index=False, float_format=lambda v: f"{v:.2f}")
    gallery = "".join(
        f'<figure><img src="{escape(path.relative_to(reports).as_posix())}" '
        f'alt="{escape(path.stem)}"><figcaption>{escape(path.stem)}</figcaption></figure>'
        for path in images)
    html = (
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        f'<title>{escape(title)}</title><style>'
        'body{font:16px system-ui;max-width:1100px;margin:auto;padding:2rem}'
        'table{border-collapse:collapse;width:100%}td,th{padding:.5rem;border:1px solid #ddd}'
        'img{max-width:100%}pre{white-space:pre-wrap}figure{margin:2rem 0}</style>'
        f'<h1>{escape(title)}</h1><p>{escape(note)}</p><h2>Test metrics</h2>'
        + results.to_html(index=False, float_format=lambda v: f"{v:.4f}")
        + '<h2>Selected parameters</h2><pre>'
        + escape(json.dumps(selection, indent=2, ensure_ascii=False))
        + '</pre><h2>Analysis and limitations</h2>' + analysis_html + '<h2>Figures</h2>' + gallery + '</html>')
    (reports / f"{stem}.html").write_text(html, encoding="utf-8")

    # Use the bundled DejaVu font so model names, R2 and Vietnamese text are supported.
    font_dir = Path(get_data_path()) / "fonts/ttf"
    pdfmetrics.registerFont(TTFont("SalesReport", str(font_dir / "DejaVuSans.ttf")))
    pdfmetrics.registerFont(TTFont("SalesReportBold", str(font_dir / "DejaVuSans-Bold.ttf")))
    styles = getSampleStyleSheet()
    for name in ("Normal", "Title", "Heading1", "Heading2"):
        styles[name].fontName = "SalesReportBold" if name != "Normal" else "SalesReport"
    styles.add(ParagraphStyle(name="SmallReport", fontName="SalesReport", fontSize=8, leading=11))
    doc = SimpleDocTemplate(str(reports / f"{stem}.pdf"), pagesize=landscape(A4),
                            leftMargin=35, rightMargin=35, topMargin=32, bottomMargin=32)
    story = [Paragraph(escape(title), styles["Title"]), Spacer(1,12),
             Paragraph(escape(note), styles["Normal"]), Spacer(1,16),
             Paragraph("Test metrics", styles["Heading2"])]
    columns = list(results.columns)
    rows = [[Paragraph(escape(c), styles["SmallReport"]) for c in columns]]
    for row in results.itertuples(index=False, name=None):
        rows.append([Paragraph(escape(f"{v:.4f}" if isinstance(v,float) else str(v)),
                               styles["SmallReport"]) for v in row])
    widths = [170] + [(doc.width-170)/(len(columns)-1)]*(len(columns)-1)
    table = Table(rows, colWidths=widths, repeatRows=1, hAlign="LEFT")
    table.setStyle(TableStyle([("BACKGROUND",(0,0),(-1,0),colors.HexColor("#d9eaf2")),
                               ("GRID",(0,0),(-1,-1),.4,colors.grey),
                               ("VALIGN",(0,0),(-1,-1),"TOP"),
                               ("TOPPADDING",(0,0),(-1,-1),7),
                               ("BOTTOMPADDING",(0,0),(-1,-1),7)]))
    story.extend([table, Spacer(1,12),
                  Paragraph("Selected parameters", styles["Heading2"])])
    for name, parameters in selection.items():
        story.append(Paragraph(escape(name + " " + json.dumps(parameters)), styles["SmallReport"]))
    story.extend([PageBreak(), Paragraph("Analysis and limitations", styles["Heading2"])])
    for line in paragraphs:
        story.append(Paragraph(escape(line), styles["Normal"]))
        story.append(Spacer(1, 9))
    story.append(Paragraph("Largest prediction errors", styles["Heading2"]))
    error_rows = [[Paragraph(escape(c), styles["SmallReport"]) for c in error_columns]]
    for row in errors[error_columns].itertuples(index=False, name=None):
        error_rows.append([Paragraph(escape(f"{v:.2f}" if isinstance(v, float) else str(v)), styles["SmallReport"]) for v in row])
    error_table = Table(error_rows, colWidths=[doc.width / len(error_columns)] * len(error_columns))
    error_table.setStyle(TableStyle([("GRID", (0, 0), (-1, -1), .4, colors.grey), ("BACKGROUND", (0, 0), (-1, 0), colors.HexColor("#d9eaf2")), ("VALIGN", (0, 0), (-1, -1), "TOP")]))
    story.append(error_table)
    for path in images:
        story.extend([PageBreak(),Paragraph(escape(path.stem),styles["Heading2"])])
        with PILImage.open(path) as image:
            width,height = image.size
        scale = min(doc.width/width,(doc.height-45)/height)
        story.append(Image(str(path),width=width*scale,height=height*scale))
    doc.build(story)

    update_history(reports / "metrics_summary.csv", results, run.name)
    trials = pd.read_csv(run / "tuning/validation_metrics.csv")
    update_history(reports / "validation_summary.csv", trials, run.name)
    archive = reports / "archive"
    archive.mkdir(exist_ok=True)
    # Report files contain run IDs, so preserve every run when regenerating another report.
    for path in list(reports.iterdir()):
        is_report = path.name.startswith("report_") or re.fullmatch(
            r"(?:\d{8}T\d{6}Z-[a-f0-9]{8}|\d{4}-\d{2}-\d{2}_\d{2}-\d{2}-\d{2}(?:_\d+)?).md",path.name)
        if not path.is_file() or not is_report or path.suffix not in {".md",".html",".pdf"}:
            continue
        if path.stem == stem:
            continue
        if path.suffix in {".md",".html"}:
            text = path.read_text(encoding="utf-8")
            text = text.replace("../results/runs/", "../../results/runs/")
            text = re.sub(r'(?<![./])figures/', "../figures/", text)
            path.write_text(text,encoding="utf-8")
        path.replace(archive / path.name)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-id", required=True)
    args = parser.parse_args()
    runs = (LAB / "results/runs").resolve()
    run = (runs / args.run_id).resolve()
    if run.parent != runs or not (run / "run.json").is_file():
        parser.error("Unknown run ID.")
    if json.loads((run / "run.json").read_text())["status"] != "success":
        parser.error("Reports require a completed successful run.")
    results = pd.read_csv(run / "evaluation/metrics.csv")
    selection_path = run / "tuning/selection.json"
    selection = json.loads(selection_path.read_text())
    markdown = generate(run, results, selection)
    (run / "report.md").write_text(
        markdown.replace(f"../results/runs/{run.name}/figures/", "figures/"),encoding="utf-8")
    print(LAB / "reports" / f"report_{run.name}.html")


if __name__ == "__main__":
    main()
