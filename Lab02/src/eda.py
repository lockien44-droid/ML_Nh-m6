"""Export daily demand summaries."""
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


def analyze(frame, output):
    frame.groupby("StockCode").sales.describe().to_csv(output / "product_statistics.csv")
    frame.groupby("date").sales.sum().plot(figsize=(11, 4), title="Daily units sold - selected products")
    plt.ylabel("Units")
    plt.tight_layout()
    plt.savefig(output / "daily_sales.png")
    plt.close()
