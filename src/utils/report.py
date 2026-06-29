from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt


REPORT_DIR = Path("reports/reports")
REPORT_DIR.mkdir(parents=True, exist_ok=True)


def save_model_report(results):

    df = pd.DataFrame(results)

    # Sort by F1 Score
    df = df.sort_values(
        by="F1",
        ascending=False
    ).reset_index(drop=True)

    csv_path = REPORT_DIR / "model_comparison.csv"

    md_path = REPORT_DIR / "model_comparison.md"

    png_path = REPORT_DIR / "model_comparison.png"

    ########################################
    # CSV
    ########################################

    df.to_csv(
        csv_path,
        index=False
    )

    ########################################
    # Markdown
    ########################################

    markdown = "# Model Comparison Report\n\n"

    markdown += df.to_markdown(index=False)

    md_path.write_text(
        markdown,
        encoding="utf-8"
    )

    ########################################
    # PNG Table
    ########################################

    fig, ax = plt.subplots(
        figsize=(12, 2 + len(df) * 0.5)
    )

    ax.axis("off")

    table = ax.table(
        cellText=df.round(4).values,
        colLabels=df.columns,
        loc="center"
    )

    table.auto_set_font_size(False)
    table.set_fontsize(10)
    table.scale(1.2, 1.6)

    plt.savefig(
        png_path,
        dpi=300,
        bbox_inches="tight"
    )

    plt.close()

    print("\nReports generated successfully")

    print(csv_path)

    print(md_path)

    print(png_path)