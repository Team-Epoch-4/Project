import os
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import warnings
from pathlib import Path

# 📌 기본 설정
mpl.rcParams['font.family'] = 'DejaVu Sans'
mpl.rcParams['axes.unicode_minus'] = False
warnings.filterwarnings("ignore", category=UserWarning, module='matplotlib')

# ✅ BASE_DIR: yolov11/
BASE_DIR = Path(__file__).resolve().parent.parent

# 📁 저장 폴더 생성 (없으면 자동 생성)
PLOT_DIR = BASE_DIR / "results" / "plots"
PLOT_DIR.mkdir(parents=True, exist_ok=True)

# 📌 모델별 결과 경로
base_paths = {
    "YOLOv11-s": BASE_DIR / "runs" / "yolov11s",
    "YOLOv11-m": BASE_DIR / "runs" / "yolov11m",
    "YOLOv11-l": BASE_DIR / "runs" / "yolov11l"
}

# 📌 성능 지표 키와 라벨
metrics = {
    "metrics/mAP50(B)": "mAP@0.5",
    "metrics/precision(B)": "Precision",
    "metrics/recall(B)": "Recall",
    "val/box_loss": "Box Loss"
}

# 📌 최신 results.csv 찾기
def find_latest_exp_csv(base_dir):
    if not base_dir.exists():
        return None
    exp_dirs = [d for d in base_dir.iterdir() if d.is_dir() and d.name.startswith("exp")]
    exp_dirs = sorted(exp_dirs, key=lambda x: x.stat().st_mtime, reverse=True)
    for exp in exp_dirs:
        csv_path = exp / "results.csv"
        if csv_path.exists():
            return csv_path
    return None

# 📊 그래프 시각화 및 저장
for metric_key, metric_label in metrics.items():
    print(f"\n📊 Metric: {metric_label} (비교 차트)")
    plt.figure(figsize=(10, 6))
    valid_plot = False

    for model_name, base_dir in base_paths.items():
        csv_path = find_latest_exp_csv(base_dir)
        if csv_path:
            df = pd.read_csv(csv_path)
            if metric_key not in df.columns:
                print(f"❌ 컬럼 '{metric_key}' 없음 in {csv_path}")
                continue
            plt.plot(df["epoch"], df[metric_key], label=model_name)
            valid_plot = True
        else:
            print(f"❌ results.csv 없음 in {base_dir}")

    if valid_plot:
        plt.title(f"{metric_label} Comparison")
        plt.xlabel("Epoch")
        plt.ylabel(metric_label)
        plt.legend(loc="lower right")
        plt.grid(True)
        plt.tight_layout()

        # mAP, Recall 범위 조정
        if "mAP" in metric_key or "recall" in metric_key.lower():
            plt.ylim(0.9, 1.0)

        save_name = metric_label.replace(" ", "_").lower() + ".png"
        plt.savefig(PLOT_DIR / save_name)
        print(f"📁 그래프 저장됨: {save_name}")
        plt.close()
    else:
        print("⚠️ 유효한 데이터를 찾지 못해 시각화를 건너뜀.")

# 📊 요약 테이블 생성
summary = []
for model_name, base_dir in base_paths.items():
    csv_path = find_latest_exp_csv(base_dir)
    if not csv_path:
        print(f"❌ {model_name}: results.csv not found.")
        continue

    df = pd.read_csv(csv_path)
    metrics_result = {
        "Model": model_name,
        "Best mAP@0.5": df["metrics/mAP50(B)"].max(),
        "Best Precision": df["metrics/precision(B)"].max(),
        "Best Recall": df["metrics/recall(B)"].max(),
        "Min Box Loss": df["val/box_loss"].min(),
        "Total Time (s)": df["time"].sum()
    }
    summary.append(metrics_result)

# 📊 요약 결과 출력 및 저장
if summary:
    summary_df = pd.DataFrame(summary)
    summary_df["Total Time (min)"] = (summary_df["Total Time (s)"] / 60).round(1)
    summary_df = summary_df.sort_values(by="Best mAP@0.5", ascending=False).reset_index(drop=True)

    print("\n✅ YOLOv11 모델 성능 비교 요약:\n")
    print(summary_df.to_string(index=False))

    # ✅ CSV 저장도 가능
    summary_df.to_csv(BASE_DIR / "results" / "summary.csv", index=False)
    print("📁 summary.csv 저장 완료")
else:
    print("\n❗ 요약할 results.csv 데이터를 찾을 수 없습니다.")