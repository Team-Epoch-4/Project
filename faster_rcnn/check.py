import pandas as pd

train_df = pd.read_csv("data/train_df.csv")
print(train_df["image_name"].unique()[:5])  # 상위 5개만 미리 확인
print(f"총 이미지 수: {train_df['image_name'].nunique()}")
