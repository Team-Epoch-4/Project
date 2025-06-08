# main.py
import subprocess

scripts = [
    "merge_to_coco.py",
    "class_analysis.py",
    "augment_rare_classes.py",
    "make_class_mappings.py",
    "convert_to_yolo.py",
    "train.py",
    "predict.py"
]

if __name__ == "__main__":
    for i, script in enumerate(scripts):
        print(f"\nStep {i}: 실행 중 → {script}")
        try:
            subprocess.run(["python", script], check=True)
        except subprocess.CalledProcessError as e:
            print(f"{script} 실행 중 오류 발생:")
            print(e)
            break
    print("\n전체 파이프라인 실행 완료")