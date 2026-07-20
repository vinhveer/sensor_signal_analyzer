# ResNet1D

ResNet1D ba residual blocks cho tín hiệu đầu vào `(batch, 2, 1024)`.

## Kaggle Setup

Trong Notebook Settings, chọn accelerator `GPU T4 x2`, sau đó:

```python
!git clone https://github.com/vinhveer/sensor_signal_analyzer.git
%cd sensor_signal_analyzer
!pip install -q -r requirements.txt
!pip install -q -e .
```

Dataset root phải chứa trực tiếp bốn class folder. Kiểm tra:

```python
from pathlib import Path

DATASET_ROOT = Path("/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10")
print([path.name for path in DATASET_ROOT.iterdir()])
```

Kết quả cần có `Dinh`, `Giang`, `Healthy`, `Nut`. Nếu bốn folder nằm trực tiếp trong `dataset-1-10`, dùng đường dẫn cha thay cho `DATASET_ROOT`.

## Train AMP

Khuyến nghị đầu tiên trên T4:

```python
!python -m resnet1d.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/resnet1d_amp" \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 2 \
  --amp \
  --n-feature-maps 64 \
  --seeds 42
```

## Các Kiểu Train

FP32 baseline:

```python
!python -m resnet1d.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/resnet1d_fp32" \
  --epochs 100 --batch-size 32 --num-workers 2 --no-amp --seeds 42
```

Nhiều seed để báo cáo kết quả ổn định:

```python
!python -m resnet1d.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/resnet1d_final" \
  --epochs 100 --batch-size 32 --num-workers 2 --amp \
  --seeds 42,43,44,45,46
```

Smoke test trước khi train dài:

```python
!python -m resnet1d.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/resnet1d_smoke" \
  --epochs 1 --batch-size 32 --num-workers 0 --amp --no-save-zip
```

## T4 x2

Một lệnh train hiện dùng một GPU. Pipeline chưa có DDP. Có thể tận dụng hai T4 bằng hai experiment độc lập:

```python
import subprocess

root = "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10"
jobs = [
    subprocess.Popen(f"CUDA_VISIBLE_DEVICES=0 python -m resnet1d.train --dataset-root '{root}' --output-root /kaggle/working/resnet1d_seed42 --epochs 100 --batch-size 32 --num-workers 2 --amp --seeds 42", shell=True),
    subprocess.Popen(f"CUDA_VISIBLE_DEVICES=1 python -m resnet1d.train --dataset-root '{root}' --output-root /kaggle/working/resnet1d_seed43 --epochs 100 --batch-size 32 --num-workers 2 --amp --seeds 43", shell=True),
]
[job.wait() for job in jobs]
```

Không dùng cùng `--output-root` cho hai process.

## Inference

```python
!python -m resnet1d.inference \
  --checkpoint "/kaggle/working/resnet1d_amp/seed42/best_dataset_1_10_resnet1D.pt" \
  --input "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10/Healthy/Processed/NS0R6.npy"
```

## Visualization

```python
!python -m resnet1d.visualize \
  --checkpoint "/kaggle/working/resnet1d_amp/seed42/best_dataset_1_10_resnet1D.pt" \
  --input "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10/Healthy/Processed/NS0R6.npy" \
  --output-dir "/kaggle/working/resnet1d_visualization"
```

Output:

```text
resnet1d_visualization/
├── inference_NS0R6.json
└── inference_NS0R6.png
```

PNG gồm tín hiệu hai kênh, xác suất trung bình theo class và prediction của từng sliding window. JSON lưu đầy đủ class, probabilities, window starts và probabilities theo window.
