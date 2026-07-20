# MLSTM-FCN

Mô hình hai nhánh: LSTM xử lý `(batch, time, channels)` và FCN-SE xử lý `(batch, channels, time)`.

## Kaggle Setup

Chọn accelerator `GPU T4 x2`:

```python
!git clone https://github.com/vinhveer/sensor_signal_analyzer.git
%cd sensor_signal_analyzer
!pip install -q -r requirements.txt
!pip install -q -e .
```

Dataset root dự kiến:

```text
/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10
```

Kiểm tra root có trực tiếp `Dinh`, `Giang`, `Healthy`, `Nut`. Nếu chúng nằm ngay trong `dataset-1-10`, dùng đường dẫn cha.

## Train AMP

```python
!python -m mlstmfcn.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/mlstmfcn_amp" \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 2 \
  --amp \
  --lstm-hidden 128 \
  --lstm-layers 1 \
  --conv1-channels 128 \
  --conv2-channels 256 \
  --conv3-channels 128 \
  --lstm-dropout 0.2 \
  --conv-dropout 0.1 \
  --se-reduction 16 \
  --seeds 42
```

## Các Kiểu Train

FP32 baseline:

```python
!python -m mlstmfcn.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/mlstmfcn_fp32" \
  --epochs 100 --batch-size 32 --num-workers 2 --no-amp --seeds 42
```

Nhiều seed:

```python
!python -m mlstmfcn.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/mlstmfcn_final" \
  --epochs 100 --batch-size 32 --num-workers 2 --amp \
  --seeds 42,43,44,45,46
```

Smoke test:

```python
!python -m mlstmfcn.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/mlstmfcn_smoke" \
  --epochs 1 --batch-size 32 --num-workers 0 --amp --no-save-zip
```

Batch-size ablation:

```python
!python -m mlstmfcn.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/mlstmfcn_batch64" \
  --epochs 100 --batch-size 64 --num-workers 2 --amp --seeds 42
```

## T4 x2

Pipeline chưa có DDP, nên một job chỉ dùng một T4. Có thể chạy hai seed độc lập song song:

```python
import subprocess

root = "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10"
base = "python -m mlstmfcn.train --epochs 100 --batch-size 32 --num-workers 1 --amp"
jobs = [
    subprocess.Popen(f"CUDA_VISIBLE_DEVICES=0 {base} --dataset-root '{root}' --output-root /kaggle/working/mlstmfcn_seed42 --seeds 42", shell=True),
    subprocess.Popen(f"CUDA_VISIBLE_DEVICES=1 {base} --dataset-root '{root}' --output-root /kaggle/working/mlstmfcn_seed43 --seeds 43", shell=True),
]
[job.wait() for job in jobs]
```

MLSTM-FCN có nhánh LSTM nên mức tăng AMP có thể khác CNN thuần. So sánh `epoch_time_seconds`, F1 và accuracy giữa AMP/FP32.

## Inference

```python
!python -m mlstmfcn.inference \
  --checkpoint "/kaggle/working/mlstmfcn_amp/seed42/best_dataset_1_10_mlstmfcn.pt" \
  --input "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10/Healthy/Processed/NS0R6.npy"
```

## Visualization

```python
!python -m mlstmfcn.visualize \
  --checkpoint "/kaggle/working/mlstmfcn_amp/seed42/best_dataset_1_10_mlstmfcn.pt" \
  --input "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10/Healthy/Processed/NS0R6.npy" \
  --output-dir "/kaggle/working/mlstmfcn_visualization"
```

Sinh `inference_NS0R6.json` và `inference_NS0R6.png`. JSON chứa cả kết quả tổng hợp và kết quả từng sliding window; PNG gồm signal, probability bar chart và window prediction timeline.
