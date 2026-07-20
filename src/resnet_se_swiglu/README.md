# ResNet-SE-SwiGLU

ResNet1D với squeeze-excitation trong mỗi residual block và SwiGLU trước global pooling.

## Kaggle Setup

Chọn accelerator `GPU T4 x2` rồi chạy:

```python
!git clone https://github.com/vinhveer/sensor_signal_analyzer.git
%cd sensor_signal_analyzer
!pip install -q -r requirements.txt
!pip install -q -e .
```

Dataset root:

```text
/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10
```

Thư mục này phải chứa trực tiếp `Dinh`, `Giang`, `Healthy`, `Nut`. Nếu các class nằm trực tiếp trong `dataset-1-10`, bỏ phần `/dataset_1_10`.

## Train AMP

```python
!python -m resnet_se_swiglu.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/resnet_se_swiglu_amp" \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 2 \
  --amp \
  --n-feature-maps 64 \
  --se-ratio 0.0625 \
  --ffn-ratio 2.6666667 \
  --dropout 0.01 \
  --seeds 42
```

## Các Kiểu Train

FP32 để so sánh tốc độ và metric:

```python
!python -m resnet_se_swiglu.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/resnet_se_swiglu_fp32" \
  --epochs 100 --batch-size 32 --num-workers 2 --no-amp --seeds 42
```

Nhiều seed:

```python
!python -m resnet_se_swiglu.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/resnet_se_swiglu_final" \
  --epochs 100 --batch-size 32 --num-workers 2 --amp \
  --seeds 42,43,44,45,46
```

Smoke test:

```python
!python -m resnet_se_swiglu.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/resnet_se_swiglu_smoke" \
  --epochs 1 --batch-size 32 --num-workers 0 --amp --no-save-zip
```

Ablation overlap, chỉ chạy sau baseline:

```python
!python -m resnet_se_swiglu.train \
  --dataset-root "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10" \
  --output-root "/kaggle/working/resnet_se_swiglu_overlap50" \
  --epochs 100 --batch-size 32 --num-workers 2 --amp --overlap 0.5 --seeds 42
```

## T4 x2

Một job chưa dùng DDP và chỉ dùng một T4. Để chạy hai seed song song:

```python
import subprocess

root = "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10"
base = "python -m resnet_se_swiglu.train --epochs 100 --batch-size 32 --num-workers 2 --amp"
jobs = [
    subprocess.Popen(f"CUDA_VISIBLE_DEVICES=0 {base} --dataset-root '{root}' --output-root /kaggle/working/resnet_se_swiglu_seed42 --seeds 42", shell=True),
    subprocess.Popen(f"CUDA_VISIBLE_DEVICES=1 {base} --dataset-root '{root}' --output-root /kaggle/working/resnet_se_swiglu_seed43 --seeds 43", shell=True),
]
[job.wait() for job in jobs]
```

Hai process phải dùng output root khác nhau. Nếu CPU/RAM trở thành bottleneck, giảm mỗi process xuống `--num-workers 1`.

## Inference

```python
!python -m resnet_se_swiglu.inference \
  --checkpoint "/kaggle/working/resnet_se_swiglu_amp/seed42/best_dataset_1_10_resnet_se_swiglu.pt" \
  --input "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10/Healthy/Processed/NS0R6.npy"
```

## Visualization

```python
!python -m resnet_se_swiglu.visualize \
  --checkpoint "/kaggle/working/resnet_se_swiglu_amp/seed42/best_dataset_1_10_resnet_se_swiglu.pt" \
  --input "/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10/Healthy/Processed/NS0R6.npy" \
  --output-dir "/kaggle/working/resnet_se_swiglu_visualization"
```

Sinh `inference_NS0R6.json` và `inference_NS0R6.png`. Biểu đồ gồm raw signal, mean class probability và class dự đoán theo từng window.
