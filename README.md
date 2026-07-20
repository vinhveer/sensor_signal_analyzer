# Sensor Signal Analyzer

Phân loại tín hiệu cảm biến hai kênh theo cửa sổ thời gian bằng PyTorch.

## Cấu trúc

```text
src/
├── lib/                         Thư viện dùng chung, không biết model cụ thể
│   ├── apps.py                  Abstract TrainApp và InferenceApp
│   ├── config/                  Config mặc định và validation
│   ├── data/                    Discovery, conversion, Dataset, DataLoader
│   ├── engine/                  Train, evaluate, checkpoint, metrics, report
│   ├── utils/                   Seed, JSON và archive
│   └── visualization/           Learning curve và confusion matrix
│
└── resnet1d/                    Một model độc lập
    ├── model.py                 Kiến trúc, chỉ phụ thuộc PyTorch
    ├── train.py                 CLI train kế thừa lib.TrainApp
    └── inference.py             CLI inference kế thừa lib.InferenceApp

src/resnet_se_swiglu/
├── model.py                     ResNet1D + SE + SwiGLU
├── train.py                     CLI train riêng
└── inference.py                 CLI inference riêng

src/mlstmfcn/
├── model.py                     MLSTM-FCN + SE
├── train.py                     CLI train riêng
└── inference.py                 CLI inference riêng
```

Dependency chỉ đi theo một chiều:

```text
resnet1d/train.py ────────┐
resnet1d/inference.py ────┼──> lib
resnet1d/model.py ────────┘    không import lib
```

## Cài đặt

```bash
python -m venv .venv
./.venv/bin/pip install -r requirements.txt
./.venv/bin/pip install -e .
```

## Dataset

```text
dataset_root/
├── Dinh/Processed/*.npy
├── Giang/Processed/*.npy
├── Healthy/Processed/*.npy
└── Nut/Processed/*.npy
```

Mỗi `.npy` có shape `(time, 2)`. Tên file chứa run ID:

```text
R1-R4: train
R5:    validation
R6:    test
```

Trên Kaggle, dataset dự kiến nằm tại:

```text
/kaggle/input/datasets/platformciviltwin/dataset-1-10/dataset_1_10
```

`--dataset-root` phải trỏ tới thư mục chứa trực tiếp `Dinh`, `Giang`, `Healthy`, `Nut`. Nếu các class nằm trực tiếp trong `/kaggle/input/datasets/platformciviltwin/dataset-1-10`, hãy dùng đường dẫn đó thay vì nối thêm `/dataset_1_10`.

## Hướng Dẫn Theo Model

- [ResNet1D](src/resnet1d/README.md)
- [ResNet-SE-SwiGLU](src/resnet_se_swiglu/README.md)
- [MLSTM-FCN](src/mlstmfcn/README.md)

Kiểm tra Kaggle nhận đủ hai T4:

```python
import torch

print(torch.cuda.device_count())
for index in range(torch.cuda.device_count()):
    print(index, torch.cuda.get_device_name(index))
```

Pipeline hiện chưa có DDP: một lệnh train dùng một T4. README của từng model có ví dụ dùng `CUDA_VISIBLE_DEVICES=0/1` để chạy hai seed hoặc hai experiment độc lập cùng lúc trên T4 x2.

Mỗi model có đủ ba command:

```text
python -m <model>.train       Train và sinh checkpoint/learning curve/confusion matrix
python -m <model>.inference   Load checkpoint và trả class/probabilities
python -m <model>.visualize   Load checkpoint và sinh inference JSON/PNG
```

## Train ResNet1D

Sau khi `pip install -e .`:

```bash
python -m resnet1d.train \
  --dataset-root "/Users/nguyenquangvinh/Desktop/project_ce/dataset_1_10" \
  --output-root History \
  --epochs 100 \
  --batch-size 32 \
  --n-feature-maps 64
```

Nếu chưa cài editable package:

```bash
PYTHONPATH=src python -m resnet1d.train --dataset-root /path/to/dataset
```

## Inference

Inference trên một file tín hiệu `.npy`:

```bash
python -m resnet1d.inference \
  --checkpoint History/seed42/best_dataset_1_10_resnet1D.pt \
  --input /path/to/signal.npy
```

Pipeline tự cắt sliding window, chuẩn hóa, chạy từng batch và lấy trung bình xác suất của các cửa sổ.

## Thêm Model

Tạo package ngang hàng với `resnet1d`:

```text
src/newmodel/
├── __init__.py
├── model.py
├── train.py
└── inference.py
```

`model.py` chỉ chứa kiến trúc PyTorch, không phụ thuộc `lib`:

```python
from torch import nn


class NewModel(nn.Module):
    def __init__(self, in_channels, num_classes, window, hidden_size=64):
        super().__init__()
        self.network = ...

    def forward(self, x):
        return self.network(x)
```

`train.py` chỉ nối tham số riêng của model vào pipeline chung:

```python
import argparse

from lib.apps import TrainApp
from .model import NewModel


class NewModelTrainApp(TrainApp):
    model_name = "newmodel"

    def add_model_args(self, parser: argparse.ArgumentParser):
        parser.add_argument("--hidden-size", type=int, default=64)

    def model_config(self, args):
        return {"hidden_size": args.hidden_size}

    def build_model(self, in_channels, num_classes, window, config):
        return NewModel(in_channels, num_classes, window, config["hidden_size"])


if __name__ == "__main__":
    NewModelTrainApp().run()
```

Model mới tự dùng lại data loading, optimizer, loss, scheduler, checkpoint, metrics và toàn bộ biểu đồ trong `lib`.

Lệnh module chuẩn của Python là `python -m resnet1d.train`, không phải `python resnet1d.train`.

## Train ResNet-SE-SwiGLU

Chạy trên dataset local:

```bash
./.venv/bin/python -m resnet_se_swiglu.train \
  --dataset-root "/Users/nguyenquangvinh/Desktop/project_ce/dataset_1_10" \
  --output-root "History/resnet_se_swiglu" \
  --epochs 100 \
  --batch-size 32 \
  --window 1024 \
  --overlap 0.75 \
  --n-feature-maps 64 \
  --se-ratio 0.0625 \
  --ffn-ratio 2.6666667 \
  --dropout 0.01 \
  --seeds 42
```

Máy local không có CUDA sẽ chạy FP32. Trên Kaggle T4, bật AMP và DataLoader workers:

```bash
python -m resnet_se_swiglu.train \
  --dataset-root "/kaggle/input/.../dataset_1_10" \
  --output-root "/kaggle/working/resnet_se_swiglu_amp" \
  --epochs 100 \
  --batch-size 32 \
  --num-workers 2 \
  --amp \
  --n-feature-maps 64 \
  --seeds 42
```

Để benchmark FP32 công bằng, chạy cùng cấu hình với `--no-amp` và output root khác. History lưu:

```text
train_time_seconds
val_time_seconds
epoch_time_seconds
train_samples_per_second
peak_gpu_memory_mb
use_amp
```

Train cấu hình cuối bằng nhiều seed:

```bash
./.venv/bin/python -m resnet_se_swiglu.train \
  --dataset-root "/Users/nguyenquangvinh/Desktop/project_ce/dataset_1_10" \
  --output-root "History/resnet_se_swiglu_final" \
  --epochs 100 \
  --batch-size 32 \
  --n-feature-maps 64 \
  --seeds 42,43,44,45,46
```

Inference:

```bash
./.venv/bin/python -m resnet_se_swiglu.inference \
  --checkpoint "History/resnet_se_swiglu/seed42/best_dataset_1_10_resnet_se_swiglu.pt" \
  --input "/path/to/signal.npy"
```

Chưa bật DDP, GPU normalization hoặc temporal downsampling. Các thay đổi này cần benchmark/profiler hoặc ablation trước vì dataset hiện chỉ có 765 training windows.

## Train MLSTM-FCN

Package `mlstmfcn` dùng chung pipeline dữ liệu. Mỗi batch vào model có shape `(batch, 2, 1024)`; nhánh CNN dùng trực tiếp tensor này, nhánh LSTM transpose thành `(batch, 1024, 2)`.

Train với cấu hình từ model gốc:

```bash
./.venv/bin/python -m mlstmfcn.train \
  --dataset-root "/Users/nguyenquangvinh/Desktop/project_ce/dataset_1_10" \
  --output-root "History/mlstmfcn" \
  --epochs 100 \
  --batch-size 32 \
  --window 1024 \
  --overlap 0.75 \
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

Trên Kaggle T4 có thể thêm:

```text
--amp --num-workers 2
```

Inference:

```bash
./.venv/bin/python -m mlstmfcn.inference \
  --checkpoint "History/mlstmfcn/seed42/best_dataset_1_10_mlstmfcn.pt" \
  --input "/path/to/signal.npy"
```
