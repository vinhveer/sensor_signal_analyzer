# AGENTS.md

## 1. Mục tiêu của project

Project này dùng để train mô hình deep learning cho bài toán **time-series / sensor classification**.

Bài toán hiện tại:

* Input là tín hiệu sensor dạng chuỗi thời gian.
* Mỗi sample/window có độ dài `1024` time steps.
* Model hiện tại là `1DCNN`.
* Dữ liệu có thể đến từ:

  * Sensor 1
  * Sensor 2
  * Sensor 1 + Sensor 2
* Output là nhãn phân loại thuộc các class của dataset.
* Project phải chạy được trên Kaggle sau khi clone từ GitHub.
* Sau khi clone repo, người dùng chỉ cần chạy một lệnh CLI để bắt đầu train và sinh kết quả.

Mục tiêu refactor:

> Refactor file code dài hiện tại thành một project Python có cấu trúc rõ ràng, dễ maintain, dễ mở rộng model, dễ chạy trên Kaggle, và không làm thay đổi logic training/evaluation hiện tại nếu không được yêu cầu.

---

## 2. File code gốc cần refactor

File code gốc:

```text
train_1DCNN.py
```

File này hiện đang chứa nhiều trách nhiệm trong cùng một script:

* Khai báo constants/config mặc định.
* Detect dataset root trên Kaggle.
* Convert dữ liệu CSV sang `.npy` nếu chưa có `.npy`.
* Đọc CSV linh hoạt với nhiều separator.
* Chọn cột acceleration.
* Load `.npy`.
* Cắt sliding windows.
* Normalize từng window.
* Tạo PyTorch Dataset và DataLoader.
* Split train/val/test theo run id trong filename.
* Định nghĩa model `Conv1DClassifier`.
* Định nghĩa custom layer `Conv1dSame`.
* Train model.
* Evaluate model.
* Tính metrics.
* Vẽ learning curve.
* Vẽ confusion matrix.
* Lưu checkpoint.
* Lưu history, scores, run_info.
* Zip toàn bộ kết quả.

Không được xóa các chức năng này. Chỉ được tách chúng ra các module/folder phù hợp.

---

## 3. Yêu cầu đầu ra sau khi refactor

Sau khi refactor, project cần có cấu trúc như sau:

```text
sensor_time_series_classification/
│
├── README.md
├── AGENTS.md
├── requirements.txt
├── pyproject.toml
│
├── configs/
│   └── kaggle_1dcnn_s12.yaml
│
├── data/
│   └── .gitkeep
│
├── src/
│   └── sensorcls/
│       ├── __init__.py
│       │
│       ├── cli/
│       │   ├── __init__.py
│       │   └── train.py
│       │
│       ├── config/
│       │   ├── __init__.py
│       │   └── loader.py
│       │
│       ├── data/
│       │   ├── __init__.py
│       │   ├── constants.py
│       │   ├── conversion.py
│       │   ├── discovery.py
│       │   ├── dataset.py
│       │   ├── preprocessing.py
│       │   └── dataloaders.py
│       │
│       ├── models/
│       │   ├── __init__.py
│       │   └── cnn1d.py
│       │
│       ├── engine/
│       │   ├── __init__.py
│       │   ├── train.py
│       │   ├── evaluate.py
│       │   └── metrics.py
│       │
│       ├── utils/
│       │   ├── __init__.py
│       │   ├── seed.py
│       │   ├── amp.py
│       │   ├── io.py
│       │   └── archive.py
│       │
│       └── visualization/
│           ├── __init__.py
│           ├── curves.py
│           └── confusion_matrix.py
│
├── scripts/
│   └── train.py
│
├── outputs/
│   └── .gitkeep
│
├── reports/
│   └── .gitkeep
│
└── tests/
    ├── test_dataset.py
    ├── test_model_forward.py
    └── test_metrics.py
```

---

## 4. Nguyên tắc refactor bắt buộc

### 4.1. Không thay đổi hành vi nếu không cần thiết

AI Agent phải ưu tiên giữ nguyên logic hiện tại.

Không tự ý thay đổi:

* Window size.
* Overlap.
* Step size.
* Batch size.
* Learning rate.
* Optimizer.
* Scheduler.
* Model architecture.
* Data split logic.
* Metric logic.
* Cách lưu output.
* Cách đặt tên artifact.
* Cách detect Kaggle path.

Nếu cần thay đổi để code sạch hơn, phải ghi rõ trong commit/message.

---

### 4.2. Không hard-code rải rác

Các giá trị cấu hình không được hard-code ở nhiều file khác nhau.

Những giá trị sau phải đưa vào file config YAML:

```yaml
dataset_name: dataset_1_10
model_name: 1dcnn

data:
  kaggle_dataset_subdir: Processed
  kaggle_working_dataset_root: /kaggle/working/dataset_1_10
  kaggle_input_root_candidates:
    - /kaggle/input/datasets/thanhhieu03092004/data-set-1-10/dataset_1_10
  expected_channels: 2
  acc_columns:
    - "Acceleration - x (m/s²)"
    - "Acceleration - x (m/s²).1"
  source_subdir_candidates:
    - "."
    - "Processed"
  csv_sep_candidates:
    - ","
    - ";"
    - null

windowing:
  window: 1024
  overlap: 0.75
  return_ct: true
  eps: 1e-8

split:
  expected_files_per_class: 18
  train_run_ids: [1, 2, 3, 4]
  val_run_ids: [5]
  test_run_ids: [6]
  split_seed: 42

training:
  seeds: [42]
  batch_size: 32
  epochs: 100
  learning_rate: 0.0001
  weight_decay: 0.001
  num_workers: 0
  use_amp: false
  grad_clip_norm: 1.0

model:
  name: cnn1d
  fc_dim: 128
  dropout: 0.4

outputs:
  root: /kaggle/working/History
  save_zip: true
```

---

## 5. Mapping từ code cũ sang file mới

Khi refactor, hãy di chuyển function/class theo mapping dưới đây.

### 5.1. `src/sensorcls/data/constants.py`

Chứa các constant mặc định nếu cần.

Tuy nhiên, ưu tiên đọc từ YAML config.

Có thể chứa:

```python
DEFAULT_CSV_SEP_CANDIDATES = [",", ";", None]
DEFAULT_SOURCE_SUBDIR_CANDIDATES = [".", "Processed"]
```

Không đặt toàn bộ config training ở đây nếu đã có YAML.

---

### 5.2. `src/sensorcls/data/discovery.py`

Chứa các function liên quan đến phát hiện dataset root.

Di chuyển các function sau vào đây:

```python
class_dirs
find_source_dir_for_class
detect_npy_dataset_root
detect_source_root
prepare_dataset_root
```

Vai trò:

* Tìm dataset `.npy` có sẵn.
* Nếu chưa có `.npy`, tìm source CSV.
* Hỗ trợ môi trường Kaggle.
* Trả về `dataset_root` cuối cùng để training dùng.

---

### 5.3. `src/sensorcls/data/conversion.py`

Chứa logic convert CSV sang NPY.

Di chuyển các function sau vào đây:

```python
normalize_col_name
read_csv_flexible
select_acc_columns
load_source_array
convert_source_to_npy
```

Vai trò:

* Đọc CSV.
* Tự detect separator.
* Chọn đúng cột acceleration.
* Convert dữ liệu sensor sang NumPy array.
* Lưu `.npy`.

Không để logic train trong file này.

---

### 5.4. `src/sensorcls/data/preprocessing.py`

Chứa logic xử lý window.

Di chuyển các function sau vào đây:

```python
starts_for_length
preprocess_window
```

Vai trò:

* Tính các vị trí bắt đầu của sliding window.
* Normalize từng window bằng local mean/std.
* Đảm bảo output có dtype `np.float32`.

---

### 5.5. `src/sensorcls/data/dataset.py`

Chứa PyTorch Dataset và dataclass cấu hình dữ liệu.

Di chuyển các thành phần sau vào đây:

```python
DataConfig
class_dir
list_npy_files
load_npy_mmap
load_window
build_index
WindowDataset
run_id_from_filename
split_files_for_class
```

Vai trò:

* Quản lý dữ liệu `.npy`.
* Cắt dữ liệu thành windows.
* Trả về sample `(x, y)`.
* Split file theo run id.
* Không tạo DataLoader trong file này.

---

### 5.6. `src/sensorcls/data/dataloaders.py`

Chứa function tạo DataLoader.

Di chuyển function sau vào đây:

```python
make_loaders
```

Vai trò:

* Nhận `classes`, `DataConfig`, `batch_size`, `num_workers`, `loader_seed`.
* Tạo `train_loader`, `val_loader`, `test_loader`.
* Trả thêm `meta` để lưu vào output.

---

### 5.7. `src/sensorcls/models/cnn1d.py`

Chứa model 1D CNN.

Di chuyển các class sau vào đây:

```python
Conv1dSame
Conv1DClassifier
```

Vai trò:

* Định nghĩa architecture model.
* Không chứa train loop.
* Không chứa data loading.
* Không chứa plotting.

Nếu thêm model mới sau này, tạo file mới trong folder `models/`, ví dụ:

```text
models/lstm.py
models/resnet1d.py
models/transformer1d.py
```

---

### 5.8. `src/sensorcls/engine/evaluate.py`

Chứa logic evaluate model.

Di chuyển function sau vào đây:

```python
evaluate_model
```

Vai trò:

* Chạy model trên validation/test loader.
* Trả về loss, accuracy, y_true, y_pred, y_probs.
* Không vẽ biểu đồ trong file này.

---

### 5.9. `src/sensorcls/engine/metrics.py`

Chứa logic tính metric.

Di chuyển function sau vào đây:

```python
build_scores
```

Vai trò:

* Tính accuracy.
* Tính precision.
* Tính recall.
* Tính f1-score.
* Tính confusion matrix.

---

### 5.10. `src/sensorcls/engine/train.py`

Chứa logic train chính.

Di chuyển và refactor phần logic trong function sau vào đây:

```python
run_seed
```

File này nên có function chính:

```python
def run_training(config: dict) -> dict:
    ...
```

hoặc:

```python
def run_seed(global_seed: int, config: dict, output_root: str) -> dict:
    ...
```

Vai trò:

* Load config.
* Chuẩn bị dataset root.
* Tạo DataLoader.
* Tạo model.
* Tạo optimizer/scheduler/loss.
* Train qua nhiều epoch.
* Lưu checkpoint tốt nhất.
* Evaluate trên test set.
* Lưu history, scores, run_info.
* Gọi visualization.
* Trả về result dict.

Không parse CLI arguments trong file này.

---

### 5.11. `src/sensorcls/utils/seed.py`

Chứa các function liên quan đến reproducibility.

Di chuyển các function sau vào đây:

```python
set_global_determinism
seed_worker
```

Vai trò:

* Set seed cho Python, NumPy, PyTorch.
* Set deterministic mode.
* Seed cho DataLoader workers.

---

### 5.12. `src/sensorcls/utils/amp.py`

Chứa logic AMP.

Di chuyển các function sau vào đây:

```python
make_autocast
make_scaler
```

Vai trò:

* Hỗ trợ mixed precision training.
* Không chứa train loop.

---

### 5.13. `src/sensorcls/visualization/curves.py`

Chứa function vẽ learning curve.

Di chuyển function sau vào đây:

```python
plot_performance
```

Vai trò:

* Vẽ train accuracy, validation accuracy.
* Vẽ train loss, validation loss.
* Lưu figure `.png`.

---

### 5.14. `src/sensorcls/visualization/confusion_matrix.py`

Chứa function vẽ confusion matrix.

Di chuyển function sau vào đây:

```python
plot_confusion_matrix
```

Vai trò:

* Vẽ confusion matrix.
* Lưu figure `.png`.
* Không tính metric ở đây.

---

### 5.15. `src/sensorcls/utils/archive.py`

Chứa logic nén kết quả.

Di chuyển logic zip trong `main()` cũ vào đây.

Nên có function:

```python
def zip_run_outputs(output_root: str, dataset_name: str, model_name: str, seeds: list[int]) -> str:
    ...
```

Vai trò:

* Zip stats file.
* Zip artifact từng seed.
* Trả về zip path.

---

### 5.16. `src/sensorcls/utils/io.py`

Chứa helper lưu JSON/text.

Nên có các function:

```python
def save_json(payload: dict, path: str) -> None:
    ...

def save_text(lines: list[str], path: str) -> None:
    ...
```

Vai trò:

* Tránh lặp code `open(... json.dump ...)`.
* Đảm bảo encoding `utf-8`.

---

### 5.17. `src/sensorcls/config/loader.py`

Chứa logic đọc YAML config.

Nên có function:

```python
def load_config(path: str) -> dict:
    ...
```

Yêu cầu:

* Đọc file YAML.
* Validate các field quan trọng.
* Tính `step_size = int(window * (1 - overlap))` nếu chưa có.
* Không để CLI tự xử lý toàn bộ config.

---

### 5.18. `src/sensorcls/cli/train.py`

Đây là CLI chính của project.

Nên hỗ trợ lệnh:

```bash
python -m sensorcls.cli.train --config configs/kaggle_1dcnn_s12.yaml
```

Hoặc nếu dùng entry point trong `pyproject.toml`, hỗ trợ:

```bash
sensorcls-train --config configs/kaggle_1dcnn_s12.yaml
```

Vai trò:

* Parse argument `--config`.
* Load config.
* Gọi `run_training(config)`.
* In đường dẫn output cuối cùng.

Không đặt train loop trực tiếp trong CLI.

---

### 5.19. `scripts/train.py`

File wrapper đơn giản để người dùng chạy:

```bash
python scripts/train.py --config configs/kaggle_1dcnn_s12.yaml
```

File này chỉ nên gọi lại CLI hoặc engine.

Không chứa logic training chính.

---

## 6. Yêu cầu CLI trên Kaggle

Sau khi clone repo trên Kaggle, user cần chạy được một lệnh train chính:

```bash
python scripts/train.py --config configs/kaggle_1dcnn_s12.yaml
```

Lệnh này phải:

1. Đọc config YAML.
2. Detect dataset trong `/kaggle/input/...`.
3. Nếu có sẵn `.npy`, dùng trực tiếp.
4. Nếu chưa có `.npy` nhưng có CSV, convert CSV sang `.npy`.
5. Tạo train/val/test DataLoader.
6. Train model 1DCNN.
7. Lưu checkpoint tốt nhất.
8. Evaluate trên test set.
9. Lưu:

   * history JSON
   * run info JSON
   * scores JSON
   * learning curve PNG
   * confusion matrix PNG
   * stats TXT
   * zip artifact cuối cùng
10. In ra đường dẫn output.

Output mặc định trên Kaggle:

```text
/kaggle/working/History
```

---

## 7. Yêu cầu `pyproject.toml`

Tạo `pyproject.toml` để project có thể cài dạng editable nếu cần.

Nội dung tối thiểu:

```toml
[project]
name = "sensor-time-series-classification"
version = "0.1.0"
description = "1D CNN training pipeline for sensor time-series classification"
requires-python = ">=3.10"
dependencies = [
    "numpy",
    "pandas",
    "matplotlib",
    "scikit-learn",
    "torch",
    "tqdm",
    "pyyaml"
]

[project.scripts]
sensorcls-train = "sensorcls.cli.train:main"

[tool.setuptools.packages.find]
where = ["src"]
```

---

## 8. Yêu cầu `requirements.txt`

Tạo file `requirements.txt`:

```text
numpy
pandas
matplotlib
scikit-learn
torch
tqdm
pyyaml
```

Không dùng `seaborn` nếu không thật sự cần. Nếu vẫn giữ heatmap bằng seaborn thì phải thêm:

```text
seaborn
```

---

## 9. Yêu cầu `README.md`

README phải có ít nhất các phần:

```markdown
# Sensor Time Series Classification

## Problem

## Project Structure

## Dataset

## Kaggle Usage

## Train

## Outputs

## Config

## Reproducibility
```

Trong phần Kaggle Usage, ghi rõ:

```bash
git clone <REPO_URL>
cd sensor_time_series_classification
python scripts/train.py --config configs/kaggle_1dcnn_s12.yaml
```

Nếu cần cài package:

```bash
pip install -r requirements.txt
```

---

## 10. Yêu cầu `.gitignore`

Tạo `.gitignore` để không push output nặng lên GitHub.

```gitignore
__pycache__/
*.pyc
.ipynb_checkpoints/

outputs/*
!outputs/.gitkeep

reports/*
!reports/.gitkeep

data/*
!data/.gitkeep

*.pt
*.pth
*.zip
*.npy
*.npz

.DS_Store
.env
```

Nếu dataset nhỏ và người dùng muốn push data mẫu, cần hỏi trước. Mặc định không push dataset và checkpoint.

---

## 11. Yêu cầu test tối thiểu

Tạo test cơ bản trong folder `tests/`.

### 11.1. `tests/test_model_forward.py`

Test model nhận input đúng shape và output đúng shape.

Ví dụ:

```python
import torch
from sensorcls.models.cnn1d import Conv1DClassifier

def test_cnn1d_forward_shape():
    model = Conv1DClassifier(
        in_channels=2,
        num_classes=4,
        window=1024,
        fc_dim=128,
        dropout=0.4,
    )
    x = torch.randn(8, 2, 1024)
    y = model(x)
    assert y.shape == (8, 4)
```

### 11.2. `tests/test_metrics.py`

Test metric output có key cần thiết.

```python
import numpy as np
from sensorcls.engine.metrics import build_scores

def test_build_scores_keys():
    y_true = np.array([0, 1, 1, 0])
    y_pred = np.array([0, 1, 0, 0])
    scores = build_scores(y_true, y_pred)
    assert "accuracy" in scores
    assert "precision" in scores
    assert "recall" in scores
    assert "f1score" in scores
    assert "confusion" in scores
```

### 11.3. `tests/test_dataset.py`

Test các function nhỏ như `starts_for_length`.

```python
from sensorcls.data.preprocessing import starts_for_length

def test_starts_for_length():
    starts = starts_for_length(length=2048, window=1024, step=256)
    assert starts[0] == 0
    assert starts[-1] <= 1024
```

---

## 12. Quy tắc coding style

AI Agent cần tuân thủ:

* Không viết toàn bộ logic trong một file lớn.
* Mỗi file chỉ nên có một nhóm trách nhiệm rõ ràng.
* Function nên có type hints.
* Ưu tiên tên function rõ nghĩa.
* Không dùng biến global cho config training nếu có thể đọc từ YAML.
* Không để code chạy training khi import module.
* Chỉ chạy training trong:

  * `if __name__ == "__main__":`
  * hoặc CLI entry point.
* Không để notebook là entry point chính.
* Không đổi metric hoặc split logic nếu chưa được yêu cầu.
* Không xóa output artifact hiện có.

---

## 13. Acceptance criteria

Refactor được xem là hoàn thành nếu thỏa mãn các điều kiện sau:

### 13.1. Cấu trúc project đúng

Các folder chính phải tồn tại:

```text
configs/
src/sensorcls/
scripts/
outputs/
reports/
tests/
```

### 13.2. CLI chạy được

Lệnh sau phải chạy được:

```bash
python scripts/train.py --config configs/kaggle_1dcnn_s12.yaml
```

### 13.3. Import module không gây train

Lệnh sau không được tự động train:

```python
import sensorcls
```

### 13.4. Model forward đúng shape

Với input:

```python
x.shape == (batch_size, 2, 1024)
```

Output phải là:

```python
logits.shape == (batch_size, num_classes)
```

### 13.5. Output được lưu đúng

Sau khi train, phải sinh ra các file tương đương logic cũ:

```text
history_*.json
run_info_*.json
scores_*.json
learning_curve_*.png
confusion_matrix_*_test.png
five_seed_stats_*.txt
summary_*.json
*.zip
```

### 13.6. Không mất logic Kaggle

Vẫn phải hỗ trợ:

* Tìm dataset trong `/kaggle/input/...`
* Dùng `.npy` nếu có sẵn
* Convert CSV sang `.npy` nếu cần
* Lưu kết quả vào `/kaggle/working/History`

---

## 14. Chiến lược refactor đề xuất cho AI Agent

Không refactor tất cả cùng lúc theo kiểu viết lại từ đầu.

Hãy làm theo thứ tự:

1. Tạo folder structure.
2. Tạo `configs/kaggle_1dcnn_s12.yaml`.
3. Tách data discovery/conversion.
4. Tách dataset/windowing/dataloader.
5. Tách model 1DCNN.
6. Tách metrics/evaluate.
7. Tách train loop.
8. Tách visualization.
9. Tạo CLI.
10. Tạo tests.
11. Chạy test model forward.
12. Chạy thử CLI với config.
13. So sánh output mới với output cũ.

---

## 15. Những điều không được làm

AI Agent không được:

* Xóa logic convert CSV sang `.npy`.
* Xóa logic split theo run id.
* Đổi architecture `Conv1DClassifier` nếu không được yêu cầu.
* Đổi optimizer/scheduler/loss nếu không được yêu cầu.
* Đổi output format nếu không được yêu cầu.
* Đưa dataset/checkpoint/output zip vào Git.
* Biến project thành notebook-only project.
* Viết lại code theo framework quá phức tạp như Lightning nếu không được yêu cầu.
* Tự ý thêm dependency nặng.
* Tự ý đổi đường dẫn Kaggle mặc định mà không cập nhật config.

---

## 16. Kết quả mong muốn cuối cùng

Sau refactor, user có thể làm như sau trên Kaggle:

```bash
git clone <REPO_URL>
cd sensor_signal_analyzer
python scripts/train.py --config configs/kaggle_1dcnn_s12.yaml
```

Sau khi chạy xong, kết quả nằm ở:

```text
/kaggle/working/History
```

Trong đó có:

```text
seed42/
summary_*.json
five_seed_stats_*.txt
*.zip
```

Project phải đủ sạch để sau này thêm model mới bằng cách:

1. Thêm file model vào `src/sensorcls/models/`.
2. Thêm config mới vào `configs/`.
3. Chạy lại CLI.

Không cần sửa trực tiếp train loop chính nếu chỉ thêm model mới.
