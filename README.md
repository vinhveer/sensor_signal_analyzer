# Phân loại tín hiệu cảm biến theo thời gian (Sensor Time Series Classification)

## Bài toán

Phân loại tín hiệu cảm biến dạng chuỗi thời gian thành các nhãn lớp bằng mô hình 1D CNN.
Mỗi cửa sổ tín hiệu có `1024` time step và `2` kênh (acceleration-x từ hai cảm biến).

---

## Cấu trúc project

```text
sensor_signal_analyzer/
│
├── Agents.md                        Tài liệu mô tả yêu cầu refactor
├── README.md
├── requirements.txt
├── .gitignore
│
├── scripts/
│   └── train.py                     Entry point chạy training từ dòng lệnh
│
├── src/
│   ├── cli/
│   │   └── train.py                 CLI parse params và gọi engine
│   ├── config/
│   │   └── loader.py                Default config trong code và validate
│   ├── data/
│   │   ├── constants.py             Hằng số fallback mặc định
│   │   ├── discovery.py             Tìm dataset root (.npy hoặc CSV)
│   │   ├── conversion.py            Convert CSV sang .npy
│   │   ├── preprocessing.py         Tính sliding window, normalize từng window
│   │   ├── dataset.py               WindowDataset, DataConfig, split theo run id
│   │   └── dataloaders.py           Tạo train/val/test DataLoader
│   ├── engine/
│   │   ├── context.py               RunContext (dataclass), prepare_run_context
│   │   ├── checkpoint.py            Lưu checkpoint tốt nhất
│   │   ├── evaluate.py              evaluate_model, evaluate_best_model
│   │   ├── metrics.py               build_scores (accuracy, precision, recall, f1)
│   │   ├── artifacts.py             build_run_info, save_run_artifacts
│   │   └── train.py                 train_one_epoch, train_epochs, run_seed
│   ├── models/
│   │   ├── registry.py              Route model.name tới model package
│   │   └── cnn1d/
│   │       ├── model.py             Conv1dSame, Conv1DClassifier
│   │       ├── params.py            Conv1DParams, validate param riêng
│   │       ├── train.py             build_model, build_training_objects
│   │       └── inference.py         predict_logits
│   ├── utils/
│   │   ├── seed.py                  set_global_determinism, seed_worker
│   │   ├── io.py                    save_json, save_text
│   │   └── archive.py               zip_run_outputs
│   └── visualization/
│       ├── curves.py                plot_performance (learning curve)
│       └── confusion_matrix.py      plot_confusion_matrix
│
└── tests/
    ├── conftest.py
    ├── test_dataset.py              Test starts_for_length
    ├── test_metrics.py              Test build_scores
    └── test_model_forward.py        Test forward shape của Conv1DClassifier
```

---

## Dataset

Mỗi lớp là một folder con trong dataset root. Tín hiệu được lưu trong thư mục con `Processed` dưới dạng:

- `.npy` (ưu tiên dùng trực tiếp)
- `.csv` (tự động convert sang `.npy` nếu chưa có)

Tên file phải chứa run id dạng `R1`..`R6` để phân chia train/val/test:

```text
train: R1, R2, R3, R4
val:   R5
test:  R6
```

---

## Cài đặt

```bash
pip install -r requirements.txt
```

---

## Chạy training

```bash
python scripts/train.py \
  --dataset-root /kaggle/working/dataset_1_10 \
  --kaggle-input-root <PathDataset> \
  --epochs 100 \
  --batch-size 32 \
  --fc-dim 128 \
  --dropout 0.0
```

---

## Sử dụng trên Kaggle

```bash
!git clone https://github.com/vinhveer/sensor_signal_analyzer
%cd sensor_signal_analyzer
!python scripts/train.py \
  --dataset-root /kaggle/working/dataset_1_10 \
  --kaggle-input-root <PathDataset>
```

Example

```bash
!git clone https://github.com/vinhveer/sensor_signal_analyzer
%cd sensor_signal_analyzer
!python scripts/train.py \
  --dataset-root /kaggle/working/dataset_1_10 \
  --kaggle-input-root /kaggle/input/datasets/thanhhieu03092004/test-dynamic-path/dataset_1_10
```

Kết quả mặc định được lưu tại `/kaggle/working/History`.

---

## Output sau khi train

Nằm trong `--output-root` (mặc định `/kaggle/working/History`):

```text
History/
├── seed42/
│   ├── best_<dataset>_<model>.pt
│   ├── history_<dataset>_<model>.json
│   ├── run_info_<dataset>_<model>.json
│   ├── scores_<dataset>_<model>.json
│   ├── learning_curve_<dataset>_<model>.png
│   └── confusion_matrix_<dataset>_<model>_test.png
├── seed_stats_<dataset>_<model>_seeds42.txt
├── summary_<dataset>_<model>_seeds42.json
└── <dataset>_<model>_seeds42.zip
```

---

## Tham số CLI

Mặc định nằm trong code, chỉnh bằng CLI. Không cần YAML.


| Nhóm        | Tham số chính                                          |
| ----------- | ------------------------------------------------------ |
| `data`      | đường dẫn dataset, tên cột acceleration, separator CSV |
| `windowing` | window=1024, overlap=0.75, step tự tính                |
| `split`     | train/val/test run id, số file mỗi class               |
| `training`  | batch_size, epochs, learning_rate, weight_decay        |
| `model`     | name và tham số riêng của từng model                   |
| `outputs`   | root, save_zip                                         |


`step_size` được tự tính: `int(window * (1 - overlap))`.

Các tham số thường dùng có thể ghi đè từ CLI:


| Config key                          | CLI argument                    |
| ----------------------------------- | ------------------------------- |
| `data.kaggle_working_dataset_root`  | `--kaggle-working-dataset-root` |
| `data.kaggle_input_root_candidates` | `--kaggle-input-root`           |
| `model.name`                        | `--model-name`                  |
| `model.fc_dim`                      | `--fc-dim`                      |
| `model.dropout`                     | `--dropout` hoặc `--drop-out`   |
| `training.epochs`                   | `--epochs`                      |
| `training.batch_size`               | `--batch-size`                  |
| `training.learning_rate`            | `--learning-rate`               |
| `training.seeds`                    | `--seeds 42,43`                 |
| `windowing.window`                  | `--window`                      |
| `windowing.overlap`                 | `--overlap`                     |
| `outputs.root`                      | `--output-root`                 |

Ví dụ override model từ CLI:

```bash
python scripts/train.py \
  --model-name cnn1d \
  --fc-dim 64 \
  --dropout 0.2 \
  --epochs 50
```

Thêm model mới: tạo folder trong `src/models/`, khai báo CLI args riêng trong `cli.py`, rồi đăng ký trong `src/models/registry.py`.
CLI và training loop không cần biết class cụ thể.


---

## Tái tạo kết quả (Reproducibility)

`set_global_determinism(seed)` seed Python, NumPy, PyTorch và bật `use_deterministic_algorithms`.
DataLoader worker được seed qua `seed_worker`. Seed mặc định là `42`.

---

## Chạy test

```bash
python -m pytest
```

