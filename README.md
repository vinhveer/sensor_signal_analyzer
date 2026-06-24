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
├── pyproject.toml
├── .gitignore
│
├── configs/
│   └── kaggle_1dcnn_s12.yaml        Cấu hình training chính
│
├── scripts/
│   └── train.py                     Entry point chạy training từ dòng lệnh
│
├── src/
│   └── sensorcls/                   Package Python chính
│       ├── cli/
│       │   └── train.py             CLI parse --config và gọi engine
│       │
│       ├── config/
│       │   └── loader.py            Đọc và validate YAML config
│       │
│       ├── data/
│       │   ├── constants.py         Hằng số fallback mặc định
│       │   ├── discovery.py         Tìm dataset root (.npy hoặc CSV)
│       │   ├── conversion.py        Convert CSV sang .npy
│       │   ├── preprocessing.py     Tính sliding window, normalize từng window
│       │   ├── dataset.py           WindowDataset, DataConfig, split theo run id
│       │   └── dataloaders.py       Tạo train/val/test DataLoader
│       │
│       ├── engine/
│       │   ├── context.py           RunContext (dataclass), prepare_run_context
│       │   ├── checkpoint.py        Lưu checkpoint tốt nhất
│       │   ├── evaluate.py          evaluate_model, evaluate_best_model
│       │   ├── metrics.py           build_scores (accuracy, precision, recall, f1)
│       │   ├── artifacts.py         build_run_info, save_run_artifacts
│       │   └── train.py             train_one_epoch, train_epochs, run_seed, 
│       │
│       ├── models/
│       │   └── cnn1d.py             Conv1dSame, Conv1DClassifier
│       │
│       ├── utils/
│       │   ├── seed.py              set_global_determinism, seed_worker
│       │   ├── io.py                save_json, save_text
│       │   └── archive.py           zip_run_outputs
│       │
│       └── visualization/
│           ├── curves.py            plot_performance (learning curve)
│           └── confusion_matrix.py  plot_confusion_matrix
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
  --config configs/kaggle_1dcnn_s12.yaml \
  --kaggle-working-dataset-root /kaggle/working/dataset_1_10 \
  --kaggle-input-root <PathDataset>
```

---

## Sử dụng trên Kaggle

```bash
!git clone https://github.com/vinhveer/sensor_signal_analyzer
%cd sensor_signal_analyzer
!python scripts/train.py \
  --config configs/kaggle_1dcnn_s12.yaml \
  --kaggle-working-dataset-root /kaggle/working/dataset_1_10 \
  --kaggle-input-root <PathDataset>
```

Example

```bash
!git clone https://github.com/vinhveer/sensor_signal_analyzer
%cd sensor_signal_analyzer
!python scripts/train.py \
  --config configs/kaggle_1dcnn_s12.yaml \
  --kaggle-working-dataset-root /kaggle/working/dataset_1_10 \
  --kaggle-input-root /kaggle/input/datasets/thanhhieu03092004/test-dynamic-path/dataset_1_10
```

Kết quả mặc định được lưu tại `/kaggle/working/History`.

---

## Output sau khi train

Nằm trong `outputs.root` từ config (mặc định `/kaggle/working/History`):

```text
History/
├── seed42/
│   ├── best_<dataset>_<model>.pt
│   ├── history_<dataset>_<model>.json
│   ├── run_info_<dataset>_<model>.json
│   ├── scores_<dataset>_<model>.json
│   ├── learning_curve_<dataset>_<model>.png
│   └── confusion_matrix_<dataset>_<model>_test.png
├── five_seed_stats_<dataset>_<model>.txt
├── summary_<dataset>_<model>_seeds1_5.json
└── <dataset>_<model>_seeds1_5.zip
```

---

## Cấu hình (Config)

Tất cả tham số nằm trong `configs/kaggle_1dcnn_s12.yaml`:


| Nhóm        | Tham số chính                                          |
| ----------- | ------------------------------------------------------ |
| `data`      | đường dẫn dataset, tên cột acceleration, separator CSV |
| `windowing` | window=1024, overlap=0.75, step tự tính                |
| `split`     | train/val/test run id, số file mỗi class               |
| `training`  | batch_size, epochs, learning_rate, weight_decay        |
| `model`     | fc_dim, dropout                                        |
| `outputs`   | root, save_zip                                         |


`step_size` được tự tính nếu không khai báo: `int(window * (1 - overlap))`.

Hai tham số sau trong config có thể được ghi đè từ CLI:


| Config key                          | CLI argument                    |
| ----------------------------------- | ------------------------------- |
| `data.kaggle_working_dataset_root`  | `--kaggle-working-dataset-root` |
| `data.kaggle_input_root_candidates` | `--kaggle-input-root`           |


---

## Tái tạo kết quả (Reproducibility)

`set_global_determinism(seed)` seed Python, NumPy, PyTorch và bật `use_deterministic_algorithms`.
DataLoader worker được seed qua `seed_worker`. Seed mặc định là `42`.

---

## Chạy test

```bash
python -m pytest
```

