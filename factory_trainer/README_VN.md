# Trainer sử dụng DatasetFactory/Dataset hiện có của em

Gói này chỉ thêm `factory_trainer/`. Không chép đè DatasetFactory.py, DatasetSurvival.py, model.py hoặc block/*.py. Không sử dụng load_fold hay FoldDataset của gói vast_training cũ.

## Nguyên tắc

Dữ liệu được tạo trực tiếp bằng đúng các class trong project:

```python
from dataset.DatasetFactory import SurvivalDatasetFactory
factory = SurvivalDatasetFactory(**factory_kwargs)
train_dataset, scaler = factory._get_split_from_df(
    split_path=split_csv, split_key='train', fold=fold, scaler=None, valid_cols=None)
val_dataset, _ = factory._get_split_from_df(
    split_path=split_csv, split_key='val', fold=fold, scaler=scaler, valid_cols=None)
```

Giữ nguyên quy tắc chia bin, scaler, clinical intersection, cách xử lý ID omics lặp và slide map của Factory. Trainer không bổ sung phép dedup, không sửa giá trị gene, không đổi thời gian sang time_scale. Censor=0 có sự kiện, censor=1 censored. Bản này tái sử dụng pipeline em đã kiểm tra; không tuyên bố nó là protocol khác.

`WithIndex` chỉ gắn số thứ tự dataset để truy vết case_id. `collate_original` giữ tất cả giá trị dataset trả về, ghép groups và thêm zero padding tới N lớn nhất trong batch; giữ mask padding sẵn có. Hỗ trợ `pathways` và `groups`, không hỗ trợ gene-only vì model này cần WSI.

## Model/loss được dùng

Trainer dùng **model và loss đã debug trong project/notebook của em**, không dùng class gốc trong attachment nếu nó vẫn là bản chưa sửa. Các attachment model.py cũ còn forward dùng biến chưa định nghĩa và chưa có predict/return_aux; trainer không thể làm bản đó chạy đúng mà không sửa model. Gói này chủ động không sửa các file đó.

Giao diện giống notebook em đã chạy:

```python
output = model(patch, omics, bins, time, censor, mask, return_aux=True)
losses = criterion(output, bins, time, censor)  # dictionary, có total
club_fit = criterion.club_learning_loss(output) # features được detach bên trong
params = criterion.club_parameters()
pred = model.predict(patch, omics) # không có bins/time/censor
```

`predict` trả dictionary có `shapes`, `scales`, `weights`, mỗi tensor [B,M]. Đây phải là tham số thực k, lambda >0 sau contraction+softplus; không phải coefficient [B,M,D]. Trainer loại padding từng bệnh nhân trước gọi predict, nên predict không cần tham số mask. Gọi prediction trong eval mode và no_grad.

PoE cần có `model.PoE.phi`, và PoE dùng `softmax(phi)` như code em gửi. Thứ tự tên mặc định histology/pathway theo thứ tự stack trong model. `model.PoE` được tối ưu bởi model optimizer, không có optimizer khác cho PoE.

Nếu class hoặc đường import khác, chỉnh `model_class`, `criterion_class` trong config.json (cú pháp module:Class). Trainer sẽ báo rõ đường dẫn module thực sự được import, giúp tránh notebook vô tình chạy code ở project khác.

## Cài và chạy

Giải nén folder factory_trainer vào ROOT project, cạnh model.py, block/, dataset/, datasets_csv/, 5foldcv/, WSIs/.

Cài thư viện phụ trong môi trường PyTorch đang hoạt động của em:

```bash
cd /home/hai/Projects/MultimodalWeibull
python -m pip install -r factory_trainer/requirements.txt
python -m factory_trainer.test_numerics
python -m factory_trainer.test_collate
```

Mở config.json kiểm tra đường dẫn WSI. `wsi_dir` phải là thư mục cha trực tiếp của `pt_files`, giống notebook gốc. Ví dụ nếu dùng data download layout mới thì đặt `WSIs/coadread-005`; nếu giữ nesting ban đầu thì dùng giá trị có sẵn trong config. Không tự đổi hoặc di chuyển WSI.

Kiểm tra kết nối forward/backward và tạo đồ thị validation bằng:

```bash
python -m factory_trainer.run --config factory_trainer/config.json \
  --root /home/hai/Projects/MultimodalWeibull \
  --cohorts coadread --folds 0 \
  --output runs/factory_preflight --preflight
```

Preflight dùng một batch train, backward để kiểm tra gradient, rồi đánh giá toàn validation để kiểm tra predict và vẽ đồ thị. Không optimizer.step, không train hoàn chỉnh. Có thể mất thời gian ở validation. Nó cũng kiểm tra lớp model/loss đúng phiên bản. Nếu còn import model cũ, sửa đúng import/project đang dùng thay vì chép đè dữ liệu.

Chạy fold 0 trước:

```bash
python -m factory_trainer.run --config factory_trainer/config.json \
  --root /home/hai/Projects/MultimodalWeibull \
  --cohorts coadread --folds 0 --output runs/factory_v1
```

Chạy đủ mọi cohort/fold trong config:

```bash
python -m factory_trainer.run --config factory_trainer/config.json \
  --root /workspace/MultimodalWeibull --output /workspace/runs/factory_v1
```

Resume cùng cấu hình, thêm `--resume`. Có thể tăng epochs. Checkpoint chứa cả model, criterion/CLUB, hai optimizer, scheduler, preprocessing (scaler/nhóm gene), RNG và lịch sử. Resume ở epoch đã lưu; mất kết nối giữa epoch sẽ chạy lại epoch đó. Khi resume phải giữ source, IDs và cấu hình phù hợp. Không hash nội dung từng file WSI lớn; không đổi dataset giữa lượt chạy.

## Dùng trực tiếp với m/criterion đang có trong notebook

```python
from factory_trainer.trainer import Trainer

# Factory là instance em đã tạo và kiểm tra.
train_dataset, scaler = Factory._get_split_from_df(
    split_path=fold_path, split_key='train', fold=0,
    scaler=None, valid_cols=None)
val_dataset, _ = Factory._get_split_from_df(
    split_path=fold_path, split_key='val', fold=0,
    scaler=scaler, valid_cols=None)

trainer = Trainer(
    model=m,
    criterion=criterion,
    train_dataset=train_dataset,
    val_dataset=val_dataset,
    outdir='runs/notebook/fold_0',
    config={
        'epochs': 100, 'patience': 15,
        'batch_size': 4, 'eval_batch_size': 4, 'workers': 0,
        'lr': 1e-4, 'club_lr': 1e-4, 'seed': 42,
        'plot_every': 5, 'plot_cases': 3,
        'train_eval_every': 1,
        'val_auxiliary': True,
        'poe_names': ['histology', 'pathway'],
    },
    preprocessing={'scaler': scaler, 'omic_names': Factory.omic_names},
)
trainer.fit()
```

m và criterion giữ nguyên kiến trúc/weights lúc em truyền vào. Trainer tạo optimizer mới. Nếu muốn bắt đầu từ đầu phải tạo m/criterion mới; chạy các fold phải tạo instance mới mỗi fold. CLI đã làm điều này.

## Theo dõi loss và metric

TensorBoard:

```bash
tensorboard --logdir /workspace/runs/factory_v1 --host 127.0.0.1 --port 6006
```

Trên máy cá nhân dùng SSH command của Vast và thêm `-L 16006:127.0.0.1:6006`, mở http://127.0.0.1:16006. Cách backup Drive đã hướng dẫn vẫn dùng được: chỉ đổi đường dẫn output thành /workspace/runs/factory_v1.

Các tag:

| Tag | Ý nghĩa |
|---|---|
| batch/* | Loss từng thành phần, CLUB fit, gradient norm từng train step |
| train_loss/* | Trung bình objective train theo số bệnh nhân mỗi epoch |
| train_eval/nll, c_index | Đánh giá train trong eval mode, dự đoán không dùng nhãn |
| val/nll, c_index | Đánh giá validation trong eval mode, dự đoán không dùng nhãn |
| val/aux_label_conditioned/* | Các loss phụ/total tính theo nhánh forward sử dụng nhãn, chỉ để chẩn đoán |
| poe/phi_histology, phi_pathway | Tham số PoE học trực tiếp |
| poe/alpha_histology, alpha_pathway | Trọng số softmax(phi) thực sự dùng trong PoE |
| optimizer/lr | Learning rate |
| weibull/val/case_00... | Đồ thị phân phối từng bệnh nhân validation qua epoch |

Validation predictive NLL dùng likelihood Weibull mixture cùng rate convention và ngưỡng clamp như loss em đã viết. Đánh giá không dùng thời gian/censor để chọn proxy. Chọn best checkpoint và early-stop theo **val/nll**, không theo total label-conditioned. Test chạy một lần trên best model nếu Factory tạo được test.

Train_eval vẫn gọi chính train_dataset: nếu dataset train sample=True thì nó vẫn lấy số patch theo quy tắc của em. Không đổi sample=False. Đánh giá có RNG riêng để không làm đổi chuỗi sampling của training, nhưng kết quả vẫn là đánh giá trên patch sample của train. Train objective trong train mode (có dropout, label-conditioned) không đồng nhất với train_eval NLL (eval mode, label-free).

C-index là Harrell, tính toàn tập theo negative restricted mean survival (RMST). Horizon cố định theo quantile 90% thời gian train, hoặc đặt risk_horizon. Risk cao hơn là rủi ro cao hơn. Không lấy trung bình C-index từng batch; trường hợp không có comparable pairs ghi null. Bản này không tính Uno/IBS, không dùng MAE trên censored cases.

Val auxiliary được tính theo từng batch; CLUB cần nhiều bệnh nhân để có negative pairs. eval_batch_size=1 làm MI chẩn đoán bằng 0. Nếu hết VRAM có thể giảm batch size, đổi ý nghĩa chẩn đoán CLUB tương ứng. batch_size train >=2, singleton batch cuối vẫn giữ và có cảnh báo. Val aux làm tăng thời gian evaluate vì cần thêm forward. Có thể tắt `val_auxiliary` trong config, còn predictive NLL/C-index vẫn có.

## Đồ thị Weibull và PoE

Mixture phụ thuộc bệnh nhân nên không có một đường Weibull duy nhất đại diện cho toàn dataset. Trainer theo dõi `plot_cases` bệnh nhân đầu validation (thứ tự Factory giữ ổn định), epoch 1 và mỗi plot_every epoch, cộng best model.

Mỗi hình gồm:

1. PDF f_m(t) của từng thành phần và mixture sum(pi_m*f_m).
2. Đóng góp có trọng số pi_m*f_m(t), tổng bằng mixture.
3. Survival S_m(t) và S_mix(t).
4. Trọng số mixture pi_m.

Dùng đúng f(t)=lambda*k*t^(k-1)*exp(-lambda*t^k); lambda là rate, không phải scale thời gian theo quy ước thư viện khác. Không chuẩn hóa lại density chỉ để đường nằm vừa khung. Grid t>0 để tránh điểm singular khi k<1; đường chỉ hiển thị tới plot_horizon, không nhất thiết bao hết đuôi. Mặc định horizon=train quantile90%; có thể đặt plot_horizon theo đơn vị gốc của dataset. CSV lưu k, lambda, pi để kiểm tra.

PoE alpha khác pi mixture: alpha cân bằng modality, pi cân bằng thành phần Weibull. PoE precision thực còn phụ thuộc variance (alpha/sigma²), nên alpha cao không tự chứng minh modality đó quan trọng hơn cho mọi bệnh nhân.

## File kết quả

Mỗi cohort/fold:

- history.csv, epochs.jsonl: loss train/val, metric, phi, alpha theo epoch.
- figures/history.png, figures/poe.png: đồ thị lịch sử cập nhật mỗi epoch.
- figures/epoch_XXXX/val_case_XX.png và *_plot_parameters.csv: đường Weibull và tham số từng bệnh nhân.
- tensorboard/: scalar và hình realtime.
- predictions/*_latest.csv: case_id, index, t, c, risk, NLL, k/lambda/pi từng thành phần.
- last.pt / best.pt: checkpoint, không dùng checkpoint cũ của trainer khác để resume.
- val_best_metrics.json / test_best_metrics.json: đánh giá best model.
- train/val/test_metadata.csv: metadata chính Factory trả.
- factory_membership.csv: ID có trong split có được Factory đưa vào dataset không, không tự phục hồi hoặc loại thêm ID.
- factory_stdout.log: các print của Factory, tránh tràn màn hình terminal.
- factory_preprocessing.pkl: scaler và cấu hình/nhóm gene lấy trực tiếp từ Factory.
- source_snapshot/: bản sao source model/loss/dataset để truy vết; imports.json chỉ đường dẫn import thật.

Thư mục output có training.log, config.json, environment.json, pip_freeze.txt, fold_metrics.csv và summary.csv (mean/std/count theo cohort và split). Các fold được chạy độc lập. Nếu không có test thì không báo cáo test. Không tự gộp C-index khác cohort.

## Kiểm chứng

Đã compile mọi module, pass 3 numerical tests (exponential NLL, sum mixture và survival đơn điệu, risk/C-index), render và xem đồ thị bằng dữ liệu mô phỏng. Hai test collate đã viết nhưng bị skip trong môi trường thực hiện vì chưa có PyTorch. Chưa chạy trainer forward/backward/GPU do thiếu PyTorch và dữ liệu thật. Dùng --preflight trong môi trường của em để kiểm tra tích hợp trước chạy dài. Gói không kèm bất kỳ bản thay thế nào cho DatasetFactory/Dataset/model/block của em.
