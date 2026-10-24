# Tải Google Drive → Vast.ai → training

Gói bổ sung cho `vast_training.zip` đã gửi. Không thay model hoặc train.py. Script nhận link FILE ZIP, không nhận link FOLDER Drive. Em chưa gửi link thật, nên điền các URL trong drive_manifest.json trước khi chạy.

## Chuẩn bị trên Drive

Cần 5 ZIP feature: blca-003.zip, brca-004.zip, coadread-005.zip, hnsc-001.zip, stad-002.zip. Dùng feature `.pt` đã trích xuất; không cần tải raw ảnh `.svs` cho training. Nếu ZIP có cả raw ảnh, script chỉ lấy `.pt`.

Nếu CSV/splits chưa có trên Vast, upload thêm datasets_csv.zip và 5foldcv.zip. Trong ZIP giữ cấu trúc tương ứng thư mục của em. Trên Linux có thể tạo bằng `zip -r datasets_csv.zip datasets_csv` và `zip -r 5foldcv.zip 5foldcv`; hoặc dùng phần mềm nén thông thường.

Link phải tải được không cần đăng nhập cho script gdown này; hỗ trợ Google Drive URL chia sẻ file hoặc file ID. Nếu link private, dùng luồng tải xác thực của em để lấy ZIP về cache trước; script này không triển khai OAuth. Drive quota có thể chặn dù quyền đọc đúng, script không vượt quota.

## Cấu hình

Mở drive_manifest.json, thay mọi PASTE_* bằng link thật. Ví dụ:

```json
{
  "name": "coadread-005.zip",
  "url": "https://drive.google.com/file/d/FILE_ID/view",
  "kind": "wsi",
  "cohort": "coadread",
  "destination": "WSIs/coadread-005",
  "sha256": ""
}
```

sha256 tùy chọn, dùng `sha256sum coadread-005.zip` trên bản gốc để điền checksum nếu có. Không điền checksum giả.

Nếu datasets_csv/5foldcv đã có trên máy ảo, thêm `"enabled": false` cho hai mục đó. Script vẫn kiểm tra các CSV cần cho train, không xóa hoặc ghi đè file khác nội dung.

kind=tree lấy toàn bộ thư mục ZIP. Tự loại wrapper trước datasets_csv hoặc 5foldcv nếu ZIP chứa `Project/datasets_csv/...`. Nếu ZIP không có tên thư mục anchor và có wrapper khác, thêm `"source_subdir": "ten_wrapper"` để bỏ wrapper.

kind=wsi lấy mọi `.pt` và gom basename vào `destination/pt_files`. Nếu hai member trùng basename, dừng; không chọn ngẫu nhiên một file. Luồng training trước dùng recursive `WSIs/**/pt_files/*.pt`, nên layout này tương thích. Nếu dùng DatasetFactory cũ, wsi_dir phải đổi về `WSIs/coadread-005` (thư mục cha của pt_files), không dùng đường dẫn nesting cũ nữa.

## Chạy trên Vast

Giải nén drive_setup cạnh vast_training trong project. Python 3.10+:

```bash
cd /workspace/MultimodalWeibull/drive_setup
python -m pip install gdown==5.2.0
python test_prepare.py
python prepare_data.py --root /workspace/MultimodalWeibull --manifest drive_manifest.json
```

Download progress do gdown hiển thị; log tổng nằm ở PROJECT_ROOT/download.log. Có retry 3 lần, tiếp tục tải partial, kiểm tra ZIP, giải nén từng file qua file tạm và CRC. Khi chạy lại dùng ZIP cache và marker nếu file đã cài còn đúng kích thước. Marker không phát hiện sửa nội dung mà giữ nguyên kích thước; SHA256 tùy chọn xác minh archive nguồn, không phải hash lại toàn bộ file đã cài mỗi lần.

Nếu chỉ thử một cohort:

```bash
python prepare_data.py --root /workspace/MultimodalWeibull \
  --manifest drive_manifest.json --cohorts coadread
```

Chỉ tải metadata/splits và feature của cohort chọn. Nếu metadata chưa có vẫn cần link hai ZIP đầu. Có thể dùng `--folds 0` để kiểm tra một fold; không làm giảm dung lượng tải ZIP feature.

Khi download ngắt, chạy lại cùng lệnh. Không đổi URL của cùng archive name mà giữ cache cũ: dùng name/cache mới khi dataset đổi phiên bản. Cache mặc định PROJECT_ROOT/download_cache. Muốn dùng ZIP đã tải bằng công cụ khác, đặt tên tương ứng ở cache; script sẽ không tải lại ZIP hợp lệ.

```bash
python prepare_data.py --root /workspace/MultimodalWeibull --verify-only
```

verify-only kiểm tra layout, số feature, CSV signatures và các split. Nó chưa đọc tensor/đối chiếu từng bệnh nhân; training data audit sẽ làm bước đó. download_report.json ghi số file .pt và tổng bytes mỗi cohort.

Sau đó chạy bộ training đã gửi:

```bash
cd /workspace/MultimodalWeibull/vast_training
python smoke_test.py
python -u train.py --root /workspace/MultimodalWeibull \
  --output /workspace/runs/weibull_dss_v1 \
  --cohorts blca brca coadread hnsc stad --folds 0 1 2 3 4 \
  --epochs 100 --batch-size 4 --workers 2 --device cuda
```

## Dung lượng

Khi giữ cache, cần chỗ cho tổng ZIP + tổng .pt giải nén + CSV + checkpoints/logs + môi trường. Float32 feature thường không nén nhỏ như ảnh; không giả định file ZIP rất nhỏ. Script kiểm tra dung lượng bổ sung trước giải nén và giữ dự phòng 2 GiB, nhưng download ZIP vẫn có thể hết đĩa trước bước giải nén. Tính dung lượng từ dữ liệu thật trước khi chọn disk instance. Script không tự xóa cache để tránh mất khả năng resume.

## Ước tính chi phí sau benchmark

Thời gian một epoch đã gồm train và validation (epochs.jsonl). Chạy ít nhất 2–3 epochs/fold 0 cho mỗi cohort với đúng tham số sẽ dùng, ghi vào một output benchmark riêng:

```bash
python -u train.py --root /workspace/MultimodalWeibull \
  --output /workspace/runs/benchmark \
  --cohorts blca brca coadread hnsc stad --folds 0 --epochs 3 --device cuda
```

Rồi chạy estimator. Các rate dưới đây CHỈ LÀ VÍ DỤ tính toán; thay bằng rate của offer em thuê:

```bash
python /workspace/MultimodalWeibull/drive_setup/estimate_cost.py \
  --runs /workspace/runs/benchmark --epochs 100 --folds 5 \
  --gpu-hour 0.20 --setup-hours 2 --evaluation-hours 2
```

Thêm phí storage/network theo breakdown của offer:

- --disk-gb: dung lượng disk được cấp phát.
- --storage-gb-hour: USD/GB/hour.
- --download-gb / --in-tb-price: GB tải vào và USD/TB ingress.
- --upload-gb / --out-tb-price: GB xuất ra và USD/TB egress.

Estimator dùng median các epoch sau warmup của cohort đo được, nhân epochs×folds. Nếu mới đo một cohort thì chỉ dự tính cohort đó, không suy rộng cho mọi cohort. Không tự giảm chi phí vì early stopping; thêm setup/download và final evaluation qua tham số. Storage khi dừng máy phải tính riêng theo rate stopped-state. Formula:

C = GPU_rate * hours_running + storage_GB * storage_rate * hours_storage + ingress_TB * ingress_rate + egress_TB * egress_rate.

## Kiểm chứng

Đã pass 3 test giải nén local: gom feature và chạy lại, bỏ wrapper metadata, chặn đường dẫn vượt thư mục. Đã compile Python. Chưa tải file thật từ Drive vì chưa có link; chưa đo training trên Vast. Tệp `.pt` trong test chỉ là bytes giả để test giải nén, không xác nhận tensor PyTorch.
