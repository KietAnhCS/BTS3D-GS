# Công thức toán học trong `pipeline/run.py`

File này **không chứa công thức toán học nào**. Đây là module điều phối (orchestration) cấp cao nhất của pipeline: ghép tuần tự các bước `setup → load_data → smoke_test → run_all → analytics → finish`, gọi sang `pipeline/trainer.py` (train), `pipeline/submission.py` (render + chấm điểm), `pipeline/report.py` (bảng/biểu đồ), `pipeline/deliver.py` (lưu/tải file). Toàn bộ phép toán thực sự (loss, công thức Score, densify, v.v.) nằm ở các module mà file này gọi tới, không nằm trong chính `run.py`.

Các phép tính số học duy nhất trong file chỉ là logic điều khiển luồng (ví dụ chọn số vòng lặp smoke test, số view đánh giá), không phải công thức khoa học:

Trích nguyên văn (`pipeline/run.py`, dòng 51–54):

```python
    trial = dataclasses.replace(cfg,
                                output_root=os.path.join(cfg.output_root, "_smoke"),
                                score_every=max(50, (iterations or cfg.smoke_iterations) // 2),
                                eval_views=3 if cfg.eval_views is None else min(cfg.eval_views, 3))
```

- `smoke_test`: `score_every = max(50, (iterations or cfg.smoke_iterations) // 2)` — chia đôi số vòng lặp smoke để lấy mốc chấm điểm giữa chừng, chặn dưới ở 50.
- `smoke_test`: `eval_views = 3 if cfg.eval_views is None else min(cfg.eval_views, 3)` — giới hạn số view đánh giá tối đa 3 khi chạy thử.

Đây đều là các hằng số/ngưỡng vận hành (operational heuristics) chứ không phải công thức toán có ý nghĩa khoa học, nên không đưa vào bảng công thức.

## Tóm tắt chức năng từng hàm

| Hàm | Chức năng |
|---|---|
| `setup` | Cài phụ thuộc, gắn Google Drive, kiểm tra GPU, tạo thư mục output. |
| `load_data` | Tải/giải nén dataset, tìm scene, kiểm tra đủ ảnh, in hồ sơ dữ liệu. |
| `smoke_test` | Chạy thử vài trăm vòng trên 1 scene để xác nhận pipeline chạy được trước khi train thật. |
| `run_all` | Vòng lặp train + render từng scene, autosave checkpoint lên Drive, ghi kết quả sau mỗi scene. |
| `_dump_results` | Ghi `results.json` (config + kết quả + submission). |
| `analytics` | Gọi `report.py` để tạo `history.csv`, `leaderboard.csv` và hai ảnh biểu đồ. |
| `finish` | Đóng gói `submission.zip`, kiểm tra tính hợp lệ, tải file về máy. |

---

## Bảng tương ứng cú pháp ↔ công thức

File này không có công thức toán học. Bảng dưới liệt kê các ngưỡng/heuristic vận hành duy nhất xuất hiện, để đầy đủ theo quy ước tài liệu:

| Code | Công thức/ý nghĩa |
|---|---|
| `max(50, (iterations or cfg.smoke_iterations) // 2)` | $\text{score\_every} = \max(50,\ \lfloor n_{iter}/2 \rfloor)$ (heuristic vận hành, không phải công thức khoa học) |
| `3 if cfg.eval_views is None else min(cfg.eval_views, 3)` | $\text{eval\_views} = \min(\texttt{cfg.eval\_views},\ 3)$ khi chạy smoke test |
