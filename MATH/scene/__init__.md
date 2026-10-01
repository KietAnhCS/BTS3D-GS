# Công thức toán học trong `scene/__init__.py`

## Nhận định chung

`scene/__init__.py` định nghĩa lớp `Scene`, đóng vai trò **điều phối I/O**: tải scene (COLMAP/Blender), tạo/khôi phục danh sách camera theo từng `resolution_scale`, ghi/đọc point cloud `.ply`, và lưu `cameras.json`. File **không tự tính** các đại lượng hình học (ma trận camera, chuẩn hoá scene, v.v.) — các phép tính đó nằm trong `scene/dataset_readers.py` và `utils/camera_utils.py` (ngoài phạm vi tài liệu này). Đại lượng toán học duy nhất có ý nghĩa trực tiếp trong file là $E=$ `cameras_extent` (bán kính chuẩn hoá scene), được lưu lại từ kết quả tính sẵn ở nơi khác và dùng xuyên suốt `gaussian_model.py` (xem `gaussian_model.md`).

Nguồn đã đọc: `scene/__init__.py` (toàn bộ, 96 dòng).

---

## 1. Bán kính chuẩn hoá scene (`cameras_extent`)

Trích nguyên văn (`scene/__init__.py`, dòng 69):

```python
        self.cameras_extent = scene_info.nerf_normalization["radius"]
```

$$
E \triangleq \text{cameras\_extent} = \text{scene\_info.nerf\_normalization}[\text{"radius"}]
$$

Giá trị $E$ **không được tính trong file này** — nó do `scene_info = sceneLoadTypeCallbacks["Colmap"|"Blender"](...)` (dòng 44, 47) trả về, cụ thể được tính trong `scene/dataset_readers.py` (hàm chuẩn hoá kiểu NeRF++: tìm tâm và bán kính bao các camera). `Scene.__init__` chỉ gán lại thành thuộc tính để dùng về sau:

- Truyền vào `self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)` (dòng 83) — dùng làm `spatial_lr_scale` trong `gaussian_model.py`.
- Gián tiếp tham gia ngưỡng densify $\max_k s_{i,k} \lessgtr p_{dense}\cdot E$ (xem `gaussian_model.md`, `argument/__init__.md` mục `percent_dense`).

---

## 2. Chọn iteration để khôi phục (`load_iteration`)

Trích nguyên văn (`scene/__init__.py`, dòng 33–38):

```python
        if load_iteration:
            if load_iteration == -1:
                self.loaded_iter = searchForMaxIteration(os.path.join(self.model_path, "point_cloud"))
            else:
                self.loaded_iter = load_iteration
            print("Loading trained model at iteration {}".format(self.loaded_iter))
```

Không phải một công thức số học liên tục, mà là một phép chọn rời rạc: nếu `load_iteration = -1` (giá trị sentinel "tự động"), lấy iteration lớn nhất đã lưu checkpoint:

$$
\text{loaded\_iter} =
\begin{cases}
\displaystyle\max\big\{\, k : \text{"iteration\_"} + k \in \text{listdir}(\text{model\_path/point\_cloud}) \,\big\} & \text{nếu load\_iteration} = -1 \\[2mm]
\text{load\_iteration} & \text{ngược lại}
\end{cases}
$$

(hàm `searchForMaxIteration` thực hiện phép $\max$ này; nằm trong `utils/system_utils.py`, ngoài phạm vi file đang xét).

---

## 3. Điều kiện khởi tạo Gaussian từ point cloud (`create_from_pcd` vs `load_ply`)

Trích nguyên văn (`scene/__init__.py`, dòng 77–83):

```python
        if self.loaded_iter:
            self.gaussians.load_ply(os.path.join(self.model_path,
                                                           "point_cloud",
                                                           "iteration_" + str(self.loaded_iter),
                                                           "point_cloud.ply"))
        elif self.gaussians.get_xyz.shape[0] == 0:
            self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)
```

Điều kiện logic chọn một trong hai nhánh loại trừ nhau — khôi phục từ checkpoint nếu có, ngược lại chỉ khởi tạo mới khi tập Gaussian hiện tại thực sự rỗng:

$$
\text{nhánh} =
\begin{cases}
\text{load\_ply}(\cdot) & \text{nếu loaded\_iter} \ne \text{None/0} \\
\text{create\_from\_pcd}(\mathcal{P}, E) & \text{nếu loaded\_iter là None/0 và } N_{\text{hiện tại}} = |\texttt{gaussians.get\_xyz}| = 0 \\
\text{(không làm gì)} & \text{ngược lại}
\end{cases}
$$

với $\mathcal{P}=$ `scene_info.point_cloud` (một `BasicPointCloud`, xem `utils/graphics_utils.md` mục 1) và $E=$ `cameras_extent` (mục 1 ở trên). Số Gaussian khởi tạo $N_{init} = |\mathcal{P}.\text{points}|$ được xác định hoàn toàn bởi `create_from_pcd` trong `gaussian_model.py`, không phải bởi file này.

---

## 4. Danh sách camera theo từng `resolution_scale`

Trích nguyên văn (`scene/__init__.py`, dòng 71–75):

```python
        for resolution_scale in resolution_scales:
            print("Loading Training Cameras")
            self.train_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.train_cameras, resolution_scale, args)
            print("Loading Test Cameras")
            self.test_cameras[resolution_scale] = cameraList_from_camInfos(scene_info.test_cameras, resolution_scale, args)
```

Đây là một ánh xạ (dictionary) theo chỉ số tỉ lệ $s\in\text{resolution\_scales}$, không phải một phép biến đổi số học — mỗi $s$ ứng với một danh sách camera đã được `cameraList_from_camInfos` resize/convert theo đúng tỉ lệ $s$ (logic resize nằm trong `utils/camera_utils.py`, ngoài phạm vi file này):

$$
\text{train\_cameras}[s] = \text{cameraList\_from\_camInfos}(\text{scene\_info.train\_cameras},\, s,\, \text{args}), \quad \forall s \in \text{resolution\_scales}
$$

$$
\text{test\_cameras}[s] = \text{cameraList\_from\_camInfos}(\text{scene\_info.test\_cameras},\, s,\, \text{args}), \quad \forall s \in \text{resolution\_scales}
$$

---

## Bảng tương ứng cú pháp ↔ công thức

| Code | Công thức / vai trò |
|---|---|
| `self.cameras_extent = scene_info.nerf_normalization["radius"]` (dòng 69) | $E=\text{cameras\_extent}$, lưu lại bán kính chuẩn hoá scene (tính ở `dataset_readers.py`) |
| `self.loaded_iter = searchForMaxIteration(...)` (dòng 35) | $\text{loaded\_iter}=\max\{k:\dots\}$ khi `load_iteration=-1` |
| `self.loaded_iter = load_iteration` (dòng 37) | $\text{loaded\_iter}=\text{load\_iteration}$ (giá trị cụ thể truyền vào) |
| `self.gaussians.load_ply(...)` (dòng 78–81) | nhánh khôi phục từ checkpoint khi `loaded_iter` có giá trị |
| `self.gaussians.create_from_pcd(scene_info.point_cloud, self.cameras_extent)` (dòng 83) | nhánh khởi tạo mới khi $N_{\text{hiện tại}}=0$, dùng $(\mathcal{P}, E)$ làm đầu vào |
| `self.train_cameras[resolution_scale] = cameraList_from_camInfos(...)` (dòng 73) | $\text{train\_cameras}[s]=\dots$ |
| `self.test_cameras[resolution_scale] = cameraList_from_camInfos(...)` (dòng 75) | $\text{test\_cameras}[s]=\dots$ |
| `random.shuffle(scene_info.train_cameras)` / `random.shuffle(scene_info.test_cameras)` (dòng 66–67) | hoán vị ngẫu nhiên thứ tự camera — không phải công thức toán, ảnh hưởng thứ tự lấy mẫu khi huấn luyện |

---

## Kiểm chứng tính đúng sai

Không có công thức số học sai lệch nào được phát hiện — file chỉ gọi lại các hàm đã định nghĩa đúng vai trò ở nơi khác (`dataset_readers.py`, `camera_utils.py`, `gaussian_model.py`). Điểm cần lưu ý khi đọc (không phải lỗi):

- `cameras_extent` ($E$) là một con số do file khác tính, `Scene` chỉ là nơi lưu trữ/truyền tiếp — nếu muốn kiểm chứng công thức tính $E$ (bán kính bao các tâm camera, kiểu NeRF++), phải đọc `scene/dataset_readers.py`, không phải file này.
- Điều kiện `elif self.gaussians.get_xyz.shape[0] == 0` (dòng 82) có nghĩa: nếu `gaussians` được truyền vào `Scene.__init__` đã có sẵn điểm (`N>0`) và không có `loaded_iter`, thì **không nhánh nào chạy** — Gaussian giữ nguyên trạng thái đã có sẵn trước khi gọi `Scene(...)` (ví dụ do caller tự khởi tạo thủ công trước).
