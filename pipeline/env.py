"""Kiểm tra môi trường, cài đặt phụ thuộc, theo dõi và giải phóng bộ nhớ."""

import gc
import os
import subprocess
import sys

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

DEPS_FLAG = "/content/.deps_ok"
PIP_PACKAGES = ["plyfile", "psutil", "pandas", "matplotlib", "tqdm", "lpips", "opencv-python-headless"]
SUBMODULES = [
    "submodules/diff-gaussian-rasterization_structgs",
    "submodules/fused-ssim",
    "submodules/simple-knn",
]


def _gb(value):
    return value / (1024 ** 3)


def mem():
    import psutil
    import torch

    vm = psutil.virtual_memory()
    usage = dict(ram_used_gb=_gb(vm.used), ram_total_gb=_gb(vm.total), ram_pct=vm.percent)
    if torch.cuda.is_available():
        usage["vram_alloc_gb"] = _gb(torch.cuda.memory_allocated())
        usage["vram_reserved_gb"] = _gb(torch.cuda.memory_reserved())
        usage["vram_total_gb"] = _gb(torch.cuda.get_device_properties(0).total_memory)
    return usage


def show_mem(tag=""):
    usage = mem()
    line = (f"[MEM {tag:<22}] RAM {usage['ram_used_gb']:5.2f}/{usage['ram_total_gb']:4.1f} GB "
            f"({usage['ram_pct']:4.1f}%)")
    if "vram_alloc_gb" in usage:
        line += (f" | VRAM {usage['vram_alloc_gb']:5.2f} alloc / "
                 f"{usage['vram_reserved_gb']:5.2f} reserved / {usage['vram_total_gb']:4.1f} GB")
    print(line)
    return usage


def free_memory(*objects, tag="cleanup"):
    """Xoá tham chiếu lớn, thu gom rác, trả VRAM về cho driver."""
    import torch

    for obj in objects:
        del obj
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    return show_mem(tag)


def check_gpu(require=False):
    """In thông tin GPU/torch/RAM. Trả về dict; `require=True` thì raise nếu không có GPU."""
    import torch

    info = dict(torch=torch.__version__, cuda=torch.version.cuda,
                gpu=None, vram_total_gb=None, available=torch.cuda.is_available())
    if info["available"]:
        info["gpu"] = torch.cuda.get_device_name(0)
        info["vram_total_gb"] = round(_gb(torch.cuda.get_device_properties(0).total_memory), 2)
        print(f"GPU        : {info['gpu']} ({info['vram_total_gb']} GB VRAM)")
    else:
        print("GPU        : KHÔNG THẤY -> Runtime > Change runtime type > T4 GPU")
        if require:
            raise RuntimeError("cần GPU CUDA để train 3DGS")
    print(f"torch      : {info['torch']} | CUDA {info['cuda']}")
    print(f"python     : {sys.version.split()[0]}")
    show_mem("startup")
    return info


def _run(command):
    print("$", " ".join(command))
    return subprocess.run(command, check=False).returncode


def _run_capture(command, tail=150):
    """Chạy command, in stdout/stderr nếu thất bại (thay vì nuốt lỗi bằng -q).

    Trả về returncode. Khi build lỗi (vd thiếu header, nvcc OOM-killed), in
    `tail` dòng cuối để thấy lỗi biên dịch thật thay vì chỉ "thất bại" chung chung.

    Cần `-v` trên command vì pip mặc định giấu output của subprocess
    `setup.py bdist_wheel` (lệnh nvcc/gcc thật) ngay cả khi không có -q --
    thiếu -v thì log chỉ thấy "finished with status 'error' ... See above
    for output" mà không thấy dòng lỗi biên dịch nào.
    """
    print("$", " ".join(command))
    result = subprocess.run(command, capture_output=True, text=True)
    if result.returncode != 0:
        output = (result.stdout or "") + (result.stderr or "")
        lines = output.strip().splitlines()
        print(f"--- {len(lines)} dòng log, {min(tail, len(lines))} dòng cuối ---")
        print("\n".join(lines[-tail:]))
    return result.returncode


def install_dependencies(force=False, flag_path=DEPS_FLAG):
    """Cài pip package + 3 submodule CUDA. Lần đầu ~3-5 phút, sau đó bỏ qua nhờ cờ.

    Cờ chỉ được ghi khi MỌI submodule build thành công -- trước đây cờ ghi vô điều
    kiện, nên một submodule build lỗi (vd thiếu #include) vẫn để lại /content/.deps_ok,
    khiến các lần chạy sau "dependencies đã cài" bỏ qua luôn bước cài lại và
    import ModuleNotFoundError lặp lại vô thời hạn.

    MAX_JOBS được giới hạn (mặc định 2) vì build submodule CUDA biên dịch song song
    nhiều file .cu (~5 file cho diff-gaussian-rasterization) và trên RAM giới hạn của
    Colab, nvcc/cc1plus dễ bị OOM-killed -- lỗi hiện ra chỉ là "thất bại" chung chung
    vì trước đây log bị nuốt bởi pip -q.
    """
    os.environ.setdefault("MAX_JOBS", "2")
    if os.path.exists(flag_path) and not force:
        print("dependencies đã cài (xoá", flag_path, "để cài lại)")
        return False
    _run([sys.executable, "-m", "pip", "-q", "install", *PIP_PACKAGES])
    failed = []
    for module in SUBMODULES:
        if os.path.isdir(module):
            rc = _run_capture([sys.executable, "-m", "pip", "install", "-v", "--no-build-isolation", f"./{module}"])
            if rc != 0:
                failed.append(module)
        else:
            print("bỏ qua submodule không tồn tại:", module)
    if failed:
        raise RuntimeError(
            "build submodule thất bại: " + ", ".join(failed) +
            f" -- {flag_path} KHÔNG được ghi, chạy lại install_dependencies() sau khi sửa lỗi build"
            " (lỗi biên dịch thật đã in ở log phía trên)")
    os.makedirs(os.path.dirname(flag_path) or ".", exist_ok=True)
    open(flag_path, "w").close()
    return True


def clone_repo(repo_url, repo_dir, subdir=None):
    """Clone repo nếu chưa có rồi chuyển cwd vào đó (hoặc vào `subdir` bên trong nó)."""
    if not os.path.isdir(repo_dir):
        _run(["git", "clone", "-q", repo_url, repo_dir])
    else:
        _run(["git", "-C", repo_dir, "pull", "-q"])
    target = os.path.join(repo_dir, subdir) if subdir else repo_dir
    os.chdir(target)
    if target not in sys.path:
        sys.path.insert(0, target)
    print("cwd:", os.getcwd())
    return target
