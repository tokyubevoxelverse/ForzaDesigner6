from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def test_normal_distribution_keeps_line_guide_optional():
    pyproject = (ROOT / "pyproject.toml").read_text(encoding="utf-8")
    build_bat = (ROOT / "build_exe.bat").read_text(encoding="utf-8")
    spec = (ROOT / "FD6MultiSupport.spec").read_text(encoding="utf-8")
    spec_onedir = (ROOT / "FD6MultiSupport_onedir.spec").read_text(encoding="utf-8")
    gitignore = (ROOT / ".gitignore").read_text(encoding="utf-8")

    dependencies_section = pyproject.split("[project.optional-dependencies]", 1)[0]
    assert "onnxruntime" not in dependencies_section
    assert 'line-guide = ["onnxruntime"]' in pyproject
    assert 'line-guide-gpu = ["onnxruntime-gpu"]' in pyproject
    assert 'line-guide-directml = ["onnxruntime-directml"]' in pyproject
    assert "--exclude-module onnxruntime" in build_bat
    assert "excludes=['onnxruntime']" in spec
    assert "excludes=['onnxruntime']" in spec_onedir
    assert "models;" not in build_bat
    assert "models" not in spec.split("datas =", 1)[1].split("binaries =", 1)[0]
    assert "models" not in spec_onedir.split("datas =", 1)[1].split("binaries =", 1)[0]
    assert "models/*.onnx" in gitignore
    assert "!FD6MultiSupport.spec" in gitignore
    assert "!FD6MultiSupport_onnx.spec" in gitignore
    assert "!FD6MultiSupport_onedir.spec" in gitignore
    assert "!tools/create_line_guide_sobel_onnx.py" in gitignore
    assert "!tools/line_guide_benchmark.py" in gitignore
    assert (ROOT / "models" / "README.md").exists()
    assert (ROOT / "models" / "line_guide.example.json").exists()


def test_onnx_distribution_bundles_line_guide_runtime():
    build_bat = (ROOT / "build_exe_onnx.bat").read_text(encoding="utf-8")
    spec = (ROOT / "FD6MultiSupport_onnx.spec").read_text(encoding="utf-8")

    datas_section = spec.split("datas =", 1)[1].split("binaries =", 1)[0]
    assert "tools\\create_line_guide_sobel_onnx.py" in build_bat
    assert "models\\line_guide.onnx" in build_bat
    assert "FD6MultiSupport_onnx.spec" in build_bat
    assert "collect_all('onnxruntime')" in spec
    assert "_keep_onnx_runtime_binary" in spec
    assert "filtered_binaries" in spec
    assert "onnxruntime_providers_cuda.dll" in spec
    assert "onnxruntime_providers_tensorrt.dll" in spec
    assert "excludes=[]" in spec
    assert "('models', 'models')" in datas_section
    assert "FD6MultiSupport_onnx" in spec
