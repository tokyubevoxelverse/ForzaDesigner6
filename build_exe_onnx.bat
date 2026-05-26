@echo off
REM Build a directory-based FD6 package with PyInstaller and bundled ONNX Runtime.

setlocal
cd /d "%~dp0"

python "tools\create_line_guide_sobel_onnx.py" "models\line_guide.onnx"

pyinstaller ^
    --clean ^
    --noconfirm ^
    "FD64FH6354221_onnx.spec"

copy /Y "LICENSE" "dist\FD64FH6354221_onnx\LICENSE" >nul
copy /Y "NOTICE" "dist\FD64FH6354221_onnx\NOTICE" >nul
copy /Y "THIRD_PARTY_NOTICES.md" "dist\FD64FH6354221_onnx\THIRD_PARTY_NOTICES.md" >nul

echo.
echo Built: dist\FD64FH6354221_onnx\
endlocal
