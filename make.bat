@echo off
setlocal

if "%~1"=="" goto help
if "%~1"=="all" goto help

if /i "%~1"=="build" goto build
if /i "%~1"=="package" goto package
if /i "%~1"=="wheels" goto wheels
if /i "%~1"=="docs" goto docs
if /i "%~1"=="docs_all" goto docs_all
if /i "%~1"=="docs_only_homepage" goto docs
if /i "%~1"=="test" goto test

echo Unknown target: %1
goto help

:help
echo Available targets:
echo   build              - Install the package locally (editable)
echo   package            - Build the sdist and wheel distributions
echo   wheels             - Bundle on-device zero-* and compiler wheels into docs/_static
echo   docs               - Build only the homepage for fast iteration
echo   docs_all           - Build the full documentation (incl. operators reference)
echo   test               - Run the full test suite with 100%% coverage enforcement
goto end

:build
uv pip install -e .
goto end

:package
uv build
goto end

:wheels
uv build --wheel
python -c "from pathlib import Path; from ml_switcheroo.sphinx_ext.hooks import bundle_on_device_wheels; bundle_on_device_wheels(Path('.').resolve(), Path('docs/_static').resolve())"
goto end

:docs
python scripts\build_docs.py
goto end

:docs_all
set BUILD_ALL_DOCS=1
python scripts\build_docs.py --build-all
goto end

:test
uv run pytest -n auto --cov=src --cov-fail-under=100 --cov-branch -W error::DeprecationWarning tests
goto end

:end
endlocal
