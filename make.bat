@echo off
setlocal

if "%~1"=="" goto help
if "%~1"=="all" goto help

if /i "%~1"=="build" goto build
if /i "%~1"=="package" goto package
if /i "%~1"=="docs" goto docs
if /i "%~1"=="docs_only_homepage" goto docs_only_homepage

echo Unknown target: %1
goto help

:help
echo Available targets:
echo   build              - Install the package locally (editable)
echo   package            - Build the sdist and wheel distributions
echo   docs               - Build the documentation
echo   docs_only_homepage - Build only the homepage for fast iteration
goto end

:build
uv pip install -e .
goto end

:package
uv build
goto end

:docs
python scripts\build_docs.py
goto end

:docs_only_homepage
python scripts\build_docs.py --homepage-only
goto end

:end
endlocal
