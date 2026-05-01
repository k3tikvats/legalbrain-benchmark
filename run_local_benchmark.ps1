param(
    [string]$Dataset = "Prarabdha/indian-legal-supervised-fine-tuning-data",
    [int]$EvalSize = 50,
    [int]$RetrievalCorpusSize = 1000,
    [string]$Model = "google/flan-t5-small",
    [string]$OutputDir = "benchmark_outputs/local_run",
    [switch]$SkipGeneration,
    [switch]$CpuOnly
)

$ErrorActionPreference = "Stop"
$Root = Split-Path -Parent $MyInvocation.MyCommand.Path
$env:UV_CACHE_DIR = Join-Path $Root ".uv-cache"
$env:UV_PYTHON_INSTALL_DIR = Join-Path $Root ".uv-python"
$env:HF_HOME = Join-Path $Root ".hf-cache"
$env:TRANSFORMERS_CACHE = Join-Path $env:HF_HOME "transformers"
$env:HF_DATASETS_CACHE = Join-Path $env:HF_HOME "datasets"

$argsList = @(
    "-m", "legal_benchmark.run_benchmark",
    "--dataset", $Dataset,
    "--eval-size", $EvalSize,
    "--retrieval-corpus-size", $RetrievalCorpusSize,
    "--model-name", $Model,
    "--output-dir", $OutputDir,
    "--device", "auto"
)

$ReqFile = "requirements-cuda.txt"
if ($CpuOnly) {
    $ReqFile = "requirements.txt"
}

$uvArgs = @(
    "run",
    "--python", "3.11",
    "--with-requirements", $ReqFile
)

if (-not $CpuOnly) {
    $uvArgs += @(
        "--extra-index-url", "https://download.pytorch.org/whl/cu121",
        "--index-strategy", "unsafe-best-match"
    )
}

if ($SkipGeneration) {
    $argsList += "--skip-generation"
}

$uvArgs += "python"
$uvArgs += $argsList
uv @uvArgs
