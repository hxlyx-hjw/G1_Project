param(
    [string]$TargetRoot = 'E:\TEST_RECORD\G1_Project\G1'
)
$ErrorActionPreference = 'Stop'
$appRoot = Join-Path $PSScriptRoot 'G1_LLM_Robot'
$sourceRoot = Join-Path $appRoot 'G1_LLM_Robot'
$entries = @(
    @{ Root = $sourceRoot; Name = 'runs' },
    @{ Root = $sourceRoot; Name = 'outputs' },
    @{ Root = $sourceRoot; Name = 'motions' },
    @{ Root = $sourceRoot; Name = 'skills' },
    @{ Root = $sourceRoot; Name = 'logs' },
    @{ Root = $sourceRoot; Name = 'vedio' },
    @{ Root = $appRoot; Name = 'vedio_self' }
)
if (-not (Test-Path -LiteralPath $sourceRoot)) {
    throw "Project directory not found: $sourceRoot"
}
New-Item -ItemType Directory -Force -Path $TargetRoot | Out-Null
foreach ($entry in $entries) {
    $name = $entry.Name
    $source = Join-Path $entry.Root $name
    $target = Join-Path $TargetRoot $name
    if (Test-Path -LiteralPath $source) {
        $item = Get-Item -LiteralPath $source -Force
        if ($item.LinkType) {
            $resolved = [IO.Path]::GetFullPath([string]$item.Target)
            $expected = [IO.Path]::GetFullPath($target)
            if ($resolved -ne $expected) {
                throw "Junction target mismatch for ${name}: $resolved != $expected"
            }
            Write-Host "[OK] $name -> $resolved"
            continue
        }
        if (Test-Path -LiteralPath $target) {
            throw "Both source and target exist for ${name}; merge manually: $source and $target"
        }
        Move-Item -LiteralPath $source -Destination $target
        Write-Host "[MOVED] $source -> $target"
    } else {
        if (-not (Test-Path -LiteralPath $target)) {
            New-Item -ItemType Directory -Force -Path $target | Out-Null
        }
        Write-Host "[TARGET] $target"
    }
    New-Item -ItemType Junction -Path $source -Target $target | Out-Null
    Write-Host "[LINK] $source -> $target"
}
Write-Host "Data root ready: $TargetRoot"

$externalLinks = @(
    @{ Source = 'D:\SMPL'; Target = (Join-Path $TargetRoot 'models\SMPL') },
    @{ Source = 'D:\WHAM'; Target = (Join-Path $TargetRoot 'models\WHAM') },
    @{ Source = 'D:\VcXsrv'; Target = (Join-Path $TargetRoot 'tools\VcXsrv') },
    @{ Source = 'C:\Users\lenovo\mujoco_menagerie\unitree_g1'; Target = (Join-Path $TargetRoot 'assets\mujoco_menagerie\unitree_g1') },
    @{ Source = 'C:\Users\lenovo\Desktop\EmbodiedAI\G1_LLM_Robot'; Target = (Join-Path $TargetRoot 'G1_LLM_Robot') }
)
foreach ($link in $externalLinks) {
    $source = $link.Source
    $target = $link.Target
    if (Test-Path -LiteralPath $source) {
        $item = Get-Item -LiteralPath $source -Force
        if ($item.LinkType) {
            $resolved = [IO.Path]::GetFullPath([string]$item.Target)
            $expected = [IO.Path]::GetFullPath($target)
            if ($resolved -ne $expected) {
                throw "Junction target mismatch for $source`: $resolved != $expected"
            }
            Write-Host "[OK] $source -> $resolved"
            continue
        }
        Write-Warning "Non-link path exists, leaving untouched: $source"
        continue
    }
    if (-not (Test-Path -LiteralPath $target)) {
        Write-Warning "External target missing: $target"
        continue
    }
    New-Item -ItemType Junction -Path $source -Target $target | Out-Null
    Write-Host "[LINK] $source -> $target"
}
