param(
    [Parameter(Mandatory = $true)]
    [ValidateSet('Training','Angles','FineBank','Profiles','ProfileReferences','HeightBlindContext','Diagnosis','FineHeldout','FineHeldoutBatched','FineHeldoutTail','BandCorner','Interactions','BlindX','Population','BandSample','SigmaLength','StrictContinuation','HighWidth','BandPopulation','ThirdFactor','Reduction')]
    [string]$Lane,
    [ValidateRange(1,3)]
    [int]$Slots=2,
    [ValidateRange(0,28)]
    [int]$MinFreeGiB=0
)

Set-Location -LiteralPath $PSScriptRoot
$lumericalPython = 'D:\Ansys Inc\v252\Lumerical\python\python.exe'
$env:ROUGHNESS_RCWA_SLOTS = [string]$Slots
$env:OPENBLAS_NUM_THREADS = '2'

function Invoke-StudyPython {
    param([string[]]$Arguments)
    for ($attempt = 0; $attempt -lt 3; $attempt++) {
        while ($MinFreeGiB -gt 0) {
            $memory = Get-CimInstance Win32_OperatingSystem
            if ([math]::Min($memory.FreePhysicalMemory,$memory.FreeVirtualMemory)/1MB -ge $MinFreeGiB) { break }
            Start-Sleep -Seconds 25
        }
        $stamp = Get-Date -Format 'yyyyMMdd_HHmmssfff'
        $prefix = Join-Path $PSScriptRoot ('queue\worker_{0}_{1}' -f $Lane,$stamp)
        $parameters = @{
            FilePath = $lumericalPython
            ArgumentList = @('-u','-B') + $Arguments
            WorkingDirectory = $PSScriptRoot
            WindowStyle = 'Hidden'
            RedirectStandardOutput = "$prefix.out.log"
            RedirectStandardError = "$prefix.err.log"
            PassThru = $true
        }
        $worker = Start-Process @parameters
        Write-Output ('Worker {0} {1}' -f $worker.Id,($Arguments -join ' '))
        $lastNative = $null
        $restart = $false
        while (!$worker.HasExited) {
            $native = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($worker.Id)" |
                Where-Object Name -eq 'fdtd-solutions.exe'
            if ($native) {
                $lastNative = Get-Date
            } elseif ($lastNative -and ((Get-Date) - $lastNative).TotalSeconds -gt 90) {
                $native = Get-CimInstance Win32_Process -Filter "ParentProcessId=$($worker.Id)" |
                    Where-Object Name -eq 'fdtd-solutions.exe'
                $lastMessage=Get-Content -LiteralPath "$prefix.out.log" -Tail 1 -ErrorAction SilentlyContinue
                if (!$native -and !$worker.HasExited -and $lastMessage -notmatch 'Waiting for a study RCWA worker slot|Waiting for memory for a native angle block') {
                    Write-Output ('Native child exited for worker {0}. Resuming its uncommitted cases.' -f $worker.Id)
                    Stop-Process -Id $worker.Id -ErrorAction SilentlyContinue
                    $restart = $true
                    break
                }
            }
            Start-Sleep -Seconds 20
            $worker.Refresh()
        }
        if (!$restart) {
            Write-Output ('Worker exit code {0}' -f $worker.ExitCode)
            return
        }
    }
    Write-Output 'The worker lost its native process on three launches. Completed results remain available.'
}

switch ($Lane) {
    'Training' {
        Invoke-StudyPython @('.\design.py','joint','--threads','5')
        Invoke-StudyPython @('.\design.py','factors','--threads','5')
        Invoke-StudyPython @('.\design.py','bands','--threads','5')
    }
    'Angles' {
        foreach ($cutoff in @('0.1','0.5')) {
            Invoke-StudyPython @('.\rcwa_runner.py','chart','--ku','10','--kv','12','--du','2','--dv','5',
                '--angles','5','--cutoff',$cutoff,'--with-reference','--threads','4')
        }
    }
    'FineBank' {
        Invoke-StudyPython @('.\design.py','modes','--ku','20','--threads','4')
        Invoke-StudyPython @('.\design.py','pairs','--ku','20','--threads','4')
    }
    'Profiles' {
        Invoke-StudyPython @('.\design.py','profiles','--threads','4')
        Invoke-StudyPython @('.\additional.py','height','--threads','4')
        Invoke-StudyPython @('.\contexts.py','--partial')
        Invoke-StudyPython @('.\additional.py','profile_modes','--threads','4')
        Invoke-StudyPython @('.\contexts.py')
    }
    'ProfileReferences' {
        Invoke-StudyPython @('.\additional.py','profile_refs','--threads','3')
    }
    'HeightBlindContext' {
        Invoke-StudyPython @('.\additional.py','height_blind_context','--threads','4')
        Invoke-StudyPython @('.\contexts.py','--partial')
    }
    'Diagnosis' {
        Invoke-StudyPython @('.\additional.py','blind_diagnosis','--angles','3','--threads','4')
    }
    'FineHeldout' {
        Invoke-StudyPython @('.\additional.py','fine_blind','--limit','8','--threads','4')
        Invoke-StudyPython @('.\additional.py','fine_blind','--start','9','--limit','1','--threads','4')
        foreach ($start in @('8','10')) {
            while (((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB) -lt 13) {
                Start-Sleep -Seconds 35
            }
            Invoke-StudyPython @('.\additional.py','fine_blind','--start',$start,'--limit','1','--threads','4')
        }
    }
    'FineHeldoutBatched' {
        $case = 'blind_new_combo_00_s9101'
        $setting = 'k20x16_d2x10_a3_h4'
        Invoke-StudyPython @('.\angular_batches.py',$case,$setting,'--size','1','--threads','4','--min-free','11','--indices','0','1','2','3','4')
        $angleReceipts = 0..8 | ForEach-Object { 'runs\{0}\{1}\angles_{2:00}_{2:00}\done.json' -f $case,$setting,$_ }
        while (@($angleReceipts | Where-Object { !(Test-Path -LiteralPath $_) }).Count -gt 0) {
            Start-Sleep -Seconds 25
        }
        Invoke-StudyPython @('.\angular_batches.py',$case,$setting,'--size','1','--assemble')
        Invoke-StudyPython @('.\angular_batches.py','blind_new_combo_03_s9151',$setting,'--size','1','--threads','4','--min-free','11')
    }
    'FineHeldoutTail' {
        while (!(Test-Path -LiteralPath 'runs\band_3_s4639_b2\k10x8_d4x3.125_a3_h4\done.json')) {
            Start-Sleep -Seconds 25
        }
        Invoke-StudyPython @('.\angular_batches.py','blind_new_combo_00_s9101','k20x16_d2x10_a3_h4','--size','1','--threads','4','--min-free','11','--indices','5','6','7','8')
    }
    'BandCorner' {
        Invoke-StudyPython @('.\design.py','factors','--start','27','--limit','1','--threads','3')
    }
    'Interactions' {
        Invoke-StudyPython @('.\additional.py','interaction_refinement','--threads','4')
    }
    'BlindX' {
        Invoke-StudyPython @('.\additional.py','blind_x_continuation','--angles','3','--threads','4')
    }
    'Population' {
        Invoke-StudyPython @('.\design.py','factors','--threads','5')
        Invoke-StudyPython @('.\design.py','bands','--threads','5')
    }
    'BandSample' {
        Invoke-StudyPython @('.\design.py','bands','--start','0','--limit','12','--threads','4')
        Invoke-StudyPython @('.\design.py','bands','--start','24','--limit','12','--threads','4')
    }
    'SigmaLength' {
        Invoke-StudyPython @('.\additional.py','sigma_length_refinement','--threads','4')
    }
    'StrictContinuation' {
        Invoke-StudyPython @('.\additional.py','strict_bands','--angles','3','--limit','8','--threads','5')
        Invoke-StudyPython @('.\design.py','strict','--start','14','--threads','6')
        Invoke-StudyPython @('.\additional.py','strict_pairs','--angles','3','--start','16','--threads','5')
        Invoke-StudyPython @('.\additional.py','strict_bands','--angles','3','--start','8','--threads','5')
    }
    'HighWidth' {
        Invoke-StudyPython @('.\fine_validation.py','--strong-width','--threads','4')
    }
    'BandPopulation' {
        Invoke-StudyPython @('.\design.py','bands','--threads','5')
    }
    'ThirdFactor' {
        Invoke-StudyPython @('.\third_factor.py','--threads','4')
    }
    'Reduction' {
        Invoke-StudyPython @('.\decisions.py','--direct','--refine','--threads','4')
        Invoke-StudyPython @('.\decisions.py','--summarize')
    }
}

