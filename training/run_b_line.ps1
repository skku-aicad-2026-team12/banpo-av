# [12팀 11/19] B 갈래 한 줄 실행: BC 출발 모델에서 DAgger 2회 → BC → PPO → 평가까지 명령 하나로 돈다.
#
# 사용 (저장소 루트에서):
#   powershell -ExecutionPolicy Bypass -File training\run_b_line.ps1 -Py .\.venv\Scripts\python.exe -Workers 11
#
# 필요한 것 (없으면 시작하지 않고 멈춘다):
#   results\team_bc_1119_<Tag>0\model.pt   출발 BC 모델 (시범만으로 학습한 것)
#   data\bc_demo_b\<Demo>.npz              그 모델을 학습한 선생님 시범
#
# 만드는 것 (Tag = i 기준):
#   data\bc_demo_b\i_dagger1.npz · i_dagger2.npz
#   results\team_bc_1119_i1 · team_bc_1119_i2 · team_rl_1119_M1
#   evaluation\eval\team_bc_1119_i2_30 · team_rl_1119_M1_30 · team_bc_1119_i2_100 · team_rl_1119_M1_100
#   logs\run_b_line_<Tag>.log             전체 출력
#
# 한 단계라도 실패하면 거기서 멈춘다 (옵션을 바꿔 다시 돌리지 말고 로그 끝부분을 보고한다).
# 이미 끝난 단계의 결과 파일이 있으면 그 단계는 건너뛴다 — 중간에 끊겨도 같은 명령으로 이어서 돌릴 수 있다.
param(
    [string]$Py = "python",
    [int]$Workers = 7,          # 수집 · 평가 워커 수 (논리 코어 수 - 1)
    [int]$Envs = 7,             # PPO 워커 수 — L2와 같은 설정(7 × 1168 스텝)을 유지하려면 바꾸지 않는다
    [string]$Tag = "i",
    [string]$Demo = "h8_demo",
    [string]$Rl = "M1"
)
$ErrorActionPreference = "Stop"
$root = Split-Path -Parent $PSScriptRoot
Set-Location $root
New-Item -ItemType Directory -Force logs | Out-Null
$log = "logs\run_b_line_$Tag.log"

function Step($title, $done, [string[]]$cmd) {
    if ($done -and (Test-Path $done)) { "[건너뜀] $title ($done 이미 있음)" | Tee-Object -FilePath $log -Append; return }
    $t0 = Get-Date
    "[시작 $($t0.ToString('HH:mm:ss'))] $title" | Tee-Object -FilePath $log -Append
    $ErrorActionPreference = "Continue"      # PowerShell 5.1: 파이썬이 stderr에 경고만 써도 Stop이면 멈춘다
    & $Py -X utf8 @cmd 2>&1 | ForEach-Object { "$_" } | Tee-Object -FilePath $log -Append | Select-Object -Last 3
    $code = $LASTEXITCODE; $ErrorActionPreference = "Stop"
    if ($code -ne 0) { throw "실패: $title (종료 코드 $code) — $log 끝부분을 보고할 것" }
    "[끝 $((Get-Date).ToString('HH:mm:ss')), $([int]((Get-Date) - $t0).TotalSeconds)초] $title" | Tee-Object -FilePath $log -Append
}

$bc0 = "results\team_bc_1119_${Tag}0\model.pt"
$demo = "data\bc_demo_b\$Demo.npz"
foreach ($f in @($bc0, $demo)) { if (-not (Test-Path $f)) { throw "필요한 파일이 없습니다: $f" } }
"[run_b_line] 커밋 $(git rev-parse --short HEAD), 워커 $Workers, PPO 워커 $Envs, 시작 $(Get-Date -Format 'yyyy-MM-dd HH:mm')" | Tee-Object -FilePath $log -Append

$common = @("--episodes", "420", "--workers", "$Workers", "--paired-seeds", "--safety", "off",
            "--depart-lane", "random", "--depart-speed", "limit", "--out-dir", "data/bc_demo_b")
$bcopt = @("--arrived-only", "data/bc_demo_b/$Demo.npz", "--epochs", "60", "--lane-weight-cap", "5")
$evopt = @("--workers", "$Workers", "--depart-lane", "random", "--depart-speed", "limit")
$d1 = "data/bc_demo_b/${Tag}_dagger1.npz"; $d2 = "data/bc_demo_b/${Tag}_dagger2.npz"
$m1 = "results/team_bc_1119_${Tag}1/model.pt"; $m2 = "results/team_bc_1119_${Tag}2/model.pt"
$rl = "results/team_rl_1119_$Rl/model.pt"

# 1. DAgger 1회차: 출발 모델이 흔들림(σ 0.2) 섞어 운전, 정답은 선생님
Step "DAgger 1회차 수집" $d1 (@("data_collection/collect_pomdp_data.py", "--policy", "ppo", "--model", $bc0, "--stochastic",
    "--label-policy", "rule", "--seed", "17", "--name", "${Tag}_dagger1") + $common)
Step "BC ${Tag}1" $m1 (@("training/train_team.py", "bc", "--data", "data/bc_demo_b/$Demo.npz", $d1, "--name", "team_bc_1119_${Tag}1") + $bcopt)
# 2. DAgger 2회차
Step "DAgger 2회차 수집" $d2 (@("data_collection/collect_pomdp_data.py", "--policy", "ppo", "--model", $m1, "--stochastic",
    "--label-policy", "rule", "--seed", "27", "--name", "${Tag}_dagger2") + $common)
Step "BC ${Tag}2" $m2 (@("training/train_team.py", "bc", "--data", "data/bc_demo_b/$Demo.npz", $d1, $d2, "--name", "team_bc_1119_${Tag}2") + $bcopt)
Step "평가 BC ${Tag}2 (150판)" "evaluation\eval\team_bc_1119_${Tag}2_30\metrics.csv" (@("evaluation/baseline.py", "--model", $m2,
    "--episodes", "30", "--name", "team_bc_1119_${Tag}2_30") + $evopt)
# 3. PPO (L2와 같은 설정: 업데이트당 8,176 transition, lr 5e-5, 가치 먼저 맞추기 8회, 40만 스텝)
Step "PPO $Rl" $rl @("training/train_team.py", "rl", "--init", $m2, "--timesteps", "400000", "--lr", "5e-5", "--n-steps", "8176",
    "--critic-warmup", "8", "--envs", "$Envs", "--depart-lane", "random", "--depart-speed", "limit", "--name", "team_rl_1119_$Rl")
Step "평가 PPO $Rl (150판)" "evaluation\eval\team_rl_1119_${Rl}_30\metrics.csv" (@("evaluation/baseline.py", "--model", $rl,
    "--episodes", "30", "--name", "team_rl_1119_${Rl}_30") + $evopt)
# 4. 500판 평가 (BC · PPO 둘 다 — 어느 쪽을 쓸지는 이 결과로 정한다)
Step "평가 BC ${Tag}2 (500판)" "evaluation\eval\team_bc_1119_${Tag}2_100\metrics.csv" (@("evaluation/baseline.py", "--model", $m2,
    "--episodes", "100", "--name", "team_bc_1119_${Tag}2_100") + $evopt)
Step "평가 PPO $Rl (500판)" "evaluation\eval\team_rl_1119_${Rl}_100\metrics.csv" (@("evaluation/baseline.py", "--model", $rl,
    "--episodes", "100", "--name", "team_rl_1119_${Rl}_100") + $evopt)
"[run_b_line] 전부 끝 $(Get-Date -Format 'HH:mm')" | Tee-Object -FilePath $log -Append
