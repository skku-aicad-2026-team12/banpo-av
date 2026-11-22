"""12팀 학습 스크립트 — BC로 PPO 정책을 초기화하고, 여러 SUMO 워커로 PPO를 병렬 학습한다.

수업 원본 PPO(training/algorithms/ppo.py)는 고치지 않는다. 보상 · 하이퍼파라미터(training/train.py의 REWARD_OVERRIDES · HPARAMS)는
바꿀 수 있지만 두 학습 갈래가 같은 값을 써야 하므로 PR로만 바꾼다 (docs/00.필독/공통_설명.md §4).
PPO 클래스 · 하이퍼파라미터(train.HPARAMS) · 보상(train.REWARD) · 환경 설정은 수업 코드를 그대로 가져다 쓴다.

사용 (프로젝트 루트에서):
    # 1) BC: SUMO 운전자 시범으로 PPO 정책(HybridActorCritic)을 지도학습 → PPO와 같은 형식의 model.pt
    python -X utf8 training/train_team.py bc --data data/baseline/*.npz data/bc_demo/*.npz --epochs 30

    # 2) RL: BC 가중치에서 시작해 PPO (워커 7개 병렬)
    python -X utf8 training/train_team.py rl --init results/team_bc_<시각>/model.pt --envs 7 --timesteps 1000000

    # 평가 · 영상 (나온 model.pt는 수업 PPO 형식이라 기존 도구가 그대로 읽는다)
    python -X utf8 evaluation/baseline.py --model results/team_rl_<시각>/model.pt --episodes 20
    python -X utf8 environment/road/record_video.py --model results/team_rl_<시각>/model.pt

BC (bc):
- 정책: PPO와 같은 HybridActorCritic (hidden · activation은 train.HPARAMS). 그래서 BC 결과를 PPO.load로 그대로 이어 쓴다.
  (수업 train_bc.py(11/16 지움)의 BCPolicy는 구조가 달라 PPO로 옮길 수 없다)
- 손실: 가감속 평균 μ ↔ 시범 가감속 MSE + 차로 3범주(왼 · 유지 · 오) 교차 엔트로피(드문 차로 변경에 가중치) + 가치 V ↔ 시범의 할인 리턴 MSE.
  가치까지 맞춰 두는 이유: RL 첫 업데이트에서 critic이 무작위면 advantage가 잡음이라 BC 정책이 바로 망가진다.
- 시범 데이터: data_collection/collect_pomdp_data.py --policy sumo 의 NPZ (observation · action_raw · reward · terminated · truncated · episode).
  보상은 수집할 때의 train.REWARD 값 — 보상을 바꾸면 시범을 다시 모아야 가치 초기화가 맞는다.
- 끝나면 log_std를 --log-std로 둔다 (기본 -1.6, σ ≈ 0.2): PPO 기본 탐험 폭(σ 0.61)이면 RL 시작부터 BC 행동이 크게 흔들린다.

RL (rl):
- 워커 프로세스마다 SUMO 하나(libsumo) + 정책 사본. 업데이트 1회 = 가중치를 보내고 워커마다 n_steps // 워커 수 스텝을
  각자 모음(서로 기다리지 않음) → 워커별 GAE → 합쳐서 수업 PPO.update 그대로.
  (11/16 검토 C5: 올림이면 2,051개 = 64 × 32 + 3 → 에폭마다 3개짜리 미니배치. 내림이면 2,044개)
- SUMO 오류 (11/16 검토 C3): 에피소드 도중 오류(ego 삽입 실패 등)가 나면 그 에피소드의 이번 rollout 부분만 버리고 SUMO를
  다시 띄운다. 연달아 실패하면 환경을 새로 만들고, 그래도 안 되면 워커가 오류를 알린다. 워커 프로세스가 죽거나 오류를 알리면
  부모가 그 워커만 새로 띄워 같은 rollout을 다시 시킨다 (업데이트마다 워커당 3번까지). 버린 수는 출력 · CSV `worker_errors`.
- 학습 중 평가 대신 끝난 에피소드의 end_reason 비율(도착 · 충돌 · 신호 위반 · 경로 이탈 · 정지 · 순간이동 · 시간초과 · 기타)을 TensorBoard · CSV에 기록한다.
  정식 평가는 evaluation/baseline.py --model (실패 원인별, 같은 시드).
- 체크포인트: --save-every 업데이트마다 model_<스텝>.pt(+ training_log.csv 중간 저장), 끝나면 model.pt. 설정 사본(road_config ·
  mdp_config · training/train.py · 이 파일 · args.json). 끝날 때(오류 · Ctrl+C 포함) 모델 · 로그를 먼저 저장하고 워커를 정리한다.
- 워커가 ROLLOUT_TIMEOUT_S 동안 응답이 없으면 멈춘 것으로 보고 다시 띄운다 (11/16 검토).
- 학습 중에는 도로 폴더에 .in_use_<PID> 표시를 둔다 → 다른 프로세스가 다른 설정으로 build()하면 도로 파일을 하나도 안 바꾸고
  오류 (road_builder.hold). 같은 설정이면 그대로 통과하므로 학습 중 같은 설정의 평가 · 영상은 괜찮다.
"""
import argparse
import copy
import glob
import json
import math
import multiprocessing as mp
import os
import sys
import shutil
import time
from collections import Counter, deque
from datetime import datetime

import numpy as np

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402
BASE = P.ROOT   # 저장소 루트 (results/ · data/는 루트에)
END_KEYS = ["arrived", "collided", "red_light", "off_route", "stuck", "teleported", "timeout", "other"]
MAX_ENV_FAILS = 5        # 워커 안에서 연달아 실패하면 환경을 새로 만들고, 그 두 배면 부모에 알린다
MAX_WORKER_RESTARTS = 3  # 업데이트 한 번에 워커 하나를 다시 띄우는 최대 횟수
ROLLOUT_TIMEOUT_S = 900  # 워커가 이 시간(초) 안에 rollout을 못 돌려주면 멈춘 것으로 보고 다시 띄운다
                         # (평소 업데이트당 rollout 약 15~20초 — 에피소드 재시작이 몇 번 겹쳐도 넉넉하게)


def _hparams(**over):
    from train import HPARAMS
    return {**HPARAMS, **over}


def _run_dir(prefix, name):
    d = os.path.join(BASE, "results", name or datetime.now().strftime(f"{prefix}_%Y%m%d_%H%M%S"))
    os.makedirs(d, exist_ok=False)
    return d


def _snapshot(run_dir, args):
    for src in [os.path.join(P.ENV_PKG, "road_config.py"), os.path.join(P.ENV_PKG, "mdp_config.py"),
                os.path.join(P.TRAINING, "train.py"), os.path.join(P.TRAINING, "train_team.py")]:
        shutil.copy(src, run_dir)
    with open(os.path.join(run_dir, "args.json"), "w", encoding="utf-8") as f:
        json.dump(vars(args), f, ensure_ascii=False, indent=1)


# ══════════════════════════════════════════════════════════════════
# BC
# ══════════════════════════════════════════════════════════════════
def load_demos(paths, gamma, arrived_only=()):
    """NPZ들을 읽어 (obs, accel, lane_idx, value_target, file_episode_key)."""
    obs, acc, lane, ret, key = [], [], [], [], []
    for fi, p in enumerate(paths):
        with np.load(p) as d:
            o, a, r = d["observation"], d["action_raw"], d["reward"]
            done = d["terminated"] | d["truncated"]
            ep = d["episode"]
            end = d["end_reason"] if "end_reason" in d.files else None
        g = np.zeros(len(r), dtype=np.float32)
        run = 0.0
        for t in range(len(r) - 1, -1, -1):           # 에피소드 안에서 뒤로 할인 리턴 (끝 = 0, 시범은 전부 도착)
            if done[t] or (t + 1 < len(r) and ep[t + 1] != ep[t]):
                run = 0.0
            run = r[t] + gamma * run
            g[t] = run
        keep = np.ones(len(r), dtype=bool)
        if p in arrived_only and end is not None:
            # [12팀 11/19] 선생님이 도착하지 못한 시범 판은 뺀다 (end_reason 1 = arrived). 조기 종료를 끈 뒤로 실패 판에는
            # "잘못된 차로 끝에 서서 정지 유지"가 수백 스텝씩 들어 있어, 그대로 배우면 틀린 차로에서 서 있기를 배운다.
            # DAgger 파일(모델이 운전, 선생님 정답)에는 쓰지 않는다 — 실패 직전 상태의 정답이 필요하다.
            last = np.r_[ep[1:] != ep[:-1], True]
            ok = set(ep[last][end[last] == 1].tolist())
            keep = np.isin(ep, list(ok))
            print(f"[BC] {os.path.basename(p)}: 도착 못 한 판 {len(set(ep.tolist())) - len(ok)}개 · {int((~keep).sum()):,} 스텝 제외")
        obs.append(o.astype(np.float32)[keep])
        acc.append(a[:, 0].astype(np.float32)[keep])
        lane.append((1 - np.round(a[:, 1])).astype(np.int64).clip(0, 2)[keep])   # 명령 +1/0/−1 → 인덱스 0/1/2 (HybridActorCritic.LANE_CMDS)
        ret.append(g[keep])
        key.append((fi * 100000 + ep.astype(np.int64))[keep])
    return (np.concatenate(obs), np.concatenate(acc), np.concatenate(lane), np.concatenate(ret), np.concatenate(key))


def cmd_bc(args):
    import torch
    import torch.nn.functional as F
    from algorithms.ppo import PPO

    paths = sorted({p for pat in args.data for p in glob.glob(pat)})
    if not paths:
        raise SystemExit(f"시범 데이터가 없습니다: {args.data}")
    hp = _hparams(device=args.device, seed=args.seed)
    only = {p for pat in (args.arrived_only or []) for p in glob.glob(pat)}
    obs, acc, lane, ret, key = load_demos(paths, hp["gamma"], only)
    agent = PPO(obs_dim=obs.shape[1], act_dim=2, **hp)
    pol, dev = agent.policy, agent.device

    rng = np.random.default_rng(args.seed)
    eps = np.unique(key)
    val_eps = set(rng.choice(eps, size=max(1, int(len(eps) * args.val_fraction)), replace=False).tolist())
    is_val = np.isin(key, list(val_eps))
    tr, va = np.where(~is_val)[0], np.where(is_val)[0]
    cnt = np.bincount(lane[tr], minlength=3).astype(np.float64)
    w = np.clip(cnt.sum() / (3 * np.maximum(cnt, 1)), 1.0, args.lane_weight_cap)   # 드문 변경에 가중치 (상한)
    w = torch.as_tensor(w / w[1], dtype=torch.float32, device=dev)                  # 유지 = 1
    T = lambda x: torch.as_tensor(x, device=dev)  # noqa: E731
    O, A, L, R = T(obs), T(acc), T(lane), T(ret)
    # [12팀 11/19] 차로를 옮겨야 하는 상태(관측만으로 판정: 경로 방향 [31]·[33]으로 내 차로가 못 감 [35..38])에 표본 가중치.
    # 이런 스텝은 전체의 몇 %뿐이라 손실에 묻혀, g1 · g2가 선생님의 "틈 만들기 감속"을 못 따라 해 경로 이탈로 끝났다 (side-banpo 12번 판 추적).
    # 기본 1.0이면 전과 같은 손실.
    SW = torch.ones(len(obs), dtype=torch.float32, device=dev)
    if args.must_change_weight != 1.0:
        if obs.shape[1] != 49:
            raise SystemExit("--must-change-weight는 관측 49차원(lane_dir_obs 켬) 데이터에만 쓸 수 있습니다.")
        d1, d2 = obs[:, 31], obs[:, 33]
        ok1 = np.where(d1 > 0, obs[:, 35], np.where(d1 < 0, obs[:, 36], 1.0))
        ok2 = np.where(d2 > 0, obs[:, 37], np.where(d2 < 0, obs[:, 38], 1.0))
        must = ((d1 != 0) & (ok1 < 0.5)) | ((d2 != 0) & (ok2 < 0.5))
        SW = T(np.where(must, args.must_change_weight, 1.0).astype(np.float32))
        print(f"[BC] 차로를 옮겨야 하는 상태: {int(must.sum()):,} / {len(obs):,} ({must.mean():.1%}), 가중치 {args.must_change_weight}")

    # [12팀 11/19 저녁] 달리는 중에 선생님이 감속하는 스텝의 가감속 손실 가중치 (--brake-weight, 기본 1 = 안 씀).
    # g2 · L2의 남은 실패(신호 위반 · 합류/회전교차로 충돌)를 추적하면 전부 "선생님보다 덜 감속"이었고, 학습 데이터 안에서도
    # 황색 앞 감속 스텝(전체의 0.15%)에서 선생님 평균 -0.38에 모델 -0.17이었다. 드문 감속 스텝이 손실에 묻히지 않게 한다.
    # 조건: 정답 가감속 < --brake-thr(기본 -0.2) 이고 속도 > 2 m/s (서 있을 때의 정지 유지 -0.56은 제외). 차로 손실에는 안 쓴다.
    SWA = SW
    if args.brake_weight != 1.0:
        brake = (acc < args.brake_thr) & (obs[:, 0] * 22.22 > 2.0)
        if args.brake_scope == "no-lane":
            # 차로를 옮겨야 하는 상태(관측 [31..38]로 판정)는 가중치에서 뺀다. 그 감속(틈 만들기)까지 강조하면 모델이 잘못된 차로에서
            # 아예 서 버리고 다시 출발하지 못했다 (g2b4 측도-반포 23 · 27번 판: 분기 앞 3번 차로에서 450초 정지 → 시간초과)
            if obs.shape[1] != 49:
                raise SystemExit("--brake-scope no-lane은 관측 49차원 데이터에만 쓸 수 있습니다.")
            b1, b2 = obs[:, 31], obs[:, 33]
            k1 = np.where(b1 > 0, obs[:, 35], np.where(b1 < 0, obs[:, 36], 1.0))
            k2 = np.where(b2 > 0, obs[:, 37], np.where(b2 < 0, obs[:, 38], 1.0))
            brake &= ~(((b1 != 0) & (k1 < 0.5)) | ((b2 != 0) & (k2 < 0.5)))
        SWA = SW * T(np.where(brake, args.brake_weight, 1.0).astype(np.float32))
        print(f"[BC] 달리는 중 감속 스텝: {int(brake.sum()):,} / {len(obs):,} ({brake.mean():.1%}), 가감속 손실 가중치 {args.brake_weight}")
    # [12팀 11/19 밤] 선생님이 차로 변경을 참는 스텝의 차로 손실 가중치 (--hold-weight, 기본 1 = 안 씀).
    # 대상: 차로를 옮겨야 하는 상태(관측 [31..38])인데 정답이 "유지"인 스텝 — 옮길 쪽 앞 · 뒤차 때문에 선생님이 기다리는 순간.
    # BC 모델들이 이 스텝의 5~8%에서 변경 명령을 냈고, 환경 안전 게이트가 통과시킨 일부가 옆 차와 충돌했다
    # (r8b4n 새 시드 1,000판: 충돌 8판 중 7판이 올림픽대로에서 선생님이 참는 순간의 변경).
    SWL = SW
    if args.hold_weight != 1.0:
        if obs.shape[1] != 49:
            raise SystemExit("--hold-weight는 관측 49차원 데이터에만 쓸 수 있습니다.")
        h1, h2 = obs[:, 31], obs[:, 33]
        q1 = np.where(h1 > 0, obs[:, 35], np.where(h1 < 0, obs[:, 36], 1.0))
        q2 = np.where(h2 > 0, obs[:, 37], np.where(h2 < 0, obs[:, 38], 1.0))
        hold = (((h1 != 0) & (q1 < 0.5)) | ((h2 != 0) & (q2 < 0.5))) & (lane == 1)
        SWL = SW * T(np.where(hold, args.hold_weight, 1.0).astype(np.float32))
        print(f"[BC] 선생님이 차로 변경을 참는 스텝: {int(hold.sum()):,} / {len(obs):,} ({hold.mean():.1%}), 차로 손실 가중치 {args.hold_weight}")
    run_dir = _run_dir("team_bc", args.name)
    _snapshot(run_dir, args)
    print(f"[BC] 파일 {len(paths)}개, transition {len(obs):,} (학습 {len(tr):,} / 검증 {len(va):,}, 에피소드 {len(eps)})")
    print(f"[BC] 차로 명령 수 (왼/유지/오): {cnt.astype(int).tolist()}, 손실 가중치 {np.round(w.cpu().numpy(), 1).tolist()}")
    opt = torch.optim.Adam(pol.parameters(), lr=args.lr)
    # [12팀 11/19] --lr-decay: 학습률을 코사인으로 0.02배까지 낮춘다 (드문 상태의 가감속 오차를 끝까지 줄이려고 — 기본 끔 = 전과 같음)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, T_max=args.epochs, eta_min=args.lr * 0.02) if args.lr_decay else None

    def losses(idx):
        mu, _, logits, v = pol(O[idx])
        sw, swa = SWL[idx], SWA[idx]
        la = (swa * (mu.squeeze(-1) - A[idx]) ** 2).sum() / swa.sum()
        ll = (sw * F.cross_entropy(logits, L[idx], weight=w, reduction="none")).sum() / (sw * w[L[idx]]).sum()
        lv = F.mse_loss(v, R[idx])
        return la, ll, lv, logits

    log = []
    for epoch in range(1, args.epochs + 1):
        pol.train()
        perm = rng.permutation(tr)
        tot = np.zeros(3)
        for s in range(0, len(perm), args.batch_size):
            idx = T(perm[s:s + args.batch_size])
            la, ll, lv, _ = losses(idx)
            loss = la + args.lane_coef * ll + args.value_coef * lv
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(pol.parameters(), 1.0)
            opt.step()
            tot += [la.item() * len(idx), ll.item() * len(idx), lv.item() * len(idx)]
        if sched is not None:
            sched.step()
        pol.eval()
        with torch.no_grad():
            vi = T(va)
            la, ll, lv, logits = losses(vi)
            pred = logits.argmax(-1)
            chg = L[vi] != 1
            recall = float((pred[chg] == L[vi][chg]).float().mean()) if chg.any() else float("nan")
            false_chg = float((pred[~chg] != 1).float().mean())
        row = dict(epoch=epoch, train_accel_mse=tot[0] / len(tr), train_lane_ce=tot[1] / len(tr), train_value_mse=tot[2] / len(tr),
                   val_accel_mse=la.item(), val_lane_ce=ll.item(), val_value_mse=lv.item(),
                   val_lane_change_recall=recall, val_false_change_rate=false_chg)
        log.append(row)
        if epoch == 1 or epoch % 5 == 0 or epoch == args.epochs:
            print(f"[BC] epoch {epoch:3d} | 가감속 MSE {row['train_accel_mse']:.4f}/{row['val_accel_mse']:.4f} | "
                  f"차로 CE {row['train_lane_ce']:.3f}/{row['val_lane_ce']:.3f} (변경 재현율 {recall:.2f}, 잘못된 변경 {false_chg:.3f}) | "
                  f"가치 MSE {row['train_value_mse']:.3f}/{row['val_value_mse']:.3f}", flush=True)
    with torch.no_grad():
        pol.log_std.fill_(args.log_std)
    model = os.path.join(run_dir, "model.pt")
    agent.save(model)
    with open(os.path.join(run_dir, "bc_log.json"), "w", encoding="utf-8") as f:
        json.dump(dict(files=paths, transitions=int(len(obs)), lane_counts=cnt.astype(int).tolist(), epochs=log), f, ensure_ascii=False, indent=1)
    print(f"[BC] 저장: {model}  (σ = {math.exp(args.log_std):.2f})")
    print(f"     평가: python -X utf8 evaluation/baseline.py --model {os.path.relpath(model, BASE)} --episodes 10")
    print(f"     RL:   python -X utf8 training/train_team.py rl --init {os.path.relpath(model, BASE)}")


# ══════════════════════════════════════════════════════════════════
# RL — 병렬 워커
# ══════════════════════════════════════════════════════════════════
def _worker(remote, sumocfg, seed, safety, hp, depart=None):
    """워커 프로세스: SUMO 환경 하나 + 정책 사본. ("rollout", (가중치, T)) → 이 워커가 T 스텝을 혼자 모아 돌려준다.

    워커끼리 스텝을 맞추지 않는다 — 한 워커가 에피소드를 새로 시작(SUMO 재시작 + 배경 교통 흘려 두기, 수 초)해도
    다른 워커는 기다리지 않는다 (11/15: 스텝마다 맞추는 방식은 워커 7개로 60 스텝/초였다).

    SUMO 오류 (11/16 검토 C3): step · reset에서 예외가 나면 진행 중인 에피소드의 이번 rollout 부분을 버리고 SUMO를 다시
    띄운다 (버린 스텝은 T에 세지 않으므로 그만큼 더 모은다). 정상으로 끝난 에피소드 없이 MAX_ENV_FAILS번 연달아 실패하면
    환경을 새로 만들고, 2 × MAX_ENV_FAILS번이면 부모에 오류를 알리고 끝난다 (부모가 이 워커를 새로 띄운다).
    에피소드가 이전 rollout에서 시작했다면 그 앞부분은 이미 보낸 대로 둔다 (그때 끝 가치로 이어 받았으므로 그대로 유효)."""
    import numpy as _np
    import torch
    torch.set_num_threads(1)
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import SIMULATION, OBSERVATION, ACTION
    from env.sumo_env import SumoHighwayEnv
    from utils.networks import HybridActorCritic
    from train import REWARD
    _np.random.seed(seed)                    # sumo_env가 SUMO 시드 · 시나리오 · 경로를 전역 난수에서 뽑는다
    torch.manual_seed(seed)
    action = dict(ACTION, safety_brake=dict(ACTION.get("safety_brake") or {}, enabled=safety))

    def make_env():
        return SumoHighwayEnv(cfg_path=sumocfg, road=ROAD, ego=dict(EGO, sumo_driver=False), mdp_sim=SIMULATION,
                              mdp_obs=OBSERVATION, action=action, reward=REWARD, traffic=TRAFFIC, gui=False)

    st = {"env": make_env(), "fails": 0}     # fails: 정상 종료 에피소드 없이 연달아 난 SUMO 오류 수
    pol = HybridActorCritic(st["env"].observation_space.shape[0], hidden_sizes=hp["hidden_sizes"],
                            activation=hp["activation"], init_log_std=hp["init_log_std"], lane_keep_bias=hp["lane_keep_bias"])
    pol.eval()
    gamma = hp["gamma"]
    keys = ("obs", "actions", "log_probs", "rewards", "values", "dones")

    def value(o):
        with torch.no_grad():
            return float(pol(torch.as_tensor(o, dtype=torch.float32).unsqueeze(0))[-1].item())

    def on_fail(e):
        """오류 한 번 처리: 연속 실패 수를 세고, 많으면 환경을 새로 만들거나 예외를 다시 던진다."""
        st["fails"] += 1
        if st["fails"] >= 2 * MAX_ENV_FAILS:
            raise RuntimeError(f"SUMO 오류 {st['fails']}번 연속") from e
        if st["fails"] % MAX_ENV_FAILS == 0:
            try:
                st["env"].close()
            except Exception:                 # noqa: BLE001 — 이미 망가진 SUMO 정리 실패는 무시
                pass
            st["env"] = make_env()

    def reset():
        while True:
            try:
                return st["env"].reset(options=(depart or None))[0]   # [12팀 11/19] 출발 차로 · 속도 옵션 (--depart-lane · --depart-speed)
            except Exception as e:            # noqa: BLE001 — SUMO 쪽 예외 종류가 traci/libsumo마다 다르다
                on_fail(e)

    try:
        obs = reset()
        ep_ret, ep_len = 0.0, 0
        while True:
            cmd, data = remote.recv()
            if cmd == "close":
                break
            state_dict, T = data
            pol.load_state_dict(state_dict)
            out = {k: [] for k in keys}
            episodes, interventions, errors = [], 0, 0
            ep_start, ep_interv = 0, 0            # 진행 중인 에피소드가 이번 rollout에서 시작한 위치 · 그 에피소드의 개입 수
            while len(out["obs"]) < T:
                with torch.no_grad():
                    a, lp, v = pol.sample(torch.as_tensor(obs, dtype=torch.float32).unsqueeze(0))
                a = a.squeeze(0).numpy().astype(_np.float32)
                try:
                    nxt, r, term, trunc, info = st["env"].step(a)
                except Exception as e:            # noqa: BLE001
                    errors += 1
                    for k in keys:
                        del out[k][ep_start:]
                    interventions -= ep_interv
                    ep_ret, ep_len, ep_interv = 0.0, 0, 0
                    on_fail(e)
                    obs = reset()
                    ep_start = len(out["obs"])
                    continue
                rr = r
                if trunc and not term:            # 시간 제한으로 잘림 → 이어졌다면 받았을 가치로 보정. 수업 PPO는 잘림도
                    rr += gamma * value(nxt)      # 실패처럼 0으로 끝내므로, 이 처리가 수업 코드보다 정확하다
                for k, x in (("obs", obs), ("actions", a), ("log_probs", float(lp.item())), ("rewards", rr),
                             ("values", float(v.item())), ("dones", float(term or trunc))):
                    out[k].append(x)
                ep_ret += r
                ep_len += 1
                sb = bool(info.get("safety_brake"))
                interventions += sb
                ep_interv += sb
                if term or trunc:
                    episodes.append((ep_ret, ep_len, info.get("end_reason") or "other"))
                    ep_ret, ep_len, ep_interv = 0.0, 0, 0
                    st["fails"] = 0
                    nxt = reset()
                    ep_start = len(out["obs"])
                obs = nxt
            out["last_value"] = value(obs)
            out["episodes"] = episodes
            out["interventions"] = interventions
            out["errors"] = errors
            remote.send(out)
    except Exception as e:                    # noqa: BLE001 — 부모에 알리고 끝냄 (부모가 이 워커를 새로 띄운다)
        import traceback
        try:
            remote.send({"error": repr(e) + "\n" + traceback.format_exc()[-1500:]})
        except (BrokenPipeError, OSError):
            pass
    finally:
        try:
            st["env"].close()
        except Exception:                     # noqa: BLE001
            pass
        remote.close()


class Workers:
    def __init__(self, n, sumocfg, seed, safety, hp, depart=None):
        self.args = (sumocfg, seed, safety, hp)
        self.depart = depart
        self.ctx = mp.get_context("spawn")
        self.remotes, self.procs, self.gen = [None] * n, [None] * n, [0] * n
        for i in range(n):
            self._spawn(i)

    def _spawn(self, i):
        sumocfg, seed, safety, hp = self.args
        a, b = self.ctx.Pipe()
        # 다시 띄운 워커는 시드를 바꾼다 (같은 시드면 같은 오류를 되풀이할 수 있다)
        p = self.ctx.Process(target=_worker, args=(b, sumocfg, seed * 1000 + i + 100 * self.gen[i], safety, hp, self.depart), daemon=True)
        p.start()
        b.close()
        self.remotes[i], self.procs[i] = a, p

    def _restart(self, i):
        try:
            self.remotes[i].close()
        except OSError:
            pass
        p = self.procs[i]
        if p.is_alive():
            p.terminate()
        p.join(timeout=15)
        self.gen[i] += 1
        self._spawn(i)

    def rollout(self, state_dict, T):
        """모든 워커에 rollout을 시키고 결과를 모은다. 오류를 알리거나 죽은 워커는 새로 띄워 같은 rollout을 다시 시킨다.
        반환: (워커별 결과, 이번에 다시 띄운 워커 수)."""
        for r in self.remotes:
            r.send(("rollout", (state_dict, T)))
        outs, restarts = [], 0
        for i in range(len(self.remotes)):
            tries = 0
            while True:
                try:
                    if self.remotes[i].poll(ROLLOUT_TIMEOUT_S):
                        o = self.remotes[i].recv()
                    else:                                # 응답 없음 — SUMO가 예외 없이 멈춘 경우 (11/16 검토)
                        o = {"error": f"워커 응답 없음 ({ROLLOUT_TIMEOUT_S} s)"}
                except (EOFError, OSError) as e:         # 워커 프로세스가 죽음 (libsumo 치명 오류 등)
                    o = {"error": f"워커 프로세스 종료 ({e!r}, exitcode {self.procs[i].exitcode})"}
                if "error" not in o:
                    outs.append(o)
                    break
                tries += 1
                if tries > MAX_WORKER_RESTARTS:
                    raise RuntimeError(f"워커 {i}가 {MAX_WORKER_RESTARTS}번 다시 띄워도 실패: {o['error']}")
                print(f"[RL] 워커 {i} 오류 → 다시 띄움 ({tries}/{MAX_WORKER_RESTARTS}): {o['error'].splitlines()[0]}", flush=True)
                self._restart(i)
                restarts += 1
                try:
                    self.remotes[i].send(("rollout", (state_dict, T)))
                except (BrokenPipeError, OSError):       # 새 워커가 곧바로 죽음 → 다음 recv에서 EOF로 처리
                    pass
        return outs, restarts

    def close(self, wait=15):
        for r in self.remotes:
            try:
                r.send(("close", None))
            except (BrokenPipeError, OSError):
                pass
        for p in self.procs:
            p.join(timeout=wait)
            if p.is_alive():
                p.terminate()


def critic_update(agent, anchor, returns, hp, anchor_coef):
    """가치 맞추기 업데이트 1회 (--critic-warmup): 가치 V를 이번 rollout의 GAE 리턴에 회귀하고, 정책 출력은 출발점에 묶어 둔다.

    BC의 가치는 시범 데이터의 옛 보상으로 맞춘 것이라, 보상을 바꾼 뒤 그대로 PPO를 시작하면 첫 업데이트부터 advantage가
    어긋난다. 가치와 정책이 trunk를 함께 쓰므로 가치만 학습해도 정책이 바뀔 수 있어, 출발점 정책(anchor)과의 차이
    (가감속 평균 μ의 MSE + 차로 분포 KL)를 anchor_coef만큼 손실에 더한다. log σ는 건드리지 않는다.
    반환 통계: value_loss(마지막 에폭 평균) · anchor_mu(|Δμ| 평균) · anchor_lane_kl (업데이트 뒤, 이번 관측 전체)."""
    import torch
    import torch.nn.functional as F
    pol, dev = agent.policy, agent.device
    obs_t, _, _ = agent.buffer.to_tensors(dev)
    ret_t = torch.as_tensor(returns, device=dev)
    with torch.no_grad():
        mu0, _, lg0, _ = anchor(obs_t)
        lp0 = F.log_softmax(lg0, -1)
    params = [p for n, p in pol.named_parameters() if n != "log_std"]
    opt = getattr(agent, "_critic_opt", None)
    if opt is None:
        opt = agent._critic_opt = torch.optim.Adam(params, lr=hp["lr"])
    idx = np.arange(len(ret_t))
    for _ in range(hp["n_epochs"]):
        np.random.shuffle(idx)
        tot, nb = 0.0, 0
        for s in range(0, len(idx), hp["minibatch_size"]):
            mb = torch.as_tensor(idx[s:s + hp["minibatch_size"]], device=dev)
            mu, _, lg, v = pol(obs_t[mb])
            lv = F.mse_loss(v, ret_t[mb])
            lane_kl = F.kl_div(F.log_softmax(lg, -1), lp0[mb], log_target=True, reduction="batchmean")
            loss = lv + anchor_coef * (F.mse_loss(mu, mu0[mb]) + lane_kl)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
            opt.step()
            tot, nb = tot + lv.item(), nb + 1
    with torch.no_grad():
        mu, _, lg, _ = pol(obs_t)
        kl = F.kl_div(F.log_softmax(lg, -1), lp0, log_target=True, reduction="batchmean").item()
    return {"value_loss": tot / max(nb, 1), "anchor_mu": float((mu - mu0).abs().mean()), "anchor_lane_kl": kl,
            "policy_loss": 0.0, "entropy": 0.0, "approx_kl": 0.0, "clip_frac": 0.0}


def teacher_update(agent, label, obs, hp, lr, epochs, brake_weight, lane_cap=5.0, lane_coef=0.5):
    """[12팀 11/20] 선생님 보정 1회 (--teacher-lr): PPO 업데이트 직후, 이번 rollout의 관측에 선생님(규칙 정책) 정답을 매겨 BC 한 번.

    PPO를 오래 돌리면 신호 위반은 줄지만 감속이 풀려 충돌이 늘었다 (P1 · P2 · Q · T5). 보상이 끄는 쪽은 PPO가, 감속은 선생님이
    잡게 한다. 손실은 BC와 같다: 가감속 MSE(달리는 중 감속 스텝 × brake_weight, 차로를 옮겨야 하는 상태 제외) + 0.5 × 차로 CE
    (드문 변경 가중치, 상한 5). 가치 · log σ는 건드리지 않는다. 세기는 lr · epochs로 정한다 (Adam이라 손실 배율은 효과가 없다).
    수업 PPO.update는 그대로 쓰고, 이 함수는 그 뒤에 따로 돈다. 최종 모델은 신경망뿐이고 규칙 정책은 학습 때의 정답으로만 쓰인다."""
    import torch
    import torch.nn.functional as F
    pol, dev = agent.policy, agent.device
    tgt = np.asarray([label(o) for o in obs], dtype=np.float32).reshape(len(obs), -1)
    acc, lane = tgt[:, 0], (1 - np.round(tgt[:, 1])).astype(np.int64).clip(0, 2)
    swa = np.ones(len(obs), dtype=np.float32)
    if brake_weight != 1.0 and obs.shape[1] == 49:
        b1, b2 = obs[:, 31], obs[:, 33]
        k1 = np.where(b1 > 0, obs[:, 35], np.where(b1 < 0, obs[:, 36], 1.0))
        k2 = np.where(b2 > 0, obs[:, 37], np.where(b2 < 0, obs[:, 38], 1.0))
        brake = (acc < -0.2) & (obs[:, 0] * 22.22 > 2.0) & ~(((b1 != 0) & (k1 < 0.5)) | ((b2 != 0) & (k2 < 0.5)))
        swa = np.where(brake, brake_weight, 1.0).astype(np.float32)
    cnt = np.bincount(lane, minlength=3).astype(np.float64)
    w = torch.as_tensor(np.clip(cnt.sum() / (3 * np.maximum(cnt, 1)), 1.0, lane_cap), dtype=torch.float32, device=dev)
    O, A, L, S = (torch.as_tensor(x, device=dev) for x in (obs.astype(np.float32), acc, lane, swa))
    params = [p for n, p in pol.named_parameters() if n != "log_std"]
    opt = getattr(agent, "_teacher_opt", None)
    if opt is None:
        opt = agent._teacher_opt = torch.optim.Adam(params, lr=lr)

    def gap():
        with torch.no_grad():
            mu, _, lg, _ = pol(O)
            return float((mu.squeeze(-1) - A).abs().mean()), float((lg.argmax(-1) != L).float().mean())

    g0 = gap()
    idx = np.arange(len(obs))
    for _ in range(epochs):
        np.random.shuffle(idx)
        for s in range(0, len(idx), hp["minibatch_size"]):
            mb = torch.as_tensor(idx[s:s + hp["minibatch_size"]], device=dev)
            mu, _, lg, _ = pol(O[mb])
            la = (S[mb] * (mu.squeeze(-1) - A[mb]) ** 2).sum() / S[mb].sum()
            ll = F.cross_entropy(lg, L[mb], weight=w)
            opt.zero_grad()
            (la + lane_coef * ll).backward()
            torch.nn.utils.clip_grad_norm_(params, hp["max_grad_norm"])
            opt.step()
    g1 = gap()
    return {"teacher_mu_before": g0[0], "teacher_mu_after": g1[0], "teacher_lane_before": g0[1], "teacher_lane_after": g1[1]}


def cmd_rl(args):
    import torch
    from algorithms.ppo import PPO
    from utils.buffer import RolloutBuffer
    from utils.logger import RunLogger
    from env import road_builder
    from env.road_config import ROAD, EGO, TRAFFIC
    from env.mdp_config import OBSERVATION

    hp = _hparams(device=args.device, seed=args.seed, **({"lr": args.lr} if args.lr else {}),
                  **({"n_steps": args.n_steps} if args.n_steps else {}))
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC, P.SUMO_DIR)   # 부모에서 한 번 (워커는 읽기만)
    run_dir = _run_dir("team_rl", args.name)
    _snapshot(run_dir, args)
    logger = RunLogger(run_dir)
    from env.sumo_env import SumoHighwayEnv
    from env.mdp_config import SIMULATION, ACTION
    from train import REWARD
    obs_dim = SumoHighwayEnv(cfg_path=sumocfg, road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION, action=ACTION,
                             reward=REWARD, traffic=TRAFFIC).observation_space.shape[0]   # 생성만으로는 SUMO를 띄우지 않는다
    agent = PPO(obs_dim=obs_dim, act_dim=2, buffer=RolloutBuffer(), **hp)
    if args.init:
        agent.load(args.init)
        print(f"[RL] 초기 가중치: {args.init} (σ = {agent.policy.current_std:.2f})")
    pol = agent.policy
    label = None
    if args.teacher_lr > 0:                   # 선생님 보정 (teacher_update 머리말)
        road_dir, cwd = os.path.join(P.ROOT, "environment", "road"), os.getcwd()
        if road_dir not in sys.path:
            sys.path.insert(0, road_dir)
        import check_env                      # 불러오며 작업 폴더를 바꾼다 → 되돌린다
        os.chdir(cwd)
        label = check_env.rule_policy(SumoHighwayEnv(cfg_path=sumocfg, road=ROAD, ego=EGO, mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
                                                     action=ACTION, reward=REWARD, traffic=TRAFFIC))
        print(f"[RL] 선생님 보정: PPO 업데이트마다 BC {args.teacher_epochs}에폭, lr {args.teacher_lr:g}, 감속 가중치 {args.teacher_brake_weight:g}", flush=True)
    n_env = args.envs
    T = hp["n_steps"] // n_env               # 11/16 검토 C5: 올림이면 마지막 미니배치가 3개짜리 (위 머리말)
    lock = road_builder.hold(os.path.dirname(sumocfg))   # 학습 중에는 다른 설정의 build가 도로 파일을 못 바꾸게 (11/16 검토)
    depart = {k: v for k, v in (("depart_lane", "random" if args.depart_lane == "random" else None),
                                ("depart_speed", "limit" if args.depart_speed == "limit" else None)) if v}
    workers = Workers(n_env, sumocfg, args.seed, args.safety, hp, depart)
    print(f"[RL] 워커 {n_env}개 × {T} 스텝 = 업데이트당 {n_env * T} transition, 총 {args.timesteps:,} 스텝 → {run_dir}", flush=True)
    ends, rets = deque(maxlen=args.window), deque(maxlen=args.window)
    n_upd, t0, total_errors = 0, time.time(), 0
    if args.critic_warmup:
        print(f"[RL] 가치 먼저 맞추기: 업데이트 {args.critic_warmup}번 (정책은 고정 — 이 스텝은 --timesteps에 세지 않는다)", flush=True)
    budget = args.critic_warmup * n_env * T + args.timesteps
    try:
        while agent.num_timesteps < budget:
            warm = n_upd < args.critic_warmup
            sd = {k: v.detach().cpu() for k, v in pol.state_dict().items()}
            t_roll = time.time()
            outs, restarts = workers.rollout(sd, T)
            t_roll = time.time() - t_roll
            agent.buffer.clear()
            advs, rts = [], []
            interventions = 0
            errors = restarts + sum(o["errors"] for o in outs)   # 버린 에피소드 조각 + 다시 띄운 워커
            for o in outs:                        # 워커별 GAE → 합쳐서 수업 PPO.update 그대로
                b = RolloutBuffer()
                for k in ("obs", "actions", "log_probs", "rewards", "values", "dones"):
                    getattr(b, k).extend(o[k])
                    getattr(agent.buffer, k).extend(o[k])
                ad, rt = b.compute_gae(o["last_value"], hp["gamma"], hp["gae_lambda"])
                advs.append(ad)
                rts.append(rt)
                interventions += o["interventions"]
                for ret, length, end in o["episodes"]:
                    ends.append(end)
                    rets.append(ret)
                    logger.log_episode(agent.num_timesteps, float(ret), int(length))
            agent.num_timesteps += n_env * T
            if warm:
                if n_upd == 0:
                    anchor = copy.deepcopy(pol).eval()
                stats = critic_update(agent, anchor, np.concatenate(rts), hp, args.anchor_coef)
            else:
                obs_np = np.asarray(agent.buffer.obs, dtype=np.float32) if label is not None else None
                stats = agent.update(np.concatenate(advs), np.concatenate(rts))
                if label is not None:
                    stats = {**stats, **teacher_update(agent, label, obs_np, hp, args.teacher_lr, args.teacher_epochs,
                                                       args.teacher_brake_weight)}
            n_upd += 1
            c, n = Counter(ends), max(len(ends), 1)
            m = {"update": n_upd, "warmup": int(warm),
                 "ep_rew_mean": float(np.mean(rets)) if rets else float("nan"), "std": pol.current_std,
                 **stats, **{f"end_{k}": c.get(k, 0) / n for k in END_KEYS}, "episodes_window": len(ends),
                 "steps_per_s": agent.num_timesteps / max(time.time() - t0, 1e-9), "rollout_s": t_roll,
                 "worker_errors": errors}
            total_errors += errors
            if args.safety:
                m["safety_brake_frac"] = interventions / (T * n_env)
            logger.log_train(agent.num_timesteps, m, verbose=False)
            if warm:                              # 가치 맞추기: 매번 출력, 체크포인트 없음 (정책이 출발점 그대로)
                print(f"[RL] 가치 맞추기 {n_upd}/{args.critic_warmup} | 리턴 {m['ep_rew_mean']:7.2f} | "
                      f"도착 {m['end_arrived']:.0%} 충돌 {m['end_collided']:.0%} 신호위반 {m['end_red_light']:.0%} "
                      f"(최근 {len(ends)}회) | 가치 손실 {stats['value_loss']:.3f} | "
                      f"정책 변화 |Δμ| {stats['anchor_mu']:.4f} 차로 KL {stats['anchor_lane_kl']:.5f}"
                      + (f" | SUMO 오류 {errors} (누적 {total_errors})" if errors else ""), flush=True)
                if n_upd == args.critic_warmup:
                    print(f"[RL] 가치 맞추기 끝 → PPO 시작 ({args.timesteps:,} 스텝)", flush=True)
                continue
            if (n_upd - args.critic_warmup) % args.log_every == 0:
                print(f"[RL] {agent.num_timesteps:>9,} 스텝 | 업데이트 {n_upd} | 리턴 {m['ep_rew_mean']:7.2f} | "
                      f"도착 {m['end_arrived']:.0%} 충돌 {m['end_collided']:.0%} 신호위반 {m['end_red_light']:.0%} "
                      f"이탈 {m['end_off_route']:.0%} 정지 {m['end_stuck']:.0%} 순간이동 {m['end_teleported']:.0%} "
                      f"시간초과 {m['end_timeout']:.0%} 기타 {m['end_other']:.0%} "
                      f"(최근 {len(ends)}회) | σ {pol.current_std:.2f} kl {stats['approx_kl']:.4f} | {m['steps_per_s']:.0f} 스텝/초"
                      + (f" | 선생님과 차이 |Δμ| {stats['teacher_mu_before']:.4f}→{stats['teacher_mu_after']:.4f} "
                         f"차로 {stats['teacher_lane_before']:.2%}→{stats['teacher_lane_after']:.2%}" if label is not None else "")
                      + (f" | SUMO 오류 {errors} (누적 {total_errors})" if errors else ""), flush=True)
            if args.save_every and (n_upd - args.critic_warmup) % args.save_every == 0:
                agent.save(os.path.join(run_dir, f"model_{agent.num_timesteps}.pt"))
                logger.save_csv()                 # 비정상 종료에도 학습 곡선 CSV가 남게
    finally:
        model = os.path.join(run_dir, "model.pt")
        agent.save(model)                     # 저장을 먼저 (워커 정리가 늦어지는 동안 강제 종료돼도 남게)
        logger.close()
        print(f"[RL] 저장: {model}", flush=True)
        workers.close(wait=5)
        lock.release()
        print(f"     평가: python -X utf8 evaluation/baseline.py --model {os.path.relpath(model, BASE)} --episodes 20")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    sub = ap.add_subparsers(dest="cmd", required=True)
    b = sub.add_parser("bc", help="SUMO 운전자 시범으로 PPO 정책 초기화")
    b.add_argument("--data", nargs="+",   # 기본값은 절대 경로 — 어느 폴더에서 실행해도 같은 시범 데이터
                   default=[os.path.join(P.DATA, "baseline", "*.npz"), os.path.join(P.DATA, "bc_demo", "*.npz")])
    b.add_argument("--epochs", type=int, default=30)
    b.add_argument("--batch-size", type=int, default=512)
    b.add_argument("--lr", type=float, default=1e-3)
    b.add_argument("--val-fraction", type=float, default=0.15, help="검증으로 뺄 에피소드 비율")
    b.add_argument("--lane-coef", type=float, default=0.5)
    b.add_argument("--lane-weight-cap", type=float, default=20.0, help="드문 차로 변경 가중치 상한")
    b.add_argument("--arrived-only", nargs="*", default=[], help="이 파일들(--data와 같은 식의 경로 · 패턴)에서는 도착한 판만 쓴다 — 선생님 시범용")
    b.add_argument("--brake-weight", type=float, default=1.0, help="달리는 중 선생님이 감속하는 스텝의 가감속 손실 가중치 (기본 1 = 안 씀)")
    b.add_argument("--hold-weight", type=float, default=1.0, help="차로를 옮겨야 하는데 선생님이 참는(유지) 스텝의 차로 손실 가중치 (기본 1 = 안 씀)")
    b.add_argument("--brake-scope", choices=["all", "no-lane"], default="all", help="--brake-weight 대상 범위: all = 감속 스텝 전부 · no-lane = 차로를 옮겨야 하는 상태는 제외")
    b.add_argument("--brake-thr", type=float, default=-0.2, help="--brake-weight 대상: 정답 가감속이 이 값보다 작은 스텝")
    b.add_argument("--lr-decay", action="store_true", help="학습률 코사인 감소 (끝 = lr × 0.02)")
    b.add_argument("--must-change-weight", type=float, default=1.0,
                   help="차로를 옮겨야 하는 상태(관측 [31..38]로 판정)의 표본 가중치 — 가감속 · 차로 손실 (기본 1 = 안 씀, 49차원 전용)")
    b.add_argument("--value-coef", type=float, default=0.5)
    b.add_argument("--log-std", type=float, default=-1.6, help="BC 뒤 가감속 탐험 폭 log σ (RL 시작값)")
    b.add_argument("--seed", type=int, default=0)
    b.add_argument("--device", default="cpu")
    b.add_argument("--name", default=None)
    r = sub.add_parser("rl", help="PPO 병렬 학습")
    r.add_argument("--init", default=None, help="시작 가중치 (training/train_team.py bc 또는 이전 rl의 model.pt)")
    r.add_argument("--envs", type=int, default=7)
    r.add_argument("--timesteps", type=int, default=1_000_000)
    r.add_argument("--lr", type=float, default=None, help="train.HPARAMS lr 덮어쓰기")
    r.add_argument("--n-steps", type=int, default=None, help="train.HPARAMS n_steps(업데이트당 수집 스텝) 덮어쓰기")
    r.add_argument("--critic-warmup", type=int, default=0,
                   help="PPO 전에 정책을 고정하고 가치만 맞추는 업데이트 수 (보상을 바꾼 뒤 BC 가치를 새 보상에 맞춤). "
                        "이 스텝은 --timesteps에 세지 않는다")
    r.add_argument("--anchor-coef", type=float, default=10.0, help="가치 맞추기 중 정책을 출발점에 묶는 손실 가중치")
    r.add_argument("--teacher-lr", type=float, default=0.0,
                   help="[12팀 11/20] 선생님 보정: PPO 업데이트마다 이번 rollout 관측에 규칙 정책 정답을 매겨 BC (0 = 안 씀). 값 = 그 BC의 학습률")
    r.add_argument("--teacher-epochs", type=int, default=1, help="선생님 보정 한 번의 에폭 수")
    r.add_argument("--teacher-brake-weight", type=float, default=4.0, help="선생님 보정의 달리는 중 감속 스텝 가중치 (BC --brake-weight · no-lane과 같은 조건)")
    r.add_argument("--safety", action="store_true", help="안전장치(충돌 방지 제동) 켜고 학습")
    r.add_argument("--depart-lane", choices=["best", "random"], default="best",
                   help="[12팀 11/19] AV 출발 차로: best = 경로 차로(기존), random = 경로별 출발 범위에서 무작위 (평가 · 수집과 같은 옵션)")
    r.add_argument("--depart-speed", choices=["default", "limit"], default="default",
                   help="[12팀 11/19] AV 출발 속도: default = EGO depart_speed, limit = 출발 차로 제한속도 × 0.8")
    r.add_argument("--window", type=int, default=100, help="end_reason 비율을 낼 최근 에피소드 수")
    r.add_argument("--log-every", type=int, default=1)
    r.add_argument("--save-every", type=int, default=20)
    r.add_argument("--seed", type=int, default=0)
    r.add_argument("--device", default="cpu")
    r.add_argument("--name", default=None)
    a = ap.parse_args()
    os.chdir(BASE)
    (cmd_bc if a.cmd == "bc" else cmd_rl)(a)


if __name__ == "__main__":
    main()
