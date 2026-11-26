"""PPO 학습 실행과 공통 보상 설정.

하이퍼파라미터는 HPARAMS, 보상 덮어쓰기는 REWARD_OVERRIDES에서 정한다.
도로와 MDP 기본 설정은 environment/env/, PPO 구현은 algorithms/ppo.py에 있다.
실행: python -X utf8 training/train.py
"""
import os
import sys
import shutil
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))   # 저장소 루트
import repo_paths as P  # noqa: E402
from env.road_config import ROAD, EGO, TRAFFIC
from env.mdp_config import SIMULATION, OBSERVATION, ACTION
from env.mdp_config import REWARD as REWARD_DEFAULTS
from env import road_builder
from env.sumo_env import SumoHighwayEnv

from algorithms.ppo import PPO
from utils.buffer import RolloutBuffer
from utils.logger import RunLogger

BASE = P.ROOT   # 저장소 루트 (results/는 루트에)



#  하이퍼파라미터 : 실험은 여기서

TOTAL_TIMESTEPS = 200_000   # 총 학습 스텝 수


#  보상 튜닝 : environment/env/mdp_config.py 의 기본값을 여기서 덮어쓴다.
#  (여기 적은 키만 바뀌고, 나머지는 mdp_config 기본값 그대로)
#  학습·평가·수집은 같은 REWARD를 불러온다.
#  이 파일이 run 폴더에 스냅샷되므로 "그 결과가 어떤 계수였는지"도 남는다.

REWARD_OVERRIDES = dict(
    speed_weight=0.1,          # 속도 보상 (스텝당 최대 이 값)
    collision_penalty=20.0,    # 충돌 시 종료 감점.
    red_light_penalty=20.0,    # 적색 정지선 위반 시 종료 감점.
    stuck_penalty=20.0,        # 비정상 정지 120초 시 종료 감점.
    off_route_penalty=20.0,    # 경로 이탈 시 종료 감점. 잘못된 차로 조기 종료는 현재 꺼져 있다.
    arrival_bonus=2.0,         # 완주 보너스
    close_gap_threshold=6.0,   # 이 거리(m) 미만 접근 시 (close_gap_headway > 0이면 쓰지 않음)
    close_gap_headway=0.0,     # 0이면 고정 거리 close_gap_threshold로 근접을 판정한다.
    close_gap_min_speed=5.0,   # 이 속도(m/s) 이상에서만 근접 감점을 적용한다.
    close_gap_penalty=0.1,     # 근접 조건을 만족하는 스텝의 감점.
    ttc_threshold=2.0,         # 앞차에 접근 중일 때 TTC가 이 시간(s) 미만이면 감점한다.
    ttc_penalty=0.2,           # 저속 대기열 추돌도 억제하기 위한 TTC 감점.
    ttc_max_scale=1.0,         # TTC 감점 배율 상한. 1이면 고정 감점이다.
    red_far_stop_penalty=0.0,  # 적색 앞 먼 정지 감점. 현재 0으로 비활성화.
    red_far_stop_gap=15.0,     # 먼 정지 판정에 쓰는 정지선·앞차와의 거리(m).
    blocked_penalty=0.0,       # 앞차를 밀도록 학습되는 것을 피하기 위해 비활성화.
    blocked_gap=20.0,          #   막힘 판정: 앞차 이 거리 이내
    blocked_speed_frac=0.7,    #   그리고 내 속도 < vmax × 이 값
    lane_change_penalty=0.0,   # 실행된 차로 변경의 감점.
    invalid_action_penalty=0.0,# 실행되지 않은 행동의 감점.
)
REWARD = {**REWARD_DEFAULTS, **REWARD_OVERRIDES}


HPARAMS = dict(
    # ---- 최적화 ----
    lr=1e-4,                 # 학습률.
    n_steps=2048,            # rollout 길이. 크면 advantage 추정이 안정적
    n_epochs=10,             # 모은 데이터 재사용 횟수
    minibatch_size=64,

    # ---- 할인 / advantage ----
    gamma=0.99,              # 할인율 γ. 0.99 ≈ 100스텝(50초) 앞을 내다봄
    gae_lambda=0.95,         # GAE 계수. 클수록 분산이 늘고 편향이 줄어든다.

    # ---- PPO 고유 ----
    clip_eps=0.2,            # 클리핑 범위 ε
    value_coef=0.5,          # 가치 손실 가중치
    entropy_coef=0.0,
    lane_entropy_coef=0.0,   # 차로 선택 엔트로피 보너스. 현재 0으로 비활성화.
    max_grad_norm=0.5,
    normalize_advantage=True,

    # ---- 네트워크 (training/utils/networks.py) ----
    hidden_sizes=(256, 256),
    activation="relu",       # "tanh" | "relu" | "elu"
    policy_type="hybrid",      # hybrid는 연속 가감속과 이산 차로 선택을 결합한다.
    lane_keep_bias=0.5,        # 차로 선택 head에서 유지 행동에 주는 초기 편향.
    init_log_std=-0.5,         # 연속 가감속의 초기 탐험 폭. 표준편차는 exp(init_log_std)다.

    # ---- 기타 ----
    seed=0,
    device="auto",           # "auto" | "cpu" | "cuda"
)

# ---- 평가 / 로깅 ----
EVAL_INTERVAL = 5       # 몇 번의 업데이트마다 주행 지표를 평가할지 (0이면 안 함)
EVAL_EPISODES = 5       # 평가 1회당 에피소드 수
LOG_INTERVAL = 1        # 몇 번의 업데이트마다 콘솔에 출력할지
RESULTS_DIR = "results" # 결과 저장 폴더
RUN_NAME = None         # None이면 run_날짜_시각 자동 생성.
                        # 문자열을 주면 그 이름으로 저장 (예: "exp_lr1e-4")



def make_run_dir() -> str:
    """results/<run이름>/ 폴더를 만들고 경로를 반환."""
    name = RUN_NAME or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    run_dir = os.path.join(BASE, RESULTS_DIR, name)
    os.makedirs(run_dir, exist_ok=True)
    if not os.path.isdir(run_dir):
        raise NotADirectoryError(
            f"결과 폴더를 만들 수 없습니다: {run_dir}\n"
            f"  같은 이름의 '파일'이 이미 있는지 확인하세요.")
    return run_dir


def snapshot_configs(run_dir: str):
    """이 run에 사용된 설정 파일들을 복사해둔다 (재현/비교용).

    학습 스크립트 자신(training/train.py)도 함께 저장한다 : 하이퍼파라미터가
    이 파일 안에 있으므로, 나중에 "이 결과가 어떤 설정이었나"를
    정확히 되짚을 수 있다.
"""
    for src in [os.path.join(P.ENV_PKG, "road_config.py"),
                os.path.join(P.ENV_PKG, "mdp_config.py"),
                os.path.join(P.TRAINING, "train.py")]:
        shutil.copy(src, run_dir)


if __name__ == "__main__":
    # 1) 도로 생성 (environment/env/road_config.py 기준으로 매번 재생성)
    sumocfg = road_builder.build(ROAD, EGO, TRAFFIC,
                                 P.SUMO_DIR)
    print(f"도로 생성 완료: {sumocfg}")
    print(f"  길이 {ROAD['length']}m, {ROAD['num_lanes']}차선, "
          f"교통량 {TRAFFIC['vehs_per_hour']}대/시간")

    # 2) 결과 폴더 + 로거
    run_dir = make_run_dir()
    logger = RunLogger(run_dir)
    print(f"결과 폴더: {run_dir}")

    # 3) 환경
    env = SumoHighwayEnv(
        cfg_path=sumocfg,
        road=ROAD, ego=EGO,
        mdp_sim=SIMULATION, mdp_obs=OBSERVATION,
        action=ACTION, reward=REWARD,
        traffic=TRAFFIC,
        gui=False,   # 학습은 화면 없이 (수십 배 빠름)
    )

    # 4) 경험 버퍼 + 에이전트
    #    버퍼는 메인에서 만들어 에이전트에 "주입"한다 (training/utils/buffer.py).
    #    PPO는 on-policy라 RolloutBuffer를 쓰고, 나중에 SAC/DQN 같은
    #    off-policy 알고리즘을 붙일 땐 여기서 ReplayBuffer로 갈아끼우면 된다.
    buffer = RolloutBuffer()
    agent = PPO(
        obs_dim=env.observation_space.shape[0],
        act_dim=env.action_space.shape[0],
        buffer=buffer,
        **HPARAMS,
    )
    print(f"obs_dim={env.observation_space.shape[0]}, "
          f"act_dim={env.action_space.shape[0]}, device={agent.device}")

    # 5) 학습 후 저장
    #    try/finally: Ctrl+C나 에러로 중단돼도 SUMO 정리 + 결과 저장은 수행
    try:
        agent.learn(env,
                    total_timesteps=TOTAL_TIMESTEPS,
                    logger=logger,
                    log_interval=LOG_INTERVAL,
                    eval_interval=EVAL_INTERVAL,
                    eval_episodes=EVAL_EPISODES)
    finally:
        env.close()
        logger.close()
        model_path = os.path.join(run_dir, "model.pt")
        agent.save(model_path)
        snapshot_configs(run_dir)
        print(f"\n저장 완료: {model_path}")
        print(f"재생 방법:      python test.py "
              f"{os.path.relpath(model_path, BASE)}")
        print(f"학습 곡선 보기: tensorboard --logdir {RESULTS_DIR}"
              f"  → http://localhost:6006")
