"""저장소 경로 — 모든 실행 스크립트가 맨 위에서 불러 쓴다.

폴더 구조 (Environment / Training / Evaluation / Data Collection):
    environment/env/     수업 Gym 환경 + 반포 도로망 연결 (패키지 `env`)
    environment/road/    도로 정의 · 생성 · 검증 (road_spec · build_net · routing · check_* …)
    training/            train_team.py · train.py + 패키지 `algorithms` · `utils`
    evaluation/          metrics · lanes · baseline (평가 · 기준 지표)
    data_collection/     collect_pomdp_data.py
    data/ · results/     수집 데이터 · 학습 모델 (저장소 루트)

이 모듈을 불러오면 위 폴더들이 모듈 검색 경로에 들어가, 어디서 실행하든
`from env import ...`, `import road_spec`, `from algorithms.ppo import PPO`, `import train`,
`import metrics`, `import collect_pomdp_data`가 그대로 동작한다.

실행 스크립트 첫머리 (저장소 루트까지의 깊이만 다름):
    import os, sys
    sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
    import repo_paths as P
"""
import os
import sys

ROOT = os.path.dirname(os.path.abspath(__file__))

ENVIRONMENT = os.path.join(ROOT, "environment")
ENV_PKG = os.path.join(ENVIRONMENT, "env")            # 패키지 env
SUMO_DIR = os.path.join(ENV_PKG, "sumo")              # road_builder.build()가 도로망 · 수요 파일을 만드는 곳
ROAD_DIR = os.path.join(ENVIRONMENT, "road")
ROAD_BUILD = os.path.join(ROAD_DIR, "build")          # 검증 출력 (git 제외)
VIDEOS = os.path.join(ROAD_DIR, "videos")
TRAINING = os.path.join(ROOT, "training")
EVALUATION = os.path.join(ROOT, "evaluation")
DATA_COLLECTION = os.path.join(ROOT, "data_collection")
DATA = os.path.join(ROOT, "data")
RESULTS = os.path.join(ROOT, "results")
EVIDENCE = os.path.join(ROOT, "environment", "road", "measured")

for _d in (DATA_COLLECTION, EVALUATION, TRAINING, ROAD_DIR, ENVIRONMENT, ROOT):
    if _d not in sys.path:
        sys.path.insert(0, _d)
