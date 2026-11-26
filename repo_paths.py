"""실행 스크립트가 공유하는 저장소 경로와 모듈 검색 경로.

각 실행 파일은 저장소 루트를 먼저 sys.path에 넣은 뒤 이 모듈을 불러온다.
환경·학습 하위 모듈을 기존 import 이름으로 사용할 수 있도록 경로를 등록한다.
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
