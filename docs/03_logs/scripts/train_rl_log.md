# train_rl.py — 변경 로그

## Phase 2E: 커리큘럼 확장 + target_dim 업데이트 (2026-06-24)

### 변경 개요
Manager 생성자 `target_dim=6`, 커리큘럼 4→5 Phases로 확장.

### 커리큘럼 구성
| Phase | 에피소드 범위 | 타겟 수 | disaster_prob | dynamic | 의도 |
|-------|------------|---------|--------------|---------|------|
| P1:Single | 0~5000 | 1 | 0.0 | ✗ | 단일 타겟 기본 학습 |
| P2:Multi | 5001~15000 | 3~7 | 0.0 | ✗ | 다중 타겟 탐색 |
| P3:Static | 15001~25000 | 5~10 | 0.15 | ✗ | HAZUS 정적 재해 |
| P4:Dynamic | 25001~35000 | 5~12 | 0.15 | ✓ | Continuous Aftershock |
| P5:Full | 35001~50000 | 5~15 | 0.2 | ✓ | 전체 범위 강력 재해 |

### 변경 근거
- Phase 3/4 분리: 정적 재해에서 먼저 HAZUS 가중치에 적응한 후 동적 여진 도입
- disaster_prob 0.2→0.15 (P3/P4): 학습 초반 과도한 난이도 방지

### Trial & Error
- **오류 없음**

## Manager 학습 완료 보고 (2026-06-25)
*   **학습 결과**: 50,000 에피소드 전체 커리큘럼 완료.
*   **최종 에피소드 파라미터 (P5: Full)**:
    *   타겟 수: 5~15개 무작위
    *   재난 확률: 0.2
    *   동적 재난: 활성화 (Continuous Aftershock)
*   **최종 학습 지표**:
    *   **전체 평균 속도**: 1.12s/ep (최적화 전 약 10s/ep 대비 8.9배 단축)
    *   **총 소요 시간**: 15시간 32분 (최적화 전 기준 130시간 이상 소요되었을 분량)
    *   **Best Reward**: 182.18
    *   **최종 Reward (Rw)**: 145.14
    *   **구출 성능 (Rsc)**: 평균 8.6개 타겟 구출 (최대 14~15개 기준)
    *   **성공률 (SR)**: **61.2%** (복잡한 동적 재해 및 UGV 파괴 위험 상황에서 매우 안정적인 회복 성능 도달)
    *   **평균 Manager Turn 수**: 26.6 턴
    *   **평균 Worker Step 수**: 64.4 스텝
*   **체크포인트 저장 위치**: `logs/rl_manager_stage/2026-06-24_180420_manager/best_manager.pt`

## Worker Default Hyperparameters 갱신 (2026-06-26)
*   **배경**: Layer-4 Worker 모델이 Layer-2, 3 대비 재계산 횟수(Recomputes)와 지연시간(Latency)에서 압도적인 성능을 보임에 따라 Layer-4를 공식 워커로 채택.
*   **Layer별 성능 비교 (평가 환경 기준)**:
    | Worker 구조 | 구출률 (Rescue Rate) | 지연 시간 (Latency) | 총 이동거리 (Dist) | 경로 재계산 횟수 | 차량 파괴 (Destroys) |
    |---|---|---|---|---|---|
    | **Layer-2** | 76.67 % | 2.017 s | 126.6 | 48.0 회 | 0 대 |
    | **Layer-3** | 77.33 % | 1.960 s | 127.2 | 47.6 회 | 0 대 |
    | **Layer-4** | **79.33 %** | **1.537 s** | **127.0** | **39.6 회** | **0 대** |
*   **변경 사항**: `train_rl.py`의 기본값(Defaults)을 다음과 같이 고정.
    *   `--num_layers`: 4
    *   `--batch_size`: 32
    *   `--mini_batch_size`: 192 (VRAM 24GB OOM 방지)
    *   `--use_gae`: True
    *   `--use_cosine_lr`: True

## Manager OOM 방지 및 하이퍼파라미터 동적 할당 (2026-06-26)
### 1. `batch_size` / `mini_batch_size` 동적 분기
*   **배경**: Worker와 Manager는 구조적으로 요구하는 배치 사이즈가 크게 다름. argparse에서 단계(Stage)별로 분기 처리.
*   **Manager 셋팅**:
    *   `batch_size=256`: Worker 추론이 `no_grad()`로 동결되어 속도가 빠르므로 수집량 대폭 증가.
    *   `mini_batch_size=1024`: PPO 업데이트 단위.

### 2. OOM(Out of Memory) 디버깅 내역
*   **증상**: Manager 커리큘럼이 `P2:Multi`(타겟 수 3~7개)로 진입하면서 `torch.OutOfMemoryError` 발생. (PyTorch 할당 14.8GB + 단편화 7.1GB 낭비).
*   **원인**: Manager의 `loss.backward()` 시 생성되는 GNN 역전파 계산 그래프 크기는 `mini_batch_size × num_targets × K_zones`에 비례함. Phase 1(타겟 1개)에서는 문제가 없었으나, Phase 2에서 7배로 연산 그래프가 폭증하며 VRAM 24GB를 초과함.
*   **해결책**:
    1.  `mini_batch_size`를 2048에서 **1024**로 축소하여 1회 역전파 시 올라가는 텐서 덩어리를 절반으로 감소.
    2.  `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` 환경 변수를 추가하여 PyTorch 내부의 7.1GB 달하는 메모리 단편화(Fragmentation)를 해소, 가용 공간 확충.
*   **결과**: Phase 1~5 전체 커리큘럼(최대 타겟 15개)을 OOM 없이 돌파할 수 있는 안정성 확보.

### 3. tqdm 디스플레이 개선
*   **변경**: `ncols=200`으로 확장.
*   **이유**: 80~140자에서는 성공률(SR)이 Truncate(잘림) 현상으로 보이지 않았으나, 이를 해결하여 실시간 모니터링 편의성 강화.
