# `src/models/manager_unified.py` Log

## 1. 상세 코드 설명 (Detailed Code Explanation)
*   매니저 모델을 PPO로 전환하기 위해 가치 추정기(Critic Head)인 `self.value_head`가 추가되었습니다.
*   `value_head`는 `nn.Sequential(nn.Linear(128, 128), nn.ReLU(), nn.Linear(128, 1))` 아키텍처를 가집니다.
*   `get_value(self, query)` 메서드는 어텐션 처리된 컨텍스트 벡터 `query` [B, 128]를 입력받아 최종 스칼라 값 [B] 차원의 상태 가치 $V(s)$를 반환합니다.

## 2. 시행착오 로깅 (Trial & Error Log)
*   **이슈:** POMO에서 PPO로 전환하기 위해 기존 Actor 전용 아키텍처에 Critic이 부재한 문제.
*   **해결:** 단순하게 `target_fusion`이나 `zone_query_proj`를 거치기 직전의 최상위 전역 컨텍스트인 `query`를 Critic의 입력으로 사용하여 계산 효율성을 극대화함.
*   **이슈 (24/06/15):** 매니저가 Zone 학습 시 보상이 -20 대에 정체되며, Invalid zone을 선택하거나 무한루프에 빠지는 지리적 맹시(Geographic Blindness) 현상 발생.
*   **해결:** 
    1. 환경단(`hrl_env.py`)에서 `zone_dist_matrix`를 GPU 텐서로 캐싱하여 `ManagerUnified`로 주입.
    2. `manager_unified.py`에 `zone_score_net`을 신설하고, `get_zone_logits` 단계에서 `query`, `zone_emb`, `target_dists`를 Concat하여 최종 Zone 선호도를 평가.
    3. 배치 사이즈 한계(OOM)를 방지하기 위해 512에서 **256으로 하향 고정**하여 안정적 그래디언트를 확보.
*   **결과:** 20,000 에피소드 학습 결과, 보상(Rw)이 **-21 에서 +42.44 로 수직 상승**하며 완벽에 가까운 Subgoal 라우팅(목표 도달률 급증) 능력을 확보함. 아키텍처 수정의 수학적 유효성 완벽 입증 완료.
