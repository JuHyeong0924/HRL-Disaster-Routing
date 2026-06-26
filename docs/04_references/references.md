# 참고 문헌 (References for Paper)

> 본 프로젝트에서 사용된 핵심 설계 근거의 참고 문헌 목록.
> 논문 작성 시 인용할 수 있도록 분야별로 정리.

---

## 1. 재난 피해 등급 및 도로 네트워크 가중치

### 1.1. HAZUS Earthquake Model (FEMA)

**[R1] FEMA HAZUS Earthquake Model Technical Manual**
- Federal Emergency Management Agency. (2024). *Hazus Earthquake Model Technical Manual*. U.S. Department of Homeland Security.
- URL: https://www.fema.gov/flood-maps/products-tools/hazus
- **인용 내용**: Damage State 정의 (None/Slight/Moderate/Extensive/Complete), Fragility Curve 기반 손상 확률 추정, Transportation Systems Chapter의 Residual Capacity 개념
- **활용**: 간선 가중치 배율 (×1.0/×2.0/×4.0/×20.0) 도출의 근거

### 1.2. Residual Capacity & Network Resilience

**[R2] Kilanitis, I., & Sextos, A. (2019).**
- "Integrated seismic risk and resilience assessment of roadway networks in earthquake prone areas."
- *Bulletin of Earthquake Engineering*, 17, 3009–3025.
- DOI: 10.1007/s10518-019-00609-y
- **인용 내용**: Bridge damage state → residual capacity (100%/50%/25%/0%) 매핑, 교통 네트워크 복원력 평가 프레임워크
- **활용**: HAZUS Damage State를 교통 네트워크 임피던스 팩터로 변환하는 근거

**[R3] Socio-Economic Effect of Seismic Retrofit Implemented on Bridges in the LA Highway Network**
- Shinozuka, M., Zhou, Y., Kim, S.-H., Murachi, Y., Banerjee, S., Cho, S., & Fukutake, H.
- Bureau of Transportation Statistics (BTS), FHWA-HRT-08-050.
- URL: https://rosap.ntl.bts.gov/
- **인용 내용**: Table 5.1 — Link damage state와 capacity reduction 정량화 (risk-tolerant vs risk-averse 정책)
- **활용**: Moderate=50%, Extensive=25%, Complete=0% 잔여 용량 수치의 원출처

**[R4] Dong, Y., Frangopol, D. M., & Saydam, D. (2013).**
- "Time-variant sustainability assessment of seismically vulnerable bridges subjected to multiple hazards."
- *Earthquake Engineering & Structural Dynamics*, 42(10), 1451–1467.
- DOI: 10.1002/eqe.2281
- **인용 내용**: 시간 경과에 따른 교량 잔여 용량 변화 모델, 다중 재해 하에서의 교통 네트워크 취약성

---

## 2. UGV 재난 환경 통행성 (Traversability)

### 2.1. UGV 잔해 도로 통과 능력

**[R5] Jacoff, A., Messina, E., Weiss, B. A., Tadokoro, S., & Nakagawa, Y. (2003).**
- "Test arenas and performance metrics for urban search and rescue robots."
- *Proceedings of the IEEE/RSJ International Conference on Intelligent Robots and Systems (IROS)*, Vol. 3, pp. 3396–3403.
- DOI: 10.1109/IROS.2003.1249681
- **인용 내용**: 재난 현장 잔해(rubble) 위 로봇 이동성 정량화, coverability/crossability 메트릭
- **활용**: UGV가 Complete 손상 도로를 "강행 돌파"할 수 있다는 가정의 근거

**[R6] Delmerico, J., Mueggler, E., Nitsch, J., & Scaramuzza, D. (2019).**
- "Active autonomous aerial exploration for ground robot path planning."
- *IEEE Robotics and Automation Letters*, 4(2), 2184–2191.
- DOI: 10.1109/LRA.2019.2899070
- **인용 내용**: UAV-UGV 협력 시스템에서 UGV의 험지(rough terrain) 통행 비용 모델링
- **활용**: 손상된 도로의 통행 비용을 가중치 배율로 표현하는 접근법의 선례

**[R7] Endo, M., Tanaka, K., & Ohashi, H. (2022).**
- "An Analysis Method on Post-earthquake Traversability of Road Network Considering Building Collapse."
- *International Journal of Engineering (IJE)*, 35(9), 1764–1773.
- DOI: 10.5829/ije.2022.35.09c.12
- **인용 내용**: 건물 붕괴를 고려한 지진 후 도로 네트워크 통행 가능성 분석
- **활용**: 재난 후 도로 통행성이 확률적(probabilistic)이라는 모델링 근거

---

## 3. 지진 여진 모델링 (Aftershock Modeling)

**[R8] Omori, F. (1894).**
- "On the aftershocks of earthquakes."
- *Journal of the College of Science, Imperial University of Tokyo*, 7, 111–200.
- **인용 내용**: Omori's Law — 여진 빈도가 시간에 따라 감소하는 법칙 (n(t) = K/(c+t)^p)
- **활용**: 시간 기반 여진 스케줄링의 물리적 근거 (시간 축에서 여진이 독립적으로 발생)

**[R9] Iervolino, I., Giorgio, M., & Polidoro, B. (2014).**
- "Sequence-based probabilistic seismic hazard analysis."
- *Bulletin of the Seismological Society of America*, 104(2), 1006–1012.
- DOI: 10.1785/0120130207
- **인용 내용**: 여진 시퀀스의 확률적 모델링, 주진 이후 시간 기반 여진 발생 확률
- **활용**: 에피소드 내 시간 기반 Continuous Aftershock 메커니즘의 학술적 근거

---

## 4. RL 기반 경로 계획 / 재난 대응

**[R10] Arzani, M., Chen, Y., & Sabar, N. R. (2023).**
- "Reinforcement learning for vehicle routing in disaster scenarios: A survey."
- **인용 내용**: RL 기반 차량 경로 최적화의 재난 대응 적용, 동적 환경에서의 RL 장점 (실시간 적응, 재연산 불필요)
- **활용**: Neural Manager/Worker의 실시간 적응력이 메타 휴리스틱(GA/ALNS) 대비 유리한 이유의 문헌적 근거

---

## 5. 본 프로젝트 활용 매핑 요약

| 설계 요소 | 참고 문헌 | 인용 근거 |
|----------|----------|----------|
| Damage State 4등급 분류 | [R1] FEMA HAZUS | Slight/Moderate/Extensive/Complete |
| Residual Capacity 수치 | [R2] Kilanitis & Sextos, [R3] Shinozuka et al. | 100%/50%/25%/0% |
| Weight Multiplier 공식 | [R2], [R3] | `1 / Residual Capacity` |
| UGV Complete 도로 강행 돌파 | [R5] Jacoff et al., [R7] Endo et al. | 특수 장비의 험지 통행 가능성 |
| Soft Closure (간선 비제거) | [R2], [R6] | 네트워크 연결성 보존 + 비용 모델링 |
| 시간 기반 여진 스케줄 | [R8] Omori, [R9] Iervolino et al. | 여진은 시간 축에서 독립 발생 |
| 데미지 누적 모델 | [R1] FEMA HAZUS | Cumulative damage ratio (0~1) |
| RL vs 휴리스틱 동적 환경 이점 | [R10] Arzani et al. | 실시간 적응 vs 재연산 비용 |
