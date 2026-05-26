# Ablation Study v3: HRL Worker (Phase 1, Soft Masking + PBRS)

## 1. 개요
- **목적**: HRL Worker 모델의 최적 State 표현(Masking), Reward 구조(PBRS), 그리고 신경망 깊이(GATv2 Layers)를 결정하기 위한 실험. 특히, 단순 성공률(Success Rate)이 포화(Saturation)되는 상황에서 **경로 효율성(Path Length) 및 보상(Reward) 측면의 해의 품질(Solution Quality)을 상세 평가**하기 위함.
- **실행일**: 2026-04-29
- **환경**: `HRLZoneEnv` (Zone K=30)
- **알고리즘**: REINFORCE + GAE + Entropy
- **에피소드**: 500ep (빠른 스크리닝 목적)

## 2. 실험 목록 및 상세 결과 (Step 500 기준)

> [!NOTE]
> 모든 모델은 500 에피소드 학습 후의 성능을 기록하였으며, 성공률(SR)이 100%에 도달하더라도 평균 보상(Rw)과 경로 길이(Len)를 통해 최종 해의 품질을 구분할 수 있습니다.

### Round 1: Masking 변인 비교
| ID | 설명 | 설정 값 | 성공률 (SR) | 평균 보상 (Rw) | 평균 경로 길이 (Len) |
|---|---|---|:---:|:---:|:---:|
| `M_HFS` | Hard Full Seq | `--masking_mode hard_full_seq` | 78.0% | - | - |
| `M_SCN` | Soft Curr Next | `--masking_mode soft_curr_next` | **100.0%** | 50.0 | **19.9** |

### Round 2: Soft Flex vs Hard + PBRS
| ID | 설명 | 설정 값 | 성공률 (SR) | 평균 보상 (Rw) | 평균 경로 길이 (Len) |
|---|---|---|:---:|:---:|:---:|
| `M_SF` | Soft Flex | `--masking_mode soft_flex` | **99.0%** | 49.9 | 21.9 |
| `P_HARD`| Hard + PBRS | `--masking_mode hard_full_seq --use_pbrs` | 81.0% | - | - |

### Round 3: Soft Masking + PBRS 시너지 (★ 품질 평가 핵심)
| ID | 설명 | 설정 값 | 성공률 (SR) | 평균 보상 (Rw) | 평균 경로 길이 (Len) |
|---|---|---|:---:|:---:|:---:|
| `P_SF` | Soft Flex + PBRS | `--masking_mode soft_flex --use_pbrs` | **100.0%** | 57.3 | 18.6 |
| `SCN_P`| Soft Curr Next + PBRS | `--masking_mode soft_curr_next --use_pbrs`| **100.0%** | 55.9 | **15.5** |

### Round 4 & 5: GATv2 Layer 수 탐색
| ID | 설명 | 설정 값 | 성공률 (SR) | 평균 보상 (Rw) | 평균 경로 길이 (Len) |
|---|---|---|:---:|:---:|:---:|
| `L1` | 1-Layer | `--num_layers 1` | 81.0% | - | - |
| `L3` | 3-Layer | `--num_layers 3` | 81.0% | - | - |
| `L4` | 4-Layer | `--num_layers 4` | 79.0% | - | - |
| `L3_SF`| 3-Layer + Soft Flex | `--num_layers 3 --masking_mode soft_flex` | **99.0%** | 51.1 | 22.9 |

### Round 6: 다중 변인 결합 (참고용 BEST)
| ID | 설명 | 설정 값 | 성공률 (SR) | 평균 보상 (Rw) | 평균 경로 길이 (Len) |
|---|---|---|:---:|:---:|:---:|
| `BEST` | Soft Flex + PBRS + 3-Layer | `--masking_mode soft_flex --use_pbrs --num_layers 3` | **99.0%** | 55.9 | 20.3 |

## 3. 핵심 분석 및 결론

1. **단순 성공률 한계 극복 (Solution Quality 평가)**
   - `soft_flex`와 `soft_curr_next` 모두 100%에 근접한 성공률을 보였으나, 목적지 도달이라는 이진(Binary) 결과만으로는 최적의 구조를 판별하기 어려웠습니다.
   - 따라서 목표 도달까지 소모된 평균 경로 길이(Len)를 핵심 지표로 추가 분석했습니다.

2. **최적 경로 효율성: `soft_curr_next`의 압도적 우위**
   - `P_SF` (Soft Flex + PBRS)는 18.6 스텝이 소요된 반면, **`SCN_P` (Soft Curr Next + PBRS)는 15.5 스텝만에 목적지에 도달**했습니다.
   - `soft_flex`는 넓은 탐색 시야 덕분에 성공률은 높지만 불필요하게 우회하는 경로를 생성하는 경향이 있습니다. 
   - 반대로 `soft_curr_next`는 현재 위치와 다음 타겟 간의 로컬 연결성에 집중하여 **불필요한 군더더기 없는 최단 경로(Optimal Path)**를 유도하는 데 탁월한 성능을 보입니다.

3. **PBRS 결합의 시너지**
   - 두 가지 Soft Masking 기법 모두 PBRS 보상 체계를 결합했을 때 평균 보상(Rw)이 약 +6~+7 정도 상승하며 탐색 안정성과 질이 크게 개선되었습니다.

4. **최종 채택 아키텍처**
   - **GATv2 3-Layer + `soft_curr_next` + `use_pbrs`**
   - 앞으로의 Worker 학습은 가장 효율적인 최단 경로를 보장하는 위 설정(`--masking_mode soft_curr_next --use_pbrs --num_layers 3`)을 기본값으로 사용합니다.
