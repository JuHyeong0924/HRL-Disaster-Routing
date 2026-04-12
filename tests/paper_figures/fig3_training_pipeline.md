# Figure 3. Training Pipeline

> 아래 Mermaid 코드를 [Mermaid Live Editor](https://mermaid.live) 또는 VS Code Mermaid 플러그인으로 렌더링하세요.

```mermaid
graph TD
    classDef dataNode fill:#dbeafe,stroke:#3b82f6,stroke-width:2px,color:#1e3a5f
    classDef slNode fill:#dcfce7,stroke:#22c55e,stroke-width:2px,color:#14532d
    classDef rlNode fill:#fef3c7,stroke:#f59e0b,stroke-width:2px,color:#78350f
    classDef evalNode fill:#fce7f3,stroke:#ec4899,stroke-width:2px,color:#831843
    classDef noteNode fill:#f1f5f9,stroke:#94a3b8,stroke-width:1px,color:#475569

    A["🗺️ Road Network Graph<br/>G = (V, E)<br/>TNTP Format"]:::dataNode
    B["📐 A* Expert Paths<br/>APSP Matrix<br/>Hop Shortest Path"]:::dataNode

    C["Phase 0: SL Pre-training<br/>━━━━━━━━━━━━━━<br/>• DAgger-style Error Recovery<br/>• Teacher Forcing 1.0→0.0<br/>• Manager CE + Worker CE<br/>• 20 Epochs"]:::slNode

    D["Phase 1: APTE RL<br/>━━━━━━━━━━━━━━<br/>• Worker-only Goal-Conditioned<br/>• Hidden Checkpoint Guidance<br/>• PBRS + Curriculum (0→100%)<br/>• 10,000 Episodes, POMO=32"]:::rlNode

    E["Phase 1: Joint RL<br/>━━━━━━━━━━━━━━<br/>• Manager + Worker End-to-End<br/>• Readiness Gate Check<br/>• Stage-aware PBRS"]:::rlNode

    F["Phase 2: Disaster RL<br/>━━━━━━━━━━━━━━<br/>• Dynamic Edge Destruction<br/>• Seismic Schedule<br/>• Adaptive Re-routing"]:::rlNode

    G{"Readiness<br/>Gate"}:::evalNode

    H["📊 Evaluation<br/>Success Rate, PLR,<br/>Inference Latency"]:::evalNode

    A --> B
    B --> C
    C -- "model_sl_final.pt" --> D
    D -- "best.pt" --> G
    G -- "Pass" --> E
    G -- "Fail: Re-train" --> D
    E -- "best.pt" --> F
    D --> H
    E --> H
    F --> H

    N1["SL Checkpoint:<br/>Manager + Worker"]:::noteNode
    N2["RL Phase1 Checkpoint:<br/>Worker-only (APTE)"]:::noteNode

    C -.- N1
    D -.- N2
```

## Pipeline Summary

| Phase | 목적 | 입력 | 출력 | 핵심 기법 |
|-------|------|------|------|-----------|
| Phase 0 (SL) | 기본 경로 모방 | A* Expert Paths | `model_sl_final.pt` | DAgger, Teacher Forcing |
| Phase 1 (APTE) | Goal-conditioned 강화 | SL Checkpoint | `best.pt` (Worker) | Hidden Checkpoints, PBRS, Curriculum |
| Phase 1 (Joint) | Manager-Worker 조율 | Phase 1 APTE best | `best.pt` (Joint) | Stage-aware PBRS, Readiness Gate |
| Phase 2 (Disaster) | 재난 적응 | Joint best | `rl_finetune_phase2/` | Dynamic Edge, Seismic Schedule |
