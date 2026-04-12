# Figure 2. Neural Network Architecture

> 아래 Mermaid 코드를 [Mermaid Live Editor](https://mermaid.live) 또는 VS Code Mermaid 플러그인으로 렌더링하세요.

## (a) Manager: Graph Transformer with Pointer Network

```mermaid
graph LR
    subgraph Encoder["🔵 Graph Transformer Encoder"]
        A["Node Features<br/>x ∈ ℝ<sup>N×4</sup><br/>[x, y, is_start, is_goal]"]
        B["GATv2Conv<br/>(Topology Encoding)<br/>heads=4, concat=False"]
        C["+ Laplacian PE<br/>(k=8 eigenvectors)<br/>→ Linear(8, H)"]
        D["LayerNorm"]
        E["Transformer Encoder<br/>×3 layers<br/>d_model=H, heads=4<br/>FFN=4H"]
        A --> B --> C --> D --> E
    end

    subgraph Decoder["🟠 Autoregressive Decoder"]
        F["SOS Token<br/>+ Positional Emb"]
        G["Transformer Decoder<br/>×3 layers<br/>Cross-Attention to Memory"]
        H["Pointer Network<br/>Q = Linear(H)<br/>K = Linear(H)<br/>score = v<sup>T</sup>tanh(Q+K)"]
        I["Subgoal Sequence<br/>s₁, s₂, ..., sₖ, EOS"]
        F --> G --> H --> I
    end

    E -- "Memory [B, N+1, H]" --> G
```

## (b) Worker: GATv2-LSTM with Multi-head Scorer

```mermaid
graph LR
    subgraph Spatial["🟢 Spatial Encoder (GATv2 ×3)"]
        A2["Node Features<br/>x ∈ ℝ<sup>N×8</sup><br/>[x, y, is_curr, is_tgt,<br/>dist, dir_x, dir_y,<br/>is_final_phase]"]
        B2["GATv2Conv Layer 1<br/>+ Input Projection<br/>+ Residual + LayerNorm"]
        C2["GATv2Conv Layer 2<br/>+ Residual + LayerNorm"]
        D2["GATv2Conv Layer 3<br/>+ Residual + LayerNorm"]
        A2 --> B2 --> C2 --> D2
    end

    subgraph Temporal["🟣 Temporal Memory"]
        E2["LSTMCell<br/>(H, H)"]
        F2["Current Node<br/>Embedding"]
        D2 -- "h[is_current]" --> F2 --> E2
    end

    subgraph Decision["🔴 Decision Heads"]
        G2["Multi-head Scorer<br/>×4 heads<br/>Linear(2H, H/4) → ReLU → Linear(H/4, 1)<br/>→ mean ensemble"]
        H2["Critic MLP<br/>Linear(H, H) → ReLU → Linear(H, 1)"]
        I2["Next Node Action<br/>a = argmax(scores)"]
        J2["State Value<br/>V(s)"]
        E2 -- "h_next" --> G2 --> I2
        E2 -- "h_next" --> H2 --> J2
    end

    D2 -- "All node emb [B*N, H]" --> G2
```

## (c) Tensor Shape Summary

| Component | Input Shape | Output Shape | Parameters |
|-----------|-------------|--------------|------------|
| Manager GATv2 | [B×N, 4] | [B×N, H] | heads=4, edge_dim=5 |
| Manager LPE | [N, 8] | [N, H] | Linear(8, H) |
| Manager TF Encoder | [B, N, H] | [B, N, H] | 3 layers, 4 heads |
| Manager TF Decoder | [B, L, H] | [B, L, H] | 3 layers, 4 heads |
| Manager Pointer | [B, L, H] × [B, N+1, H] | [B, L, N+1] | v ∈ ℝ^H |
| Worker GATv2 ×3 | [B×N, 8] | [B×N, H] | Residual, heads=4 |
| Worker LSTM | [B, H] | [B, H] | LSTMCell |
| Worker Scorer | [B×N, 2H] | [B×N, 1] | 4-head ensemble |
| Worker Critic | [B, H] | [B, 1] | 2-layer MLP |

*H = hidden_dim (default: 256)*
