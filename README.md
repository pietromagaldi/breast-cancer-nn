# Breast Cancer MLP — TP3 MC906

Rede neural **100% NumPy** (forward, backprop, mini-batch SGD) para classificar tumores do dataset **Breast Cancer Wisconsin (Diagnostic)**. Entrega principal: `breast-cancer-nn.ipynb`. Enunciado: `2026s1_IA_T3.md`.

---

## Dataset

| Item | Detalhe |
|------|---------|
| Fonte | `sklearn.datasets.load_breast_cancer` (569 amostras, 30 features numéricas) |
| Labels | B=0 (benigno), M=1 (maligno) — invertemos o padrão sklearn |
| Balanceamento | ~357 benignos (62,7%) / ~212 malignos (37,3%) |
| Features | Medidas de núcleo celular: `mean_*`, `worst_*`, `*_error` (raio, textura, perímetro, etc.) |
| Pré-processamento | Split estratificado **64% treino / 16% val / 20% teste** (363 / 91 / 115); z-score por feature (μ, σ só do treino) |

---

## Experimento (fluxo do notebook)

| Seção | O que faz | Hiperparâmetros / notas |
|-------|-----------|-------------------------|
| **3 — Baseline** | Treino inicial + curvas | `arch=[30,32,2]`, `lr=0.1`, `batch=32`, 150 épocas, 3 seeds |
| **4 — Exploração** | Varia um hiperparâmetro por vez (100 épocas); seleção por **val acc** | lr ∈ {0.001, 0.01, 0.1}, batch ∈ {8, 32, 64}, arch ∈ {[30,16,2], [30,32,2], [30,64,32,2]} |
| **5 — Teste final** | Early stopping na val → retreino em treino+val pelo nº ótimo de épocas → **uma** avaliação no teste | Melhor config típica: **`arch=[30,16,2]`, `lr=0.01`, `batch=8`** → ~95–97% test acc |
| **6 — Interpretabilidade** | Gradientes globais, ablação (CE), waterfalls individuais | Usa `final_model` |
| **7 — He + Momentum** | 4 setups comparados; **`batch=32` só para visualização** (modelo final mantém batch=8) | Base, +He, +Momentum, +He+Momentum |
| **8 — sklearn** | Tabela NumPy vs `MLPClassifier` | Mesma arch/épocas do final |
| **9 — SHAP** | Summary plot + waterfalls nos mesmos pacientes da 6.3 | `figures/shap_summary.png` |

**Metodologia importante:** hiperparâmetros escolhidos na **validação**; teste usado uma vez no final (Seção 5).

---

## Como rodar

```bash
cd breast-cancer-nn
python -m venv .venv && source .venv/bin/activate
pip install numpy pandas matplotlib scikit-learn shap jupyter
jupyter notebook breast-cancer-nn.ipynb
```

Run All regenera outputs e figuras em `figures/`. A Seção 4 leva alguns minutos.

Documentação de contexto: `handoff.md`, `roadmap.md`.

---

## Catálogo de figuras (`figures/`)

Tabela para redação do relatório: **onde colocar**, **o que descrever**, **números visíveis**.

| Arquivo | Seção notebook | Sugestão no relatório | O que a figura mostra | Pontos para comentar |
|---------|----------------|----------------------|------------------------|----------------------|
| [`baseline_training_history.png`](figures/baseline_training_history.png) | 3 | **Experimentos** ou **Resultados** (modelo baseline) | Dois painéis (150 épocas, média de 3 seeds ± sombra): **loss** treino (vermelho) vs val (azul tracejado); **acurácia** treino vs val (teal). Marcador amarelo = melhor val acc (época ~12). | Val loss sobe após ~época 15 (overfitting leve); treino continua caindo. Val acc estabiliza ~0,97–0,98. Baseline já converge rápido com lr=0,1. |
| [`pca_2d_projection_baseline.png`](figures/pca_2d_projection_baseline.png) | 3 | **Dados** ou **Resultados** | Dois scatter plots PCA 2D: **esquerda** = rótulos reais (teal B, vermelho M); **direita** = P(maligno) do baseline (colormap azul→vermelho). | Classes separáveis linearmente em 2D; modelo baseline alinha probabilidade com os clusters. Fronteira de decisão visível na região de overlap (PC1 ≈ 0–2). |
| [`hyperparam_heatmap.png`](figures/hyperparam_heatmap.png) | 4 | **Experimentos** (busca de hiperparâmetros) | Heatmap **val acc média (5 seeds)** para grid lr × batch. | Células escuras ≈ >0,95. **lr=0,001** com batch 32/64 fica ~**0,63** (underfitting — platô majoritário). Melhores: lr=0,01 batch=8 (**0,978**), entre outros ≥0,965. |
| [`architecture_bar_chart.png`](figures/architecture_bar_chart.png) | 4 | **Experimentos** | Barras agrupadas train (vermelho) vs val (teal) para 3 arquiteturas, com SEM (5 seeds). | Val **0,978** igual nas três; treino sobe levemente com rede maior (0,992→0,995). Rede rasa `[30,16,2]` basta — escolhida por parsimônia. |
| [`hyperparam_training_curves.png`](figures/hyperparam_training_curves.png) | 4 | **Experimentos** ou **Discussão** (fase de latência) | Três painéis: curvas **treino (:)** vs **val (-)** ao longo de 100 épocas para lr, batch (lr=0,01) e arquitetura. | Todas começam ~**0,63** e ficam planas antes de subir (**symmetry-breaking**). Latência: lr=0,1 ~2 ep; lr=0,01 ~18 ep; lr=0,001 nunca sai; batch 8 mais rápido que 64; rede profunda `[30,64,32,2]` ~45 ep. |
| [`stage1_early_stopping.png`](figures/stage1_early_stopping.png) | 5 | **Metodologia** + **Resultados** | Dois painéis (até 200 épocas): loss e acurácia **treino vs validação**; linha vertical = época de **early stopping** (mínimo val loss). | Val loss sobe após o ponto ótimo (overfitting); treino continua melhorando. Época ótima típica ~**37** com config final. Justifica retreino em treino+val por exatamente esse número de épocas. |
| [`stage2_final_training.png`](figures/stage2_final_training.png) | 5 | **Resultados** | Dois painéis: retreino em **treino+val** por `best_epoch` épocas; curva de **teste** acompanha o histórico (sem val separada). | Test acc sobe até ~**0,97**; test loss cai junto com train loss. Modelo final usado na matriz de confusão e interpretabilidade. |
| [`final_test_confusion_matrix.png`](figures/final_test_confusion_matrix.png) | 5 | **Resultados** | Heatmap 2×2 no **test set** (115 amostras): contagem true vs predicted (B/M), colormap azul, anotações nas células. | Diagonal forte (70 TN benigno, 42 TP maligno). **2 FP** (benigno→maligno), **1 FN** (maligno→benigno). Test acc **97,4%**. Complementa a T5. |
| [`gradient_importance_top10.png`](figures/gradient_importance_top10.png) | 6.1 | **Interpretabilidade** | Bar chart horizontal: top-10 features por média de **\|gradiente de entrada\|** w.r.t. classe predita (test set). | Destaque para `worst_*` e concavidade (`worst concave points`, `worst concavity`, `mean concave points`). Medida local por amostra, agregada globalmente. |
| [`ablation_feature_ce.png`](figures/ablation_feature_ce.png) | 6.2 | **Interpretabilidade** | Top-10 features por **aumento de cross-entropy** ao zerar cada feature (média padronizada). | `worst radius`, `worst concave points`, `worst texture` no topo. CE contínuo captura mudança de confiança mesmo quando acurácia discreta não muda. |
| [`ablation_family_ce.png`](figures/ablation_family_ce.png) | 6.2 | **Interpretabilidade** | Barras: ablação simultânea de famílias `mean_*`, `worst_*`, `*_error`. | Família **`worst_*`** costuma ter maior ΔCE — medidas extremas mais discriminativas que médias ou erros. |
| [`ablation_attribute_ce.png`](figures/ablation_attribute_ce.png) | 6.2 | **Interpretabilidade** | Barras horizontais: ablação dos 3 descritores por **atributo-base** (mean + error + worst). | **Raio** e **concave points** lideram; textura/smoothness menores. Concorda parcialmente com gradientes e SHAP. |
| [`optimizer_training_curves.png`](figures/optimizer_training_curves.png) | 7 | **Extras / Discussão** (He, momentum) | Uma curva: 4 configs (Base, +He, +Momentum, +He+Momentum), arch/lr ótimos, **batch=32** (visualização). Época 0 incluída; linha tracejada = majority baseline ~0,626. | **batch=32** de propósito (legenda explica). Base demora ~18 ep no platô ~0,63; He/Momentum aceleram (90% val em ~10 / ~5 / ~2 ep). Final ~0,96–0,98. *Não usar batch=32 como config final.* |
| [`shap_summary.png`](figures/shap_summary.png) | 9 | **Interpretabilidade** (complemento ao gradiente) | SHAP beeswarm: impacto de cada feature em P(maligno); cor = valor da feature (baixo azul, alto vermelho). | Top: `worst radius`, `worst concave points`, `worst texture`, `worst perimeter`. Alto valor → SHAP positivo (empurra maligno). Concorda em parte com gradientes/ablação (`worst_*`, concavidade). |
| [`waterfall_patient_0.png`](figures/waterfall_patient_0.png) | 6.3 | **Interpretabilidade** (caso individual) | Waterfall **gradiente×input** — Paciente **#0**: verdade **benigno**, P(maligno)=**0,396** (acerto). | Barras vermelhas empurram para maligno; **`worst smoothness`** (teal, −0,466) puxa fortemente para benigno. Exemplo de decisão correta com features conflitantes. |
| [`waterfall_patient_1.png`](figures/waterfall_patient_1.png) | 6.3 | **Interpretabilidade** | Paciente **#1**: benigno, P(maligno)=**0,000** (acerto confiante). | Quase todas as barras teal; destaque **`worst smoothness`** (−0,155), **`worst texture`** (−0,066). Caso “fácil”. |
| [`waterfall_patient_2.png`](figures/waterfall_patient_2.png) | 6.3 | **Interpretabilidade** | Paciente **#2**: maligno, P(maligno)=**0,992** (acerto). | Barras vermelhas dominantes; top **`worst texture`** (+0,090), **`worst radius`** (+0,058). Evidência coerente para maligno. |
| [`waterfall_patient_39.png`](figures/waterfall_patient_39.png) | 6.3 | **Interpretabilidade** ou **Discussão** (erro) | Paciente **#39**: verdade **benigno**, P(maligno)=**0,602** (**falso positivo**). | `mean radius`, `worst symmetry` puxam para maligno; `worst radius`, `worst concavity` puxam para benigno — sinal misto levou a erro. Útil para limitações do modelo. |

> **Nota:** A Seção 6.3 gera waterfalls para os índices em `examples` (1 benigno correto, 1 maligno correto, 1 erro). Os arquivos acima correspondem à última execução; índices podem variar se o test set mudar.

---

## Figuras só no notebook (não exportadas em `figures/`)

| Conteúdo | Seção | Sugestão no relatório |
|----------|-------|----------------------|
| SHAP scatter (top-5) e waterfall interativos | 9 | **Interpretabilidade** — opcional se couber |

---

## Mapa rápido: seção do relatório → figuras

```
Introdução / Dados     → (texto) + opcional pca_2d_projection_baseline
Implementação          → (equações) — sem figura obrigatória
Metodologia            → stage1_early_stopping (early stopping na val)
Experimentos           → hyperparam_heatmap, architecture_bar_chart, hyperparam_training_curves
Resultados             → baseline_training_history, stage2_final_training, final_test_confusion_matrix
Interpretabilidade     → gradient_importance_top10, ablation_*_ce, shap_summary, 1–2 waterfalls
Extras / Discussão     → optimizer_training_curves; waterfall_patient_39 (erro); fase de latência
```

---

## Resultados de referência (última execução típica)

- Melhor config (val): `arch=[30,16,2]`, `lr=0.01`, `batch=8`
- Test acc final: **97,4%** (115 amostras)
- Majority-class baseline (val): **62,6%**

Valores exatos dependem da execução; a seção abaixo reflete a **última execução salva** no notebook.

---

## Tabelas do notebook (última execução)

Referência para redação do relatório — **não é necessário abrir o `.ipynb`**.

### T0 — Dataset e split

| Item | Valor |
|------|-------|
| Amostras totais | 569 |
| Features | 30 |
| Classe B (benigno) | 357 (62,7%) |
| Classe M (maligno) | 212 (37,3%) |
| Treino | 363 |
| Validação | 91 |
| Teste | 115 |

---

### T1 — Exploração: learning rate (Seção 4)

`arch=[30,32,2]`, `batch=32`, 100 épocas. Seleção por **final_val_acc**.

| value (lr) | final_train_acc | final_val_acc | final_loss |
|------------|-----------------|---------------|------------|
| 0.001 | 0.6281 | 0.6264 | 0.6581 |
| **0.01** | 0.9862 | **0.9780** | 0.0644 |
| 0.1 | 1.0000 | 0.9670 | 0.0085 |

→ `best_lr = 0.01`

---

### T2 — Exploração: batch size (Seção 4)

`arch=[30,32,2]`, `lr=0.01`, 100 épocas.

| value (batch) | final_train_acc | final_val_acc | final_loss |
|---------------|-----------------|---------------|------------|
| **8** | 0.9917 | **0.978** | 0.0242 |
| 32 | 0.9862 | 0.978 | 0.0644 |
| 64 | 0.9642 | 0.967 | 0.1413 |

→ `best_batch = 8` (empate 0.978 com batch 32 → mantém o mais simples)

---

### T3 — Exploração: arquitetura (Seção 4)

`lr=0.01`, `batch=8`, 100 épocas.

| value (arch) | final_train_acc | final_val_acc | final_loss |
|--------------|-----------------|---------------|------------|
| **[30, 16, 2]** | 0.9917 | **0.978** | 0.0253 |
| [30, 32, 2] | 0.9917 | 0.978 | 0.0242 |
| [30, 64, 32, 2] | 0.9945 | 0.978 | 0.0271 |

→ `best_arch = [30, 16, 2]` (empate val → mais simples)

**Melhor config:** `arch=[30,16,2]`, `lr=0.01`, `batch=8`

---

### T4 — Heatmap lr × batch (Seção 4, média de 5 seeds)

Val accuracy média; base `arch=[30,32,2]`, 100 épocas. Alimenta `hyperparam_heatmap.png`.

| lr \\ batch | 8 | 32 | 64 |
|-------------|-----|------|------|
| 0.001 | 0.936 | 0.631 | 0.626 |
| 0.01 | **0.978** | **0.978** | 0.965 |
| 0.1 | 0.965 | 0.976 | **0.978** |

---

### T5 — Teste final (Seção 5)

Early stopping na val → **37 épocas** → retreino em treino+val (454 amostras) → avaliação única no teste.

| Métrica | Valor |
|---------|-------|
| Época ótima (Stage 1) | 37 |
| Val loss mínima (Stage 1) | 0.1083 |
| Final train loss (merged) | 0.0586 |
| Final test loss | 0.0845 |
| **Final test accuracy** | **0.974** |

**Matriz de confusão (teste):**

|  | pred B | pred M |
|--|--------|--------|
| **true B** | 70 | 2 |
| **true M** | 1 | 42 |

→ 2 falsos positivos, 1 falso negativo. Figura: `figures/final_test_confusion_matrix.png`.

---

### T6 — Top-10 importância global por gradiente (Seção 6.1)

Média de |gradiente de entrada| w.r.t. classe predita, no test set.

| Rank | Feature | Mean \|grad\| |
|------|---------|---------------|
| 1 | worst radius | 0.027984 |
| 2 | worst texture | 0.027513 |
| 3 | worst perimeter | 0.025526 |
| 4 | worst concave points | 0.025212 |
| 5 | worst area | 0.024782 |
| 6 | worst smoothness | 0.024470 |
| 7 | mean concave points | 0.022515 |
| 8 | mean radius | 0.021586 |
| 9 | mean perimeter | 0.021515 |
| 10 | mean area | 0.020337 |

---

### T7 — Ablação por feature (Seção 6.2, top-10 por CE increase)

Métrica principal: `ce_increase = ablated_ce - baseline_ce`. Accuracy drop discreto em passos de 1/115 ≈ 0.0087.

**Drops únicos de acurácia (feature isolada):** `[-0.0087, 0.0, 0.0087]`

| feature | accuracy_drop | ce_increase | true_conf_drop |
|---------|---------------|-------------|----------------|
| worst texture | 0.0087 | **0.0121** | 0.0082 |
| worst symmetry | 0.0087 | 0.0092 | 0.0056 |
| worst concavity | 0.0000 | 0.0063 | 0.0055 |
| worst concave points | 0.0087 | 0.0055 | 0.0037 |
| worst area | 0.0000 | 0.0043 | 0.0038 |
| radius error | 0.0087 | 0.0039 | 0.0029 |
| worst radius | -0.0087 | 0.0036 | 0.0038 |
| compactness error | 0.0000 | 0.0036 | 0.0001 |
| fractal dimension error | 0.0000 | 0.0032 | -0.0002 |
| area error | 0.0000 | 0.0028 | 0.0021 |

**Overlap top gradientes ∩ top ablação (CE):** `worst area`, `worst concave points`, `worst radius`, `worst texture`

---

### T8 — Ablação por família de medida (Seção 6.2)

| group | n_features | accuracy_drop | ce_increase | true_conf_drop |
|-------|------------|---------------|-------------|----------------|
| **worst_*** | 10 | 0.0348 | **0.1065** | 0.0753 |
| mean_* | 10 | 0.0174 | 0.0138 | 0.0203 |
| *_error | 10 | 0.0000 | 0.0111 | 0.0054 |

---

### T9 — Ablação por atributo-base (Seção 6.2)

Remove `mean_*`, `*_error` e `worst_*` de cada atributo juntos.

| attribute | n_features | accuracy_drop | ce_increase | true_conf_drop |
|-----------|------------|---------------|-------------|----------------|
| texture | 3 | 0.0174 | **0.0298** | 0.0179 |
| symmetry | 3 | 0.0087 | 0.0097 | 0.0055 |
| area | 3 | 0.0000 | 0.0084 | 0.0086 |
| concavity | 3 | 0.0087 | 0.0084 | 0.0086 |
| radius | 3 | 0.0000 | 0.0078 | 0.0089 |
| concave points | 3 | 0.0087 | 0.0075 | 0.0075 |
| fractal dimension | 3 | -0.0087 | 0.0049 | 0.0009 |
| compactness | 3 | 0.0000 | 0.0026 | 0.0009 |
| perimeter | 3 | 0.0000 | 0.0021 | 0.0068 |
| smoothness | 3 | 0.0087 | -0.0020 | 0.0047 |

---

### T10 — Casos individuais (Seção 6.3)

Índices no **test set** (`examples = [0, 2, 39]` na última execução).

| Patient | Verdade | Predição | P(maligno) | Tipo |
|---------|---------|----------|------------|------|
| #0 | benign (B) | benign (B) | 0.604 | acerto |
| #2 | malignant (M) | malignant (M) | 0.992 | acerto |
| #39 | benign (B) | malignant (M) | 0.602 | **erro (FP)** |

Top contribuidores (gradiente × input) — resumo:

- **#0:** `worst radius`, `worst texture`, `worst perimeter` empurram para benigno; acerto apesar de sinais mistos.
- **#2:** `worst texture`, `worst radius`, `worst concave points` empurram para maligno; caso confiante.
- **#39:** `worst texture`/`worst radius` conflitantes → falso positivo.

Waterfalls exportados: `waterfall_patient_0.png`, `waterfall_patient_2.png`, `waterfall_patient_39.png` (+ `waterfall_patient_1.png` se existir de execução anterior).

---

### T11 — He initialization & Momentum (Seção 7)

`arch=[30,16,2]`, `lr=0.01`, **`batch=32`** (só visualização). Majority-class baseline (val): **0.626**.

| Approach | Epoch 0 val acc | Epoch 0 pred M/B | Final val acc | Epoch to 90% val |
|----------|-----------------|------------------|---------------|------------------|
| Base | 0.231 | 78/13 | 0.978 | 27 |
| Base + He | 0.231 | 78/13 | 0.967 | 10 |
| Base + Momentum | 0.231 | 78/13 | 0.978 | 5 |
| Base + He + Momentum | 0.231 | 78/13 | 0.956 | 2 |

---

### T12 — NumPy vs sklearn (Seção 8)

Mesma arch oculta `[16]`, 37 épocas, treino em treino+val.

| Model | Hidden layers | Epochs | Train acc | Test acc |
|-------|---------------|--------|-----------|----------|
| NumPy MLP (manual) | [16] | 37 | 0.987 | **0.974** |
| sklearn MLPClassifier | (16,) | 37 | 0.989 | **0.974** |

→ Implementação manual reproduz acurácia idêntica ao sklearn nesta config.
