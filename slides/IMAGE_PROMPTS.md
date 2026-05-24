# Prompts para imagens da apresentação

Paleta obrigatória em todas as imagens:
- Fundo creme `#f3efe8` ou branco suave
- Teal principal `#1a5f66` (setas, caixas, títulos)
- Verde benigno `#3d7a6b`
- Vermelho-terracota maligno `#b84a3d`
- Âmbar destaque `#c17f3a`
- Texto grafite `#1c2428`
Estilo: infográfico médico-educacional flat, limpo, sem fotorealismo pesado, alta legibilidade para projeção 16:9.

Salvar em `assets/` na raiz do repositório (o HTML referencia `../assets/` a partir de `slides/`).

---

## `mlp_arch.png` — Slide 8 (Arquitetura)

```
Educational flat vector infographic, 16:9, multilayer perceptron for breast cancer tabular data.
Three layers left to right: input layer 30 green-teal nodes labeled "30 features",
hidden layer 16 nodes with label "ReLU · 16", output layer 2 nodes "Softmax · B / M".
Cream background #f3efe8, nodes and arrows in teal #1a5f66, benign class hint in green #3d7a6b,
malignant in terracotta #b84a3d, accent highlights #c17f3a. Clean sans-serif labels,
medical-scientific poster style, no 3D, no stock photo, high contrast for classroom projector.
```

---

## `forward_pass.png` — Slide 9 (Forward)

```
Educational diagram of neural network forward pass, 16:9, cream background #f3efe8.
Show data flow: input vector x → matrix multiply W1 + b1 → ReLU box (only positive pass,
negative blocked in muted gray) → second layer → softmax → output probabilities.
Color code: teal #1a5f66 for active paths, terracotta #b84a3d for output, amber #c17f3a
for math symbols. Small equations z=Wx+b and h=max(0,z). Flat vector style, minimal text,
scientific slide aesthetic, no characters, no photorealism.
```

---

## `softmax_ce.png` — Slide 10 (Softmax + CE)

```
Split infographic 16:9 explaining softmax and cross-entropy for binary classification B vs M.
Left: two logit bars transforming into probability bars summing to 100% via softmax.
Right: cross-entropy loss L = -sum y log y_hat with one-hot y and predicted probabilities.
Cream #f3efe8 background, teal #1a5f66 typography, green #3d7a6b for benign class,
terracotta #b84a3d for malignant, amber #c17f3a for gradient hint "y_hat - y".
Flat medical-education style, clear arrows, no clutter, projector-friendly.
```

---

## `ablation_ce_concept.png` — Slide novo (Por que ΔCE?)

```
Infographic explaining feature ablation importance via cross-entropy increase, 16:9.
Three steps left to right on cream #f3efe8: (1) intact feature vector → confident correct
prediction low loss; (2) same vector with one feature zeroed/masked in terracotta #b84a3d;
(3) higher CE loss bar chart delta. Label "ΔCE = CE_ablated - CE_baseline" in teal #1a5f66.
Show why discrete accuracy fails on small test set (115 samples). Green #3d7a6b for
"important feature", muted gray for unimportant. Flat scientific style, no photos.
```

---

## Placeholders clínicos (slides 4–5)

**`fna_procedure.png`**
```
Medical-education diagram of breast fine needle aspiration (FNA) workflow, 16:9,
cream #f3efe8 background. Steps: breast lesion → thin needle aspiration → slide with
stained cells → digital feature extraction. Teal #1a5f66 outlines, amber #c17f3a
for needle accent, respectful clinical illustration, no gore, flat vector, no photo faces.
```

**`benign_vs_malignant.png`**
```
Side-by-side cytology comparison infographic, 16:9, cream #f3efe8. Left: benign nucleus
regular round shape in green #3d7a6b; right: malignant irregular spiculated nucleus in
terracotta #b84a3d. Labels "Benigno (B)" and "Maligno (M)". Teal #1a5f66 headers,
minimal text, scientific poster flat style, stylized cells not photomicroscopy.
```
