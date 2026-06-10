# Exportar apresentação em PDF

## Passo a passo (Chrome ou Edge)

1. Abra `slides/index.html` no **Chrome** ou **Edge** (a partir da pasta do projeto, para carregar `../figures/`).
2. Pressione **P** na apresentação (ou `Cmd+P` / `Ctrl+P`).
3. Configurações na janela de impressão:
   - **Destino:** Salvar como PDF
   - **Layout:** Paisagem
   - **Papel:** A4
   - **Margens:** Padrão (8 mm — o CSS já está calibrado para isso)
   - **Escala:** **100%** (não use “Ajustar à área imprimível”)
   - **Gráficos de segundo plano:** **ativado** (cores teal, barras, imagens)
4. Salvar.

O PDF deve ter **exatamente 28 páginas** (1 slide por página).

## Se ainda sair com mais de 28 páginas

- Confirme **Escala = 100%** (escalas menores ou “ajustar” podem quebrar o layout).
- Use **A4 + Paisagem**; evite “Tamanho personalizado”.
- Abra o HTML pelo caminho local do projeto (`file:///.../breast-cancer-nn/slides/index.html`), não uma cópia solta do arquivo.
- No preview, role o PDF: cada slide deve caber inteiro em uma página, sem figura cortada na página seguinte.

## Por que isso importa

Na visualização web, cada slide ocupa a tela (`100vh`). Na impressão, o navegador empilha os 28 slides. O CSS `@media print` fixa cada slide em **194 mm de altura** (A4 paisagem com margem 8 mm) e esconde o que passar disso, para não gerar páginas extras nem distorcer figuras.

## Alternativa linha de comando (mesmo resultado do navegador)

```bash
"/Applications/Google Chrome.app/Contents/MacOS/Google Chrome" \
  --headless=new --disable-gpu --no-pdf-header-footer \
  --print-to-pdf="slides/TP3 MC906.pdf" \
  "file://$(pwd)/slides/index.html"
```

Salva em `slides/TP3 MC906.pdf`.
