# Exportar apresentação em PDF

## Opção rápida (recomendada)

1. Abra `slides/index.html` no Chrome ou Edge.
2. Pressione **P** (ou `Cmd+P` / `Ctrl+P`).
3. Destino: **Salvar como PDF**.
4. Layout: **Paisagem**.
5. Margens: **Padrão** ou **Mínima**.
6. Ative **Gráficos de segundo plano** (para cores e barras teal).
7. Salvar.

Todos os **28 slides** saem em sequência (um por página).

## Dicas

- Use **F** antes de imprimir se quiser tela cheia na apresentação; o PDF não depende disso.
- Se alguma figura não aparecer, abra o HTML a partir da pasta do projeto (caminhos `../figures/` e `../assets/`).
- No Safari: `Arquivo → Exportar como PDF` também funciona; confira a opção de imprimir cores de fundo.

## Alternativa (linha de comando)

Com [decktape](https://github.com/astefanutti/decktape) instalado:

```bash
npx decktape generic file://$(pwd)/slides/index.html slides/apresentacao.pdf --size 1920x1080
```

Requer Node.js; o método do navegador costuma ser mais simples.
