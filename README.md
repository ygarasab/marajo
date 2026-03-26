# Inferindo Stress Hidrico em Acai no Marajo

![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?logo=python&logoColor=white)
![Status](https://img.shields.io/badge/status-em%20pesquisa-orange)
![OS](https://img.shields.io/badge/linux-supported-success?logo=linux&logoColor=white)
![License](https://img.shields.io/badge/license-MIT-informational)

Projeto de pesquisa para detectar sinais de stress hidrico em acaiizeiros usando apenas video RGB comum (ex.: celular), com foco em extracao de modos vibracionais e analise espectral.

---

## Sumario

- [Contexto](#contexto)
- [Ideia tecnica](#ideia-tecnica)
- [Setup rapido](#setup-rapido)
  - [Requisitos](#requisitos)
  - [Instalacao](#instalacao)
- [Estrutura principal](#estrutura-principal)
- [Pipeline principal (`thin_pipeline`)](#pipeline-principal-thin_pipeline)
  - [Entrada](#entrada)
  - [Execucao](#execucao)
  - [O que o script faz](#o-que-o-script-faz)
  - [Saidas geradas](#saidas-geradas)
- [Scripts auxiliares](#scripts-auxiliares)
- [Dicas praticas](#dicas-praticas)
- [Proximos passos sugeridos](#proximos-passos-sugeridos)

---

## Contexto

Na pratica agricola, o manejo de irrigacao depende de medir, com antecedencia, quando a planta entra em condicao de deficit hidrico. Sensores dedicados existem, mas podem ter alto custo e baixa capilaridade em campo.

A proposta aqui e investigar se o movimento natural da planta, capturado por video, carrega informacao suficiente para inferir estagios de stress hidrico de forma mais acessivel.

## Ideia tecnica

A hipotese central e que a dinamica vibracional do acaiizeiro muda conforme o nivel de stress hidrico.

De forma resumida, o pipeline:

1. carrega o video e converte os frames para escala de cinza;
2. vetoriza os pixels por frame e remove a componente DC (media temporal por pixel);
3. reduz dimensionalidade com PCA;
4. separa fontes com CP (Canonical Polyadic / separacao cega);
5. estima coordenadas modais e plota resultados no tempo e na frequencia.

## Setup rapido

### Requisitos

- Python 3.10+ (recomendado)
- `pip`
- OpenCV com suporte a leitura de `.mp4`

### Instalacao

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

> [!NOTE]
> Opcional: para acelerar etapas numericas em GPU, instalar uma build de CuPy compativel com sua versao CUDA.

> [!TIP]
> Para testes rapidos, rode primeiro com `--scale 0.5` e `--max-frames 200`.

## Estrutura principal

- `scripts/thin_pipeline.py`: pipeline principal enxuto (PCA + CP + modos + figuras)
- `scripts/base_pipeline.py`: variante com augmentacao de Hilbert antes da PCA
- `scripts/initial_video_pipeline.py`: preprocessamento em lote e exportacao de `coeffs.npy` / `scores.npy`
- `scripts/cp_runner.py`: roda CP sobre arquivos `coeffs.npy` ja gerados
- `scripts/video_single_plot.py`: gera figuras e CSV de picos a partir de `unmixed.npy`
- `src/moises/`: nucleos de dados, decomposicao e solucao modal
- `src/marajomodes/`: analise espectral e funcoes de visualizacao

## Pipeline principal (`thin_pipeline`)

O `thin_pipeline` e o fluxo mais direto para analise de um video unico.

### Entrada

- caminho para um video (`.mp4`)
- parametros opcionais:
  - `--max-frames`: limite de frames processados (padrao `400`)
  - `--scale`: fator de escala espacial do frame (padrao `1`)
  - `--out-dir`: pasta de saida (padrao `out`)

### Execucao

```bash
python scripts/thin_pipeline.py input/seu_video.mp4 --max-frames 400 --scale 0.5 --out-dir out/exec_01
```

### O que o script faz

1. Le metadados do video (fps, dimensoes, total de frames).
2. Monta matriz `dataset` com shape `(n_frames, n_pixels)` em grayscale.
3. Remove media temporal por pixel.
4. Executa PCA no dataset transposto.
5. Seleciona `num_pc = 16` componentes para a etapa de CP.
6. Aplica CP para separar fontes vibracionais.
7. Resolve coordenadas modais e formas modais para fontes selecionadas.
8. Salva figuras de diagnostico.

### Saidas geradas

No diretorio definido em `--out-dir`, sao salvos:

- `sources.png`: sinais/frequencias das fontes separadas
- `mode_shapes.png`: formas modais estimadas
- `modal_coord.png`: coordenadas modais no tempo/frequencia

## Scripts auxiliares

### 1) Preprocessamento em lote

Gera artefatos intermediarios (`coeffs.npy`, `scores.npy`) para multiplos videos.

```bash
python scripts/initial_video_pipeline.py
```

> [!WARNING]
> Os caminhos de entrada/saida e combinacoes de escala/frame estao definidos no final do proprio script.

### 2) CP sobre artefatos `.npy`

```bash
python scripts/cp_runner.py out/raw/.../coeffs.npy --npc 16
```

Flags uteis:

- `--cuda`: forca execucao com CuPy/CUDA
- `--cpu`: forca execucao em CPU
- `--out-dir`: altera pasta de saida (`unmixed.npy`, `winvmix.npy`)

### 3) Plot individual a partir de `unmixed.npy`

```bash
python scripts/video_single_plot.py out/raw/.../unmixed.npy --out-dir out/processed
```

Saidas:

- `sources_large.png`
- `sources_simple.png`
- `peaks.csv` (top frequencias por fonte)

## Dicas praticas

- Comece com `--scale 0.3` ou `0.5` para reduzir custo computacional.
- Para exploracao inicial, limite `--max-frames` (ex.: `200` a `500`).
- Garanta boa estabilidade na captura (tripe/apoio) para diminuir ruido nao estrutural.

> [!IMPORTANT]
> O desempenho e a qualidade dos modos extraidos dependem muito da qualidade da captura (iluminacao, estabilidade, vento e compressao do video).

## Proximos passos sugeridos

- Padronizar experimento por protocolo de captura (distancia, iluminacao, vento).
- Versionar metadados por video (talhao, horario, condicao hidrica).
- Criar etapa de classificacao/regressao sobre features modais e espectrais.

---

Se esse projeto te ajudar na pesquisa, considere deixar uma estrela no repositorio.





