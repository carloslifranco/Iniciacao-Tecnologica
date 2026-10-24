# Geração de radiografias com modelos de difusão

Projeto de Iniciação Tecnológica voltado ao estudo de modelos generativos
e ao desenvolvimento de um pipeline baseado em modelos de difusão para
geração de radiografias sintéticas aplicadas ao ensino da radiologia.

## Status do projeto

> Projeto em desenvolvimento.

Etapa atual:

- [x] Definição inicial do tema
- [x] Levantamento das famílias de modelos generativos

## Contexto

O ensino da radiologia depende da exposição dos estudantes a uma quantidade
diversificada de casos clínicos. Entretanto, o acesso a imagens médicas pode
ser limitado por questões de privacidade, disponibilidade, balanceamento de
classes e baixa frequência de determinadas patologias.

Modelos generativos podem auxiliar na criação de imagens sintéticas para
atividades educacionais. Este projeto investiga diferentes famílias desses
modelos, com ênfase em modelos de difusão aplicados à geração de radiografias
de tórax.

## Pergunta de pesquisa

Um modelo de difusão condicionado por achados radiológicos consegue gerar
radiografias de tórax com fidelidade visual e patológica suficiente para
utilização em atividades de treinamento diagnóstico de estudantes de medicina?

## Objetivos

### Objetivo geral

Desenvolver e avaliar um pipeline baseado em modelos de difusão para geração
de radiografias sintéticas e investigar sua utilização como recurso de apoio
ao ensino da radiologia.

### Objetivos específicos

- estudar as principais famílias de modelos generativos;
- selecionar e analisar artigos fundamentais de cada família;
- implementar versões didáticas dos modelos estudados;
- identificar aplicações desses modelos em imagens médicas;
- comparar qualidade, diversidade, custo e estabilidade;
- implementar um modelo de difusão para radiografias;
- avaliar as imagens com métricas computacionais e clínicas;
- desenvolver uma plataforma de autoavaliação;
- investigar a utilidade educacional das imagens geradas.

## Escopo atual

A primeira etapa utiliza um dataset pequeno e não médico para validar:

- carregamento e preparação dos dados;
- treinamento dos modelos;
- geração de imagens;
- armazenamento de checkpoints;
- cálculo das métricas;
- reprodutibilidade dos experimentos.

Após essa validação, o pipeline será adaptado para radiografias de tórax.

## Modelos estudados

| Família | Implementação | Situação |
|---|---|---|
| VAE | VAE MLP e autoencoder convolucional para o espaço latente | Implementação didática |
| GAN | DCGAN | Implementação didática |
| Autorregressivo | PixelCNN | Implementação didática |
| Normalizing Flow | RealNVP com acoplamentos afins | Implementação didática |
| Difusão | DDPM com preditor convolucional de ruído | Implementação didática |
| Difusão latente | Autoencoder convolucional + DDPM latente | Implementação didática |

As implementações iniciais possuem finalidade didática. Elas não representam
reproduções integrais dos modelos originais nem sistemas preparados para uso
clínico.

### Executar os experimentos didáticos

Com o ambiente virtual do projeto ativo, execute um modelo por vez:

```powershell
python scripts\train_vae.py
python scripts\train_gan.py
python scripts\train_autoregressive.py
python scripts\train_normalizing_flow.py
python scripts\train_diffusion.py
python scripts\train_latent_diffusion.py
```

Todos usam Fashion-MNIST e a quantidade de imagens de treino definida em
`data.samples` de `configs/baseline.yml` (atualmente 60.000). Cada execução
cria uma pasta exclusiva em `outputs/<modelo>/<identificador>/`, com data e
hora UTC e um sufixo aleatório. A configuração contém épocas, batch size e
parâmetros específicos de cada família. Para um teste rápido, os scripts
aceitam, por exemplo, `--epochs 1 --samples 128 --batch-size 16`; `--samples`
limita a quantidade de imagens de treino usada nessa execução. Para reduzir o
número de imagens geradas durante um teste, use também
`--generated-samples 1`.

### Registro dos experimentos

Todos os scripts registram a configuração original, a configuração com os
argumentos do terminal aplicados, o ambiente, a identificação dos dados
selecionados e uma cópia do código. Nesta primeira etapa, o registro de
arquitetura, perdas, tempos e estado da execução está integrado ao script
de **difusão**. Os outros modelos ainda não registram esses itens e deixam
o estado do registro como `prepared`.

Para conferir o fluxo com uma execução curta:

```powershell
python scripts\train_diffusion.py --epochs 2 --samples 128 --batch-size 16 --generated-samples 4 --seed 42 --deterministic
```

Esse comando valida o funcionamento; duas épocas com 128 imagens não são
evidência suficiente de qualidade da geração. Para o treinamento com os
valores do baseline:

```powershell
python scripts\train_diffusion.py --deterministic
```

A pasta impressa no terminal conterá:

| Arquivo/pasta | Conteúdo |
|---|---|
| `config_original.yml` | Cópia do YAML antes dos argumentos do terminal |
| `config_effective.yml` | Valores utilizados de dados e treinamento e opções do modelo selecionado |
| `metadata.json` | Sementes, dispositivo, versões, seleção e hash dos dados, arquitetura, parâmetros do otimizador, duração e estado da execução |
| `architecture.txt` | Estrutura da rede de difusão |
| `metrics.csv` | MSE de ruído e tempos por época |
| `samples_epoch_XX.png` | Imagens geradas ao final de cada época |
| `checkpoint.pt` | Última época concluída, pesos, otimizador e argumentos para reconstruir o modelo |
| `source/` | Cópia dos scripts, módulos de `src` e `pyproject.toml`, incluindo alterações locais |

O MSE é ponderado pela quantidade de imagens em cada lote, inclusive no
último lote incompleto. Ele mede previsão de ruído, não qualidade visual.
`train_seconds` mede o treinamento; `sampling_seconds`, a geração e gravação
do PNG; `elapsed_seconds`, o tempo acumulado incluindo checkpoints. A
duração total em `metadata.json` começa na criação do registro e não inclui
o carregamento/download inicial do dataset. O histórico é gravado a cada
época; não há um gráfico automático nesta etapa.

A difusão usa a semente de treinamento para inicializar Python, NumPy,
PyTorch e um gerador independente para a ordem dos lotes. As amostras de
acompanhamento usam `seed + 1`, repetindo a mesma sequência aleatória em
cada época sem alterar os sorteios do treinamento. Isso ajuda a comparar
a evolução visual. Essas amostras são de acompanhamento, não um conjunto
independente para avaliação final.

`--seed` substitui a semente do YAML. `--deterministic` ativa algoritmos
determinísticos e pode interromper uma execução se uma operação não tiver
implementação determinística no dispositivo. Também é possível definir
`training.deterministic: true` no YAML. Mesmo assim, resultados idênticos
entre versões de bibliotecas ou equipamentos diferentes não são garantidos.

O checkpoint é atualizado após cada época de difusão, dentro da pasta
daquela execução. Uma falha ou interrupção por teclado é registrada no
metadado e propagada ao terminal; encerramentos forçados do processo podem
deixar o estado como `running`. Ainda não há retomada automática de
treinamento nem um comando separado de inferência. O checkpoint sozinho
não contém todos os estados aleatórios necessários para uma retomada exata.
As saídas antigas diretamente em `outputs/diffusion/` são preservadas.

Os registros e cópias locais estão em `outputs/`, ignorado pelo Git.
Para arquivar um resultado da pesquisa, preserve a pasta completa da execução.

Os testes de integração usam dados artificiais pequenos e CPU, sem download:

```powershell
python -m unittest discover -s tests -v
```

## Aplicações médicas investigadas

A revisão bibliográfica considera aplicações em:

- radiografias de tórax;
- ressonância magnética;
- tomografia computadorizada;
- imagens de retina;
- histopatologia.

O foco experimental final será definido a partir da disponibilidade dos dados,
dos recursos computacionais e da discussão com especialistas.

## Dados

### Dataset didático

As primeiras implementações utilizam o Fashion-MNIST para validar o pipeline
com baixo custo computacional.

### Dataset médico

O projeto prevê a utilização do CheXpert, composto por radiografias de tórax
e rótulos associados a observações radiológicas.

Os dados médicos não são armazenados neste repositório. Consulte
[`data/README.md`](data/README.md) para instruções de obtenção e preparação.

### Divisão dos dados

Nos experimentos médicos, as divisões de treino, validação e teste serão
realizadas por paciente, evitando que imagens do mesmo indivíduo apareçam
em conjuntos diferentes.

## Métricas

As implementações serão avaliadas por diferentes grupos de métricas:

### Qualidade e fidelidade

- Fréchet Inception Distance — FID;
- Kernel Inception Distance — KID;
- precision generativa.

### Diversidade

- recall generativo;
- similaridade entre amostras;
- análise de duplicações;
- comparação com imagens do treinamento.

### Reconstrução

Quando houver correspondência entre entrada e saída:

- Structural Similarity Index — SSIM;
- Peak Signal-to-Noise Ratio — PSNR.

### Avaliação clínica

Na etapa médica também serão considerados:

- coerência anatômica;
- presença da patologia condicionada;
- ocorrência de artefatos;
- avaliação cega por especialistas;
- utilidade educacional.

Nenhuma métrica computacional será interpretada isoladamente como evidência
de correção clínica.

## Estrutura do repositório

```text
.
├── configs/       # Configurações dos experimentos
├── data/          # Documentação e dados locais não versionados
├── notebooks/     # Exploração e experimentos didáticos
├── references/    # Fichamentos e referências bibliográficas
├── reports/       # Relatórios e análises
├── scripts/       # Pontos de entrada executáveis
├── src/           # Código reutilizável
├── tests/         # Testes automatizados
