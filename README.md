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
| Autorregressivo | PixelCNN | Planejada |
| VAE | VAE convolucional | Planejada |
| GAN | DCGAN | Planejada |
| Difusão | DDPM com U-Net | Planejada |
| Difusão latente | LDM condicionado | Etapa final |

As implementações iniciais possuem finalidade didática. Elas não representam
reproduções integrais dos modelos originais nem sistemas preparados para uso
clínico.

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