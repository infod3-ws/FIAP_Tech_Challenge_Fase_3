# Tech Challenge — Fase 3
## Predição e Inteligência Analítica para Alfabetização no Brasil

**Curso:** Pós-Graduação em Inteligência Artificial (AI Scientist) — FIAP PosTech
**Aluno:** Weric do Amaral Silva

---

## 1. Contexto do Problema

Na Fase 2 deste projeto, construímos um pipeline de engenharia de dados que integrou sete entidades públicas relacionadas ao **Indicador Criança Alfabetizada** — metas nacionais, estaduais e municipais, dados territoriais e microdados de avaliação — em uma arquitetura Medalhão no Google Cloud.

Compreender os dados históricos, porém, não é suficiente para apoiar decisão pública. Gestores educacionais precisam **antecipar risco**: identificar, antes do resultado final, quais alunos e municípios têm maior chance de não atingir o patamar de alfabetização — para que a intervenção aconteça a tempo, não depois do diagnóstico.

Esta fase transforma a camada Silver/Gold construída anteriormente em um modelo preditivo real, aplicando técnicas de aprendizado supervisionado sobre um problema social concreto.

## 2. Objetivo Analítico

Construir um **modelo de classificação binária** capaz de prever se um aluno do 2º ano do ensino fundamental será considerado **alfabetizado** ou **não alfabetizado**, com base em variáveis educacionais e territoriais conhecidas **antes** do resultado da avaliação — nunca a partir do resultado em si (ver Seção 8, sobre tratamento de vazamento de dados).

## 3. Descrição da Base Utilizada

A base de modelagem foi construída a partir de três fontes, todas já tratadas na Fase 2:

| Fonte | Camada | Papel |
|---|---|---|
| `alunos` | Silver | Registro individual por aluno — base da modelagem |
| `meta_alfabetizacao_municipio` | Silver | Metas de política pública por município/ano/rede |
| `meta_alfabetizacao_uf` | Bronze* | Metas de política pública por UF/ano |

*Nunca havia sido tratada na Silver durante a Fase 2 — pequena limpeza feita localmente neste projeto, sem alterar os scripts da Fase 2 já entregues.

**Volume final:** 3.354.661 registros, 11 colunas (após as correções descritas na Seção 8).

**Variável-alvo:** `alfabetizado` (Sim / Não), derivada do corte de 743 pontos na escala de proficiência do Saeb (Pesquisa Alfabetiza Brasil, INEP 2023).

**Features utilizadas:**

| Tipo | Variáveis |
|---|---|
| Categóricas | `rede` (Municipal/Estadual/Privada), `sigla_uf` |
| Numéricas | `ano`, `meta_alfabetizacao_2024/2025/2026_municipio`, `meta_alfabetizacao_2024/2025/2026_uf` |

`sigla_uf` foi derivada do código IBGE do município (2 primeiros dígitos), sem depender de tabela externa — resolvendo um ponto de melhoria apontado no feedback da Fase 2 (a entidade `uf`/`meta_alfabetizacao_uf` nunca era usada como feature).

## 4. Etapas de Modelagem

1. **Construção do dataset** (`src/preprocessing/build_dataset.py`) — integração das três fontes, remoção de colunas de vazamento e de identificadores administrativos, filtro de escopo (Seção 8).
2. **Análise exploratória** (`notebooks/01_eda.ipynb`) — distribuição do alvo, relação entre variáveis categóricas/numéricas e o alvo, correlação entre metas.
3. **Pré-processamento integrado ao modelo** (`src/modeling/train_model.py`) — via `ColumnTransformer` + `Pipeline` do scikit-learn:
   - Imputação de numéricas: `SimpleImputer(strategy="median")`
   - Escalonamento: `StandardScaler`
   - Imputação de categóricas: `SimpleImputer(strategy="most_frequent")`
   - Codificação: `OneHotEncoder(handle_unknown="ignore")`
4. **Divisão treino/teste** estratificada (80/20), `random_state` fixo em todo o pipeline.
5. **Otimização de hiperparâmetros** via `GridSearchCV` + `StratifiedKFold` (5 dobras), buscada em uma amostra estratificada de 300 mil registros — o treino final, com a melhor configuração já escolhida, roda na base completa (decisão de FinOps de tempo/memória local; ver Seção 8).
6. **Avaliação** no conjunto de teste completo (nunca visto durante a busca de hiperparâmetros).
7. **Interpretabilidade** via SHAP (`src/evaluation/interpret_shap.py`).

## 5. Escolha do Algoritmo

Comparamos dois modelos de classificação, ambos vistos no módulo de Aprendizado Supervisionado do curso:

| Modelo | ROC-AUC (teste) | Justificativa da comparação |
|---|---|---|
| Regressão Logística | 0,6623 | Baseline interpretável, rápida, referência de mercado |
| **Random Forest** (vencedor) | **0,6728** | Ensemble de árvores, captura relações não-lineares entre metas e alvo |

O Random Forest venceu por margem pequena, mas consistente. Hiperparâmetros finais: `n_estimators=200`, `max_depth=10` (profundidade limitada intencionalmente — testamos `None`, que estourou a memória da máquina local em 3+ milhões de registros; documentado como decisão de custo/performance).

## 6. Métricas de Avaliação

| Métrica | Valor (conjunto de teste) |
|---|---|
| Acurácia | 0,6412 |
| Precisão | 0,6572 |
| Recall | 0,8223 |
| F1-score | 0,7305 |
| ROC-AUC | 0,6728 |
| Gap treino-teste (ROC-AUC) | 0,0016 |

**Sobre o gap treino-teste:** a diferença de apenas 0,0016 entre a performance em treino e em teste indica **ausência de overfitting** — o modelo generaliza para dados não vistos, não apenas memoriza o conjunto de treino. Essa comparação, junto com a validação cruzada de 5 dobras, é a evidência de replicabilidade e generalização exigida pelo desafio.

## 7. Interpretação dos Resultados

A análise via SHAP (`reports/shap_ranking.csv`, `images/shap_summary.png`) revela um padrão claro: **as metas municipais de curto prazo dominam a decisão do modelo**, muito à frente de qualquer outra variável.

| Ranking | Variável | Importância média (SHAP) |
|---|---|---|
| 1º | `meta_alfabetizacao_2026_municipio` | 0,0326 |
| 2º | `meta_alfabetizacao_2025_municipio` | 0,0321 |
| 3º | `meta_alfabetizacao_2024_municipio` | 0,0262 |
| 4º | `ano` | 0,0094 |
| 5º-7º | metas por UF (2024-2026) | 0,0062 – 0,0072 |
| 8º-10º | `sigla_uf` (MG, RS) e `rede` (Estadual) | 0,0024 – 0,0048 |

**Leitura de negócio:** o contexto **municipal** pesa mais que o **estadual** na predição — faz sentido, já que a meta municipal reflete a realidade mais próxima da escola do aluno, enquanto a meta estadual é uma média que dilui variações locais. As variáveis categóricas de UF e rede específicas aparecem bem abaixo das metas numéricas, contribuindo de forma marginal e individualizada (nenhum estado ou rede isolado domina a decisão).

Isso reforça um dos achados da análise exploratória (Seção 8): a meta municipal de curto prazo (2024) já mostrava, isoladamente, a maior diferença de médias entre as classes "Sim"/"Não" — o SHAP confirma, do ponto de vista do modelo treinado, que esse sinal realmente é o mais informativo.

## 8. Insights Encontrados

**Achado metodológico — vazamento de rótulo indireto (o mais relevante do projeto):**
Na análise exploratória, identificamos que `presenca = "Ausente"` e `preenchimento_caderno = "Prova não preenchida"` correspondiam a **100% de rótulo "Não alfabetizado"** — não porque o aluno foi avaliado e não atingiu o corte, mas porque nunca teve `proficiencia` medida. Mantê-los faria o modelo aprender "o aluno fez a prova?" em vez de "o aluno está alfabetizado?".

**Correção aplicada:** filtramos o dataset para conter apenas alunos com `presenca = "Presente"` e `preenchimento_caderno = "Prova preenchida"` — população para quem o rótulo tem significado real.

| | Antes da correção | Depois da correção |
|---|---|---|
| Registros | 3.867.589 | 3.354.661 (-13,3%) |
| Distribuição do alvo | 51,3% Sim / 48,7% Não | 59,2% Sim / 40,8% Não |
| `rede` vs. alvo | Privada: 100% Não (artefato) | Privada: 50%/50% (real) |
| ROC-AUC (teste) | 0,7602 | 0,6728 |

**A queda do ROC-AUC é esperada e correta, não uma piora.** O valor anterior estava parcialmente inflado por um atalho artificial (ausência de avaliação = "Não" por definição). O valor atual reflete sinal genuíno, mais modesto — e essa é uma decisão metodológica deliberada, documentada e rastreável no histórico do Git (branch `fix/filtro-alunos-nao-avaliados`).

**Outros achados da análise exploratória:**
- **Desigualdade regional acentuada**: municípios do Ceará apresentaram taxa de alfabetização em torno de 82%, contra cerca de 30% no Rio Grande do Norte — uma diferença de mais de 50 pontos percentuais entre os extremos.
- **Metas de curto prazo têm relação real com o resultado**: alunos "Sim" vêm de municípios com meta 2024 em média 6,8 pontos percentuais mais alta que os "Não" — sinal genuíno, não ruído.
- **Metas municipais e estaduais são fortemente correlacionadas entre si** (0,73-1,00) — esperado, mesma trajetória de política pública.
- **`meta_alfabetizacao_2030` é constante** (80% para todos os municípios/UFs) — sem poder discriminativo, removida do modelo.

## 9. Limitações do Projeto

- **Sem enriquecimento externo:** por restrição de tempo, não incorporamos fontes complementares (Censo Escolar, IBGE, PNAD, FUNDEB) sugeridas como opcionais no edital. As features numéricas disponíveis se restringem a metas de política pública — nenhum dado de infraestrutura escolar ou contexto socioeconômico do aluno está presente.
- **Poder preditivo moderado (ROC-AUC 0,67):** consequência direta da decisão da Seção 8 — um modelo mais "impressionante" era possível mantendo o vazamento de rótulo, mas seria enganoso.
- **Classe levemente desbalanceada** (59%/41%) após o filtro de escopo — não tratamos com técnicas de balanceamento (SMOTE, `class_weight`), pois o desbalanceamento é moderado e as métricas de avaliação já contemplam essa assimetria (precisão, recall, F1, não só acurácia).
- **`max_depth` do Random Forest limitado a 10** por restrição de memória da máquina local, não necessariamente o ótimo absoluto — a busca de hiperparâmetros foi feita em amostra, não na base completa, por tempo/memória.
- **Meta nacional (`meta_alfabetizacao_brasil`) não incluída como feature:** possui apenas
  3 valores distintos (um por ano), tornando-a redundante com a variável `ano` já presente
  no modelo — decisão consciente, não uma omissão.

## 10. Aplicação Prática para Políticas Públicas

- **Priorização de intervenção:** o modelo pode ser aplicado a coortes de alunos no início do ano letivo, sinalizando municípios/redes com maior proporção de alunos em risco, direcionando reforço pedagógico antes da avaliação oficial.
- **Acompanhamento de metas:** como as variáveis de meta têm relação mensurável com o resultado, gestores podem simular o efeito de revisar metas municipais sobre a taxa de alfabetização projetada.
- **Redução de desigualdade regional:** o achado da Seção 8 sobre a disparidade entre UFs reforça a necessidade de políticas diferenciadas por região, não um programa nacional único.

## 11. Possíveis Evoluções Futuras

- Incorporar dados socioeconômicos externos (Censo Escolar, Cadastro Único, Atlas do Desenvolvimento Humano) para elevar o poder preditivo sem reintroduzir vazamento.
- Testar `HistGradientBoostingClassifier` ou XGBoost/LightGBM, mais eficientes em memória que Random Forest para bases de milhões de linhas.
- Rodar a busca de hiperparâmetros na base completa via ambiente com mais memória (ex: Dataproc Serverless ou Vertex AI Training), em vez de amostra local.
- Modelo de clusterização (K-Means/Agglomerative, vistos no módulo de Aprendizado Não Supervisionado) sobre indicadores municipais, para responder à pergunta "quais regiões possuem padrões semelhantes?" — não explorado nesta entrega por escopo/tempo.

## 12. Como Reproduzir

> **Nota sobre acesso:** os scripts abaixo referenciam o projeto e bucket GCP
> utilizados no desenvolvimento deste projeto. Reproduzir a leitura dos dados
> reais exige permissão de IAM nesse projeto específico, que não é compartilhada
> publicamente por questões de segurança e custo. Para rodar com seus próprios
> dados/projeto, defina as variáveis de ambiente `FIAP_GCP_PROJECT` e
> `FIAP_GCS_BUCKET` (ver `src/config.py`) antes de executar.

```bash
python3 -m venv .venv
source .venv/bin/activate
python3 -m pip install -r requirements.txt

gcloud auth application-default login   # autentica com SUA propria conta Google

python3 src/preprocessing/build_dataset.py
python3 src/modeling/train_model.py
python3 src/evaluation/interpret_shap.py
```

## 13. Visualizações Finais

Depois de rodar o pipeline completo (Seção 12), um último script consolida os
resultados em gráficos prontos para o README e para a apresentação executiva —
sem reprocessar nada pesado, só lê os relatórios já salvos:

```bash
python3 src/visualization/gerar_graficos_finais.py
```

Gera dois arquivos em `images/`:
- **`metricas_finais.png`** — acurácia, precisão, recall, F1 e ROC-AUC do modelo vencedor, lado a lado
- **`shap_ranking_top10.png`** — as 10 variáveis mais influentes na predição, segundo o SHAP (mesmo ranking da Seção 7)

## 14. Estrutura do Repositório

```
├── data/                         # dataset_ml_alunos.parquet (reproduzível, não versionado)
├── notebooks/
│   └── 01_eda.ipynb              # análise exploratória completa, incluindo a correção da Seção 8
├── src/
│   ├── config.py                 # projeto/bucket GCP centralizados
│   ├── preprocessing/
│   │   └── build_dataset.py
│   ├── modeling/
│   │   └── train_model.py
│   ├── evaluation/
│   │   └── interpret_shap.py
│   └── visualization/
│       └── gerar_graficos_finais.py   # gráficos-resumo a partir dos reports já salvos
├── reports/
│   ├── metrics.json
│   ├── shap_ranking.csv
│   └── models/                   # pipeline .joblib (reproduzível, não versionado)
├── images/                       # gráficos do EDA, SHAP e resumo final (metricas_finais.png, shap_ranking_top10.png)
├── docs/
│   └── dicionario_features.md    # documentação técnica: cada feature, tipo e tratamento no pipeline
├── requirements.txt
├── README.md
└── .gitignore
```

## 15. Apresentação Executiva

[Assista à apresentação executiva da solução](https://github.com/infod3-ws/FIAP_Tech_Challenge_Fase_3/releases/download/v0.1-fiap-tc3-apresentacao/Apresentacao_Executiva_TechChallenge_Fase3.mp4)

O vídeo (5 minutos) aborda o problema de negócio, a arquitetura da solução, o achado metodológico sobre vazamento indireto de rótulo (Seção 8), as métricas finais do modelo, a interpretabilidade via SHAP (Seção 7) e a aplicação prática para políticas públicas (Seção 10).