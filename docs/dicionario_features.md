<!-- docs/dicionario_features.md -->
# Dicionário de Features — Modelo de Predição

| Feature | Tipo | Descrição | Tratamento no pipeline |
|---|---|---|---|
| `ano` | Numérica | Ano de referência da avaliação (2023/2024) | Escalonamento (StandardScaler) |
| `rede` | Categórica | Rede de ensino do aluno (Municipal/Estadual/Privada) | One-Hot Encoding |
| `sigla_uf` | Categórica | UF derivada do código IBGE do município | One-Hot Encoding |
| `meta_alfabetizacao_2024_municipio` | Numérica | Meta de alfabetização do município para 2024 | Imputação (mediana) + Escalonamento |
| `meta_alfabetizacao_2025_municipio` | Numérica | Idem, 2025 | Imputação (mediana) + Escalonamento |
| `meta_alfabetizacao_2026_municipio` | Numérica | Idem, 2026 | Imputação (mediana) + Escalonamento |
| `meta_alfabetizacao_2024_uf` | Numérica | Meta de alfabetização da UF para 2024 | Imputação (mediana) + Escalonamento |
| `meta_alfabetizacao_2025_uf` | Numérica | Idem, 2025 | Imputação (mediana) + Escalonamento |
| `meta_alfabetizacao_2026_uf` | Numérica | Idem, 2026 | Imputação (mediana) + Escalonamento |

**Alvo:** `alfabetizado` (Sim/Não), derivado do corte de 743 pontos na escala de proficiência do Saeb.

**Colunas excluídas por data leakage:** `proficiencia`, `taxa_alfabetizacao`, `percentual_participacao`, `nivel_alfabetizacao` — ver Seção 8 do README para justificativa completa.

**Colunas removidas por variância zero:** `serie`, `meta_alfabetizacao_2030_municipio`, `meta_alfabetizacao_2030_uf`.