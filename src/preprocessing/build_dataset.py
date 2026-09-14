import logging
import sys
from pathlib import Path

import pandas as pd
from google.cloud import storage

sys.path.append(str(Path(__file__).resolve().parents[1]))
from config import BILLING_PROJECT, BUCKET_NAME, CODIGO_UF_PARA_SIGLA

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

client = storage.Client(project=BILLING_PROJECT)
bucket = client.bucket(BUCKET_NAME)

# ============================================================
# LEITURA
# ============================================================

def ler_prefixo(prefixo):
    blobs = [b for b in bucket.list_blobs(prefix=prefixo) if b.name.endswith(".parquet")]
    if not blobs:
        raise FileNotFoundError(f"Nenhum arquivo em {prefixo}")
    dfs = []
    for b in blobs:
        arquivo = f"/tmp/{b.name.split('/')[-1]}"
        b.download_to_filename(arquivo)
        dfs.append(pd.read_parquet(arquivo))
    df = pd.concat(dfs, ignore_index=True)
    log.info(f"Lido {prefixo}: {len(df)} registros")
    return df

# ============================================================
# CONSTRUÇÃO DO DATASET DE MODELAGEM
# ============================================================

# Colunas que NÃO entram no modelo: identificadores, metadados de auditoria,
# e — CRÍTICO — qualquer coluna derivada do mesmo ciclo de avaliação do aluno,
# pois isso é vazamento de dados (data leakage).
COLUNAS_VAZAMENTO = [
    "proficiencia",            # fonte direta do rótulo 'alfabetizado' (corte de 743 pontos)
    "alfabetizado_codigo", "rede_codigo", "presenca_codigo", "preenchimento_caderno_codigo",
    "taxa_alfabetizacao",      # agregado REALIZADO do mesmo ciclo — inclui o próprio aluno
    "nivel_alfabetizacao", "percentual_participacao",  # idem: resultado observado, não meta
]
COLUNAS_METADADOS = [c for c in [] ]  # placeholder — removidas via prefixo '_' abaixo

COLUNAS_ADMINISTRATIVAS = ["id_municipio", "id_escola", "caderno", "peso_aluno"]

def montar_dataset():
    log.info("=" * 60)
    log.info("CONSTRUINDO DATASET DE MODELAGEM (Fase 3)")
    log.info("=" * 60)

    df_alunos = ler_prefixo("silver/alunos/")
    df_meta_municipio = ler_prefixo("silver/meta_alfabetizacao_municipio/")
    df_meta_uf = ler_prefixo("bronze/meta_alfabetizacao_uf/")

    df_alunos["codigo_uf"] = df_alunos["id_municipio"].astype(str).str[:2]
    df_alunos["sigla_uf"] = df_alunos["codigo_uf"].map(CODIGO_UF_PARA_SIGLA)

    metas_municipio_cols = ["id_municipio", "ano", "rede",
        "meta_alfabetizacao_2024", "meta_alfabetizacao_2025",
        "meta_alfabetizacao_2026", "meta_alfabetizacao_2030"]
    df_meta_municipio = df_meta_municipio[metas_municipio_cols].rename(columns={
        c: f"{c}_municipio" for c in metas_municipio_cols if c.startswith("meta_")
    })

    # CORRIGIDO: meta_alfabetizacao_uf só tem rede="Pública" (meta única do estado,
    # não quebrada por rede de ensino) — por isso o merge NÃO usa 'rede' como chave aqui.
    df_meta_uf["ano"] = df_meta_uf["ano"].astype(int)
    metas_uf_cols = ["sigla_uf", "ano",
        "meta_alfabetizacao_2024", "meta_alfabetizacao_2025",
        "meta_alfabetizacao_2026", "meta_alfabetizacao_2030"]
    df_meta_uf = df_meta_uf[metas_uf_cols].rename(columns={
        c: f"{c}_uf" for c in metas_uf_cols if c.startswith("meta_")
    })

    df = df_alunos.merge(df_meta_municipio, on=["id_municipio", "ano", "rede"], how="left")
    df = df.merge(df_meta_uf, on=["sigla_uf", "ano"], how="left")

    colunas_para_remover = (
        [c for c in df.columns if c.startswith("_")]
        + [c for c in COLUNAS_VAZAMENTO if c in df.columns]
        + [c for c in COLUNAS_ADMINISTRATIVAS if c in df.columns]
        + ["codigo_uf"]
    )
    df = df.drop(columns=colunas_para_remover, errors="ignore")

    if "serie" in df.columns and df["serie"].nunique() <= 1:
        log.info("[EDA] Coluna 'serie' tem variancia zero — removida (sem poder preditivo)")
        df = df.drop(columns=["serie"])

    # --- Metas de 2030 são constantes (80,0 para todos os municípios/UFs) ---
    # Confirmado no heatmap de correlação do notebook de EDA: correlação indefinida
    # por variância zero. Sem poder preditivo, mesmo raciocínio de 'serie'.
    colunas_meta_2030 = ["meta_alfabetizacao_2030_municipio", "meta_alfabetizacao_2030_uf"]
    for col in colunas_meta_2030:
        if col in df.columns and df[col].nunique(dropna=True) <= 1:
            log.info(f"[EDA] Coluna '{col}' tem variancia zero — removida")
            df = df.drop(columns=[col])

    # --- Filtro de escopo: mantemos só alunos efetivamente avaliados ---
    # Achado no notebook de EDA (célula 9/10): presenca="Ausente" e
    # preenchimento_caderno="Prova não preenchida" são 100% "Não" por construção
    # (o aluno nunca teve proficiencia medida, não é um "Não alfabetizado" real).
    # Manter esses registros faria o modelo aprender "o aluno fez a prova?"
    # em vez de "o aluno está alfabetizado?" — um vazamento indireto de rótulo.
    antes = len(df)
    df = df[(df["presenca"] == "Presente") & (df["preenchimento_caderno"] == "Prova preenchida")]
    log.info(f"[ESCOPO] Filtro presenca/preenchimento: {antes} -> {len(df)} registros "
              f"({(antes - len(df)) / antes * 100:.1f}% removidos — alunos nao avaliados)")

    # Depois do filtro, presenca e preenchimento_caderno viram colunas constantes
    # (só restou "Presente"/"Prova preenchida") — sem poder preditivo, removidas.
    for col in ["presenca", "preenchimento_caderno"]:
        if col in df.columns and df[col].nunique(dropna=True) <= 1:
            log.info(f"[EDA] Coluna '{col}' ficou constante apos o filtro — removida")
            df = df.drop(columns=[col])

    log.info(f"Dataset final: {len(df)} registros, {len(df.columns)} colunas")
    log.info(f"Colunas: {list(df.columns)}")
    log.info(f"Nulos por coluna:\n{df.isna().sum()}")

    return df
        
def salvar(df):
    caminho_local = "data/dataset_ml_alunos.parquet"
    df.to_parquet(caminho_local, index=False)
    log.info(f"Salvo localmente em {caminho_local}")

    destino_gcs = "ml/dataset_alunos.parquet"
    bucket.blob(destino_gcs).upload_from_filename(caminho_local)
    log.info(f"Salvo no GCS: gs://{BUCKET_NAME}/{destino_gcs}")

if __name__ == "__main__":
    df = montar_dataset()
    salvar(df)