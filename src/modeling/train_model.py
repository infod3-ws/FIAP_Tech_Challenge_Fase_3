import json
import logging
import sys
from pathlib import Path

import joblib
import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (accuracy_score, classification_report, confusion_matrix,
                              f1_score, precision_score, recall_score, roc_auc_score)
from sklearn.model_selection import GridSearchCV, StratifiedKFold, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

RANDOM_STATE = 42  # fixo em todo o script — garante replicabilidade

# ============================================================
# CARGA
# ============================================================

df = pd.read_parquet("data/dataset_ml_alunos.parquet")
TARGET = "alfabetizado"

y = (df[TARGET] == "Sim").astype(int)
X = df.drop(columns=["id_aluno", TARGET])

# Colunas declaradas explicitamente (não por inferência de dtype — mais robusto
# entre versões de pandas; detectamos um bug real disso em teste)
NUMERIC_FEATURES = [c for c in X.columns if pd.api.types.is_numeric_dtype(X[c])]
CATEGORICAL_FEATURES = [c for c in X.columns if not pd.api.types.is_numeric_dtype(X[c])]
log.info(f"Numericas: {NUMERIC_FEATURES}")
log.info(f"Categoricas: {CATEGORICAL_FEATURES}")

# ============================================================
# PRÉ-PROCESSAMENTO — imputação + transformação, integrados ao pipeline
# ============================================================

numeric_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="median")),   # imputação de faltantes numéricos
    ("scaler", StandardScaler()),                     # escalonamento
])
categorical_transformer = Pipeline(steps=[
    ("imputer", SimpleImputer(strategy="most_frequent")),
    ("encoder", OneHotEncoder(handle_unknown="ignore")),  # transformação categórica
])
preprocessor = ColumnTransformer(transformers=[
    ("num", numeric_transformer, NUMERIC_FEATURES),
    ("cat", categorical_transformer, CATEGORICAL_FEATURES),
])

# ============================================================
# SPLIT — antes de qualquer fit, evita vazamento entre treino/teste
# ============================================================

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)
log.info(f"Treino: {len(X_train)} | Teste: {len(X_test)}")

# ============================================================
# AMOSTRA PARA BUSCA DE HIPERPARÂMETROS
# ============================================================
# Buscar a melhor configuração não precisa da base inteira — usamos uma
# amostra estratificada (mantém a proporção da classe) para tornar o
# GridSearchCV viável em tempo/memória de máquina local. O treino FINAL,
# com os melhores parâmetros já escolhidos, roda na base completa.

TAMANHO_AMOSTRA = 300_000
if len(X_train) > TAMANHO_AMOSTRA:
    X_amostra, _, y_amostra, _ = train_test_split(
        X_train, y_train,
        train_size=TAMANHO_AMOSTRA,
        random_state=RANDOM_STATE,
        stratify=y_train,
    )
else:
    X_amostra, y_amostra = X_train, y_train

log.info(f"Amostra para GridSearchCV: {len(X_amostra)} registros "
         f"({len(X_amostra)/len(X_train)*100:.1f}% do treino)")

# ============================================================
# MODELOS — pré-processamento integrado diretamente ao classificador
# ============================================================

MODELOS = {
    "logistic_regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "random_forest": RandomForestClassifier(random_state=RANDOM_STATE),
}
GRADES_HIPERPARAMETROS = {
    "logistic_regression": {"classifier__C": [0.01, 0.1, 1.0, 10.0]},
    "random_forest": {
        "classifier__n_estimators": [100, 200],
        "classifier__max_depth": [10, 20],   # removido 'None' — profundidade ilimitada
                                              # em 3M de linhas estourava memória
    },
}

cv = StratifiedKFold(n_splits=3, shuffle=True, random_state=RANDOM_STATE)  # 5 -> 3 dobras
resultados = {}
melhor_nome, melhor_pipeline, melhor_auc = None, None, -1

for nome, modelo in MODELOS.items():
    log.info(f"--- Buscando hiperparametros para {nome} (na amostra) ---")
    pipeline = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", modelo)])

    grid = GridSearchCV(pipeline, GRADES_HIPERPARAMETROS[nome], cv=cv,
                         scoring="roc_auc", n_jobs=-1)
    grid.fit(X_amostra, y_amostra)

    log.info(f"{nome}: melhores hiperparametros = {grid.best_params_} "
             f"(ROC-AUC na amostra, CV = {grid.best_score_:.4f})")

    # --- Treino FINAL: mesma configuração vencedora, agora na base COMPLETA ---
    log.info(f"--- Treinando {nome} na base completa com os melhores parametros ---")
    pipeline_final = Pipeline(steps=[("preprocessor", preprocessor), ("classifier", modelo)])
    pipeline_final.set_params(**grid.best_params_)
    pipeline_final.fit(X_train, y_train)

    y_pred = pipeline_final.predict(X_test)
    y_proba = pipeline_final.predict_proba(X_test)[:, 1]

    metricas = {
        "accuracy": accuracy_score(y_test, y_pred),
        "precision": precision_score(y_test, y_pred),
        "recall": recall_score(y_test, y_pred),
        "f1": f1_score(y_test, y_pred),
        "roc_auc_teste": roc_auc_score(y_test, y_proba),
        "roc_auc_cv_amostra": grid.best_score_,
        "melhores_hiperparametros": grid.best_params_,
        "tamanho_amostra_busca": len(X_amostra),
    }

    score_treino = roc_auc_score(y_train, pipeline_final.predict_proba(X_train)[:, 1])
    metricas["roc_auc_treino_full"] = score_treino
    metricas["gap_treino_teste"] = round(score_treino - metricas["roc_auc_teste"], 4)

    resultados[nome] = metricas
    log.info(f"{nome} (base completa): {metricas}")

    if metricas["roc_auc_teste"] > melhor_auc:
        melhor_auc = metricas["roc_auc_teste"]
        melhor_nome = nome
        melhor_pipeline = pipeline_final

log.info("=" * 60)
log.info(f"MELHOR MODELO: {melhor_nome} (ROC-AUC teste = {melhor_auc:.4f})")
log.info("=" * 60)

# ============================================================
# RELATÓRIO FINAL
# ============================================================

y_pred_final = melhor_pipeline.predict(X_test)
matriz = confusion_matrix(y_test, y_pred_final).tolist()
relatorio_classificacao = classification_report(y_test, y_pred_final, output_dict=True)

Path("reports").mkdir(exist_ok=True)
with open("reports/metrics.json", "w") as f:
    json.dump({
        "melhor_modelo": melhor_nome,
        "resultados_por_modelo": resultados,
        "matriz_confusao": matriz,
        "classification_report": relatorio_classificacao,
        "features_numericas": NUMERIC_FEATURES,
        "features_categoricas": CATEGORICAL_FEATURES,
        "colunas_excluidas_por_data_leakage": [
            "proficiencia", "taxa_alfabetizacao", "percentual_participacao", "nivel_alfabetizacao"
        ],
    }, f, indent=2, default=str)
log.info("Relatorio salvo em reports/metrics.json")

Path("reports/models").mkdir(exist_ok=True, parents=True)
joblib.dump(melhor_pipeline, f"reports/models/{melhor_nome}_pipeline.joblib")
log.info(f"Modelo salvo em reports/models/{melhor_nome}_pipeline.joblib")

# guarda X_test/y_test para a etapa de interpretabilidade (SHAP), sem precisar refazer o split
X_test.to_parquet("reports/models/X_test.parquet", index=False)
y_test.to_frame("alfabetizado").to_parquet("reports/models/y_test.parquet", index=False)