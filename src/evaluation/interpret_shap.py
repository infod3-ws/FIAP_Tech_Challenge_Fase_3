import logging

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

pipeline = joblib.load("reports/models/random_forest_pipeline.joblib")
X_test = pd.read_parquet("reports/models/X_test.parquet")

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

AMOSTRA = 300
X_test_transformado = preprocessor.transform(X_test.iloc[:AMOSTRA])
if hasattr(X_test_transformado, "toarray"):  # OneHotEncoder pode gerar matriz esparsa
    X_test_transformado = X_test_transformado.toarray()

nomes_features = preprocessor.get_feature_names_out()
log.info(f"Calculando SHAP para {X_test_transformado.shape[0]} registros, {len(nomes_features)} features")

explainer = shap.Explainer(classifier, X_test_transformado)
shap_values = explainer(X_test_transformado)

# Modelos de classificação binária podem retornar valores em 3 dimensões
# (amostras x features x classes). Selecionamos explicitamente a classe
# positiva ("Sim" = índice 1) para o grafico-resumo padrao.
valores = shap_values.values
if valores.ndim == 3:
    log.info(f"SHAP retornou {valores.shape[-1]} classes — selecionando a classe positiva (indice 1)")
    valores = valores[:, :, 1]

plt.figure()
shap.summary_plot(valores, features=X_test_transformado, feature_names=list(nomes_features),
                   max_display=15, show=False)
plt.tight_layout()
plt.savefig("images/shap_summary.png", dpi=150, bbox_inches="tight")
log.info("Grafico salvo em images/shap_summary.png")

# Bonus: ranking em texto, para conferencia rapida e para citar numeros exatos no README
importancia_media = np.abs(valores).mean(axis=0)
ranking = pd.DataFrame({"feature": nomes_features, "importancia_media_abs": importancia_media})
ranking = ranking.sort_values("importancia_media_abs", ascending=False)
log.info(f"Top 10 features por importancia SHAP:\n{ranking.head(10).to_string(index=False)}")
ranking.to_csv("reports/shap_ranking.csv", index=False)