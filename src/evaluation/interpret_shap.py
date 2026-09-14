import logging

import joblib
import matplotlib.pyplot as plt
import pandas as pd
import shap

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
log = logging.getLogger(__name__)

#pipeline = joblib.load("reports/models/logistic_regression_pipeline.joblib")  # ajuste o nome se o vencedor for outro
pipeline = joblib.load("reports/models/random_forest_pipeline.joblib") # ajustado para random_forest depois da visualização do treinamento e ser vencedor
X_test = pd.read_parquet("reports/models/X_test.parquet")

preprocessor = pipeline.named_steps["preprocessor"]
classifier = pipeline.named_steps["classifier"]

X_test_transformado = preprocessor.transform(X_test)
nomes_features = preprocessor.get_feature_names_out()

log.info(f"Calculando SHAP para {X_test_transformado.shape[0]} registros, {len(nomes_features)} features")

explainer = shap.Explainer(classifier, X_test_transformado[:200])
shap_values = explainer(X_test_transformado[:200])
shap_values.feature_names = list(nomes_features)

plt.figure()
shap.summary_plot(shap_values, show=False)
plt.tight_layout()
plt.savefig("images/shap_summary.png", dpi=150)
log.info("Grafico salvo em images/shap_summary.png")