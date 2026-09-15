# src/visualization/gerar_graficos_finais.py
import json
import matplotlib.pyplot as plt
import pandas as pd

with open("reports/metrics.json") as f:
    metrics = json.load(f)

melhor = metrics["melhor_modelo"]
m = metrics["resultados_por_modelo"][melhor]

# Gráfico 1 — métricas finais do modelo vencedor
fig, ax = plt.subplots(figsize=(7, 4))
nomes = ["Acurácia", "Precisão", "Recall", "F1", "ROC-AUC"]
valores = [m["accuracy"], m["precision"], m["recall"], m["f1"], m["roc_auc_teste"]]
ax.bar(nomes, valores, color="#C74634")
ax.set_ylim(0, 1)
ax.set_title(f"Métricas finais — {melhor}")
for i, v in enumerate(valores):
    ax.text(i, v + 0.02, f"{v:.3f}", ha="center")
plt.tight_layout()
plt.savefig("images/metricas_finais.png", dpi=150)
print("Salvo: images/metricas_finais.png")

# Gráfico 2 — ranking SHAP (top 10)
ranking = pd.read_csv("reports/shap_ranking.csv").head(10)
fig, ax = plt.subplots(figsize=(8, 5))
ax.barh(ranking["feature"], ranking["importancia_media_abs"], color="#C74634")
ax.invert_yaxis()
ax.set_title("Top 10 variáveis por importância (SHAP)")
plt.tight_layout()
plt.savefig("images/shap_ranking_top10.png", dpi=150)
print("Salvo: images/shap_ranking_top10.png")