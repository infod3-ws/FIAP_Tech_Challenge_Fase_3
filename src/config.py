import os

# Pode sobrescrever via variável de ambiente: export FIAP_GCP_PROJECT=outro-projeto
BILLING_PROJECT = os.getenv("FIAP_GCP_PROJECT", "carbide-ratio-502315-a2")
BUCKET_NAME     = os.getenv("FIAP_GCS_BUCKET", "fiap-tc2-datalake")

# Mapa oficial IBGE: 2 primeiros dígitos do código de município -> sigla da UF
# (dado público, estável — não depende de nenhuma tabela externa)
CODIGO_UF_PARA_SIGLA = {
    "11": "RO", "12": "AC", "13": "AM", "14": "RR", "15": "PA", "16": "AP", "17": "TO",
    "21": "MA", "22": "PI", "23": "CE", "24": "RN", "25": "PB", "26": "PE", "27": "AL",
    "28": "SE", "29": "BA", "31": "MG", "32": "ES", "33": "RJ", "35": "SP",
    "41": "PR", "42": "SC", "43": "RS", "50": "MS", "51": "MT", "52": "GO", "53": "DF",
}