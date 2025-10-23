import os
import time
import pandas as pd
import ast
import numpy as np
from tqdm import tqdm
from joblib import Parallel, delayed
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import KFold, cross_val_score

# ==============================================================================
#                 CONFIGURAÇÃO DE CAMINHOS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Diretório onde estão os arquivos CSV das bases
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "bases_geradas")
# Diretório onde está o arquivo de bases do KNN
KNN_DIR = os.path.join(SCRIPT_DIR, "..", "002_KNN_acuracia")

# Arquivos de entrada e saída
# --- Caminho do Arquivo de Bases KNN ---
KNN_BEST_BASES_FILE = os.path.join(
    SCRIPT_DIR,
    "..",
    "Relatorio_Bases_Ordenadas_KNN.csv",
)


# ❗ NOVO CAMINHO: Aponta para o arquivo CSV de Top 10 gerado localmente
TOP_10_CONFIGS_FILE = os.path.join(SCRIPT_DIR, "top10_mlp_configs_holdout.csv")

OUTPUT_CSV = "resultados_mlp_15bases_10fold_cv_inc.csv"

# ==============================================================================
#                 FUNÇÕES DE CARREGAMENTO INICIAL
# ==============================================================================


# 📂 Carregar a lista das 15 melhores bases do KNN
def load_base_list():
    """Carrega a lista dos nomes dos arquivos das 15 melhores bases do KNN."""
    try:
        if not os.path.exists(KNN_BEST_BASES_FILE):
            raise FileNotFoundError(
                f"Arquivo de bases não encontrado em: {KNN_BEST_BASES_FILE}"
            )

        df_bases = pd.read_csv(KNN_BEST_BASES_FILE, sep=";", decimal=",")

        df_bases = df_bases.head(15)  # Seleciona as 15 primeiras linhas
        # Garante que a coluna 'Base' existe e carrega os nomes
        if "Base" not in df_bases.columns:
            raise ValueError("Coluna 'Base' não encontrada no arquivo de bases KNN.")

        bases_escolhidas = df_bases["Base"].tolist()
        print(
            f"✅ Lista das 15 melhores bases do KNN carregada: {len(bases_escolhidas)} bases."
        )
        return bases_escolhidas
    except Exception as e:
        print(f"❌ Erro ao carregar a lista de bases KNN: {e}")
        return []


# 📥 Carregar os melhores parâmetros do Grid Search Local
def load_mlp_configs():
    """Carrega as Top 10 configurações e seus scores Holdout do Grid Search local."""
    try:
        if not os.path.exists(TOP_10_CONFIGS_FILE):
            raise FileNotFoundError(
                f"Arquivo de configurações MLP não encontrado: {TOP_10_CONFIGS_FILE}. Execute o Grid Search Holdout primeiro."
            )

        # O arquivo de saída do Grid Search Holdout usa separador padrão (vírgula)
        df_top10 = pd.read_csv(TOP_10_CONFIGS_FILE, sep=",")

        # 'params' está salvo como string (ast.literal_eval)
        df_top10["params_dict"] = df_top10["params"].apply(ast.literal_eval)

        # Retorna uma lista de dicionários contendo a config e o score Holdout
        melhores_parametros_com_score = df_top10.apply(
            lambda row: {
                "config": row["params_dict"],
                "holdout_score": row["f1_score"],
            },
            axis=1,
        ).tolist()

        print(f"✅ Top 10 Configurações MLP e scores Holdout carregados localmente.")
        return melhores_parametros_com_score
    except Exception as e:
        print(f"❌ Erro ao carregar configurações MLP: {e}")
        return []


# AQUI MUDAMOS O QUE CARREGAMOS:
base_configs_list = load_mlp_configs()
bases_escolhidas = load_base_list()

# Extrai apenas as configs para o loop de avaliação
melhores_parametros = [item["config"] for item in base_configs_list]
holdout_scores_map = {
    str(item["config"]): item["holdout_score"] for item in base_configs_list
}


# ==============================================================================
#                 FUNÇÕES DE SUPORTE
# ==============================================================================


# 🧹 Função para preparar os dados
def preparar_dados(df):
    """Limpa e separa features numéricas (X) e labels (y)."""
    # Colunas a ignorar (as mesmas usadas nos scripts anteriores)
    cols_to_drop = ["filename", "animal", "label", "race", "nome_arquivo"]

    y = df["label"]

    # Remove colunas não-features e garante que X contenha apenas numéricos
    X_temp = df.drop(
        columns=[col for col in cols_to_drop if col in df.columns], errors="ignore"
    )
    X = X_temp.select_dtypes(include=np.number)
    X.dropna(axis=1, how="all", inplace=True)

    return X, y


# 🧠 Função para avaliar uma configuração (mantida a lógica de f1_weighted)
def avaliar_config(config, X, y, nome_base, n_features, kfold):
    """Treina e avalia o MLP usando K-Fold CV."""
    config_completo = config.copy()

    # early_stopping=False é mantido conforme a sua lógica original
    if "early_stopping" not in config_completo:
        config_completo["early_stopping"] = False

    # Trata 'hidden_layer_sizes' que joblib pode ter serializado como tupla/string
    if "hidden_layer_sizes" in config_completo and isinstance(
        config_completo["hidden_layer_sizes"], str
    ):
        config_completo["hidden_layer_sizes"] = ast.literal_eval(
            config_completo["hidden_layer_sizes"]
        )

    # Remove 'random_state' se estiver presente, para usar o do classificador
    config_completo.pop("random_state", None)

    modelo = MLPClassifier(**config_completo, random_state=42)
    inicio = time.time()

    # Cross-validation
    scores = cross_val_score(
        modelo, X, y, cv=kfold, scoring="f1_weighted", n_jobs=1
    )  # n_jobs=1 para não aninhar joblib

    fim = time.time()

    score_mean = scores.mean()
    score_std = scores.std()

    print(
        f"✅ {nome_base} ({n_features} features) | F1 Médio: {score_mean:.4f} ± {score_std:.4f} | ⏱️ Tempo: {fim - inicio:.2f}s"
    )

    return {
        "base": nome_base,
        "n_features": n_features,
        "config": config_completo,
        "f1_score_medio": score_mean,
        "f1_score_std": score_std,
    }


# ==============================================================================
#                 PROCESSO PRINCIPAL
# ==============================================================================

# 📊 Verificar se já existe resultados salvos (log incremental)
if os.path.exists(OUTPUT_CSV):
    df_kfold = pd.read_csv(OUTPUT_CSV)
    # A coluna 'config' deve ser convertida de volta para dicionário para comparação
    if not df_kfold.empty and "config" in df_kfold.columns:
        df_kfold["config"] = df_kfold["config"].apply(ast.literal_eval)
        print(f"💾 Log incremental carregado com {len(df_kfold)} resultados.")
    else:
        df_kfold = pd.DataFrame()
        print("⚠️ Log incremental vazio/malformado. Iniciando novo log.")
else:
    df_kfold = pd.DataFrame()
    print("🆕 Iniciando novo log incremental.")


# 🔁 Avaliar todas as bases
for nome_base in tqdm(bases_escolhidas, desc="🔁 Avaliando as 15 Bases do KNN"):
    file_path = os.path.join(BASE_DIR, nome_base)

    if not os.path.exists(file_path):
        print(f"\n❌ Arquivo '{nome_base}' não encontrado em {BASE_DIR}. Pulando.")
        continue

    # Tenta ler com diferentes separadores (como no script anterior)
    try:
        df = pd.read_csv(file_path, sep=",")
    except Exception:
        try:
            df = pd.read_csv(file_path, sep=";")
        except Exception:
            print(f"\n⚠️ Base {nome_base} pulada: Erro ao carregar o CSV.")
            continue

    X_base, y_base = preparar_dados(df)
    n_features_base = X_base.shape[1]

    if n_features_base == 0:
        print(f"\n⚠️ Base {nome_base} pulada: Nenhuma feature numérica válida.")
        continue

    kfold = KFold(n_splits=10, shuffle=True, random_state=42)

    # 🔍 Filtrar configs ainda não avaliadas (Lógica de Log Incremental)
    configs_nao_avaliadas = []

    # Prepara o log para a comparação (apenas se não estiver vazio)
    if not df_kfold.empty:
        # Coluna auxiliar para comparação de strings
        df_kfold["config_str"] = df_kfold["config"].apply(lambda x: str(x))

    for config in melhores_parametros:
        config_completo = config.copy()

        # Garante que early_stopping seja False e remove random_state para a verificação
        config_completo.pop("random_state", None)  # Remove random_state do Grid Search
        if "early_stopping" not in config_completo:
            config_completo["early_stopping"] = False

        # Garante que hidden_layer_sizes seja tupla/lista para consistência na string
        if isinstance(config_completo.get("hidden_layer_sizes"), list):
            config_completo["hidden_layer_sizes"] = tuple(
                config_completo["hidden_layer_sizes"]
            )

        config_str_to_check = str(config_completo)

        ja_avaliado = False
        if not df_kfold.empty:
            ja_avaliado = (
                (df_kfold["base"] == nome_base)
                & (df_kfold["config_str"] == config_str_to_check)
            ).any()

        if not ja_avaliado:
            configs_nao_avaliadas.append(config)

    # Remove a coluna auxiliar do log (apenas se foi criada)
    if not df_kfold.empty and "config_str" in df_kfold.columns:
        df_kfold.drop(columns=["config_str"], inplace=True)

    if not configs_nao_avaliadas:
        continue

    # ⚡ Executar em paralelo
    print(
        f"\n⚡ Executando {len(configs_nao_avaliadas)} configurações para {nome_base}..."
    )
    novos_resultados = Parallel(n_jobs=-1)(
        delayed(avaliar_config)(
            config,
            X_base,
            y_base,
            nome_base,
            n_features_base,
            kfold,
        )
        for config in configs_nao_avaliadas
    )

    # 💾 Salvar após cada base
    df_novos = pd.DataFrame(novos_resultados)

    # Antes de concatenar, converte a coluna 'config' de dicionário para string no df_novos
    df_novos["config"] = df_novos["config"].astype(str)

    # Converte a coluna 'config' do log existente para string antes de concatenar (se não estiver vazio)
    if not df_kfold.empty:
        df_kfold["config"] = df_kfold["config"].astype(str)

    df_kfold = pd.concat([df_kfold, df_novos], ignore_index=True)
    df_kfold.to_csv(OUTPUT_CSV, index=False)

    # Volta 'config' para dicionário no log para o próximo loop (se houver)
    if not df_kfold.empty:
        df_kfold["config"] = df_kfold["config"].apply(ast.literal_eval)

# ==============================================================================
#                 PROCESSAMENTO FINAL, PIVOTAGEM E ORDENAÇÃO
# ==============================================================================

# AQUI ESTÁ A MAIOR MUDANÇA: GERAÇÃO DA TABELA FINAL

if not df_kfold.empty:

    print("\n📊 Gerando tabela de resultados final...")

    # 1. Normaliza a coluna de configuração para string padronizada
    def normalize_config_for_map(cfg_dict):
        # Cria uma cópia para evitar SettingWithCopyWarning
        cfg_dict_copy = cfg_dict.copy()
        cfg_dict_copy.pop("random_state", None)
        if "early_stopping" not in cfg_dict_copy:
            cfg_dict_copy["early_stopping"] = False
        if "hidden_layer_sizes" in cfg_dict_copy and isinstance(
            cfg_dict_copy["hidden_layer_sizes"], list
        ):
            cfg_dict_copy["hidden_layer_sizes"] = tuple(
                cfg_dict_copy["hidden_layer_sizes"]
            )
        return cfg_dict_copy

    # Cria uma coluna de string normalizada para o pivô
    df_kfold["config_key"] = (
        df_kfold["config"].apply(normalize_config_for_map).astype(str)
    )

    # 2. Pivota os dados (Formato final: Bases x Configurações)
    df_pivot = df_kfold.pivot_table(
        index="base",
        columns="config_key",
        values="f1_score_medio",
        aggfunc="first",  # Pega o primeiro (e único) score de cada Base/Config
    )

    # 3. Adiciona as linhas do Holdout (70/30) para a base 'hog_128_32x32_pca90.csv'

    # Cria uma linha para o Holdout (apenas para a base que foi usada no Holdout)
    holdout_row_data = {}
    base_holdout_name = "hog_128_32x32_pca90.csv"

    # Mapeia as chaves de configuração no df_pivot para os scores Holdout
    for col in df_pivot.columns:
        # A coluna 'col' é a string da config normalizada.
        # Procuramos o score dessa config no nosso mapa
        if col in holdout_scores_map:
            holdout_row_data[col] = holdout_scores_map[col]

    # Cria o DataFrame Holdout (que terá apenas uma linha, mas com todas as 10 colunas de config)
    df_holdout = pd.DataFrame([holdout_row_data])

    # Define o índice para o nome da base holdout
    df_holdout.index = pd.Index([base_holdout_name], name="base")

    # Garante que as colunas do df_holdout correspondam exatamente às colunas do df_pivot
    df_holdout = df_holdout.reindex(columns=df_pivot.columns)

    # Concatena a linha Holdout com os resultados K-Fold
    # Usamos o resultado do KFold para a base Holdout também, pois ele já foi calculado

    # 4. Cria o Tabela Final (com a coluna de tipo de avaliação)

    # Adiciona a linha de avaliação (Holdout 70/30)
    # 🚨 NOTA: O Holdout só foi feito na base BEST_BASE_NAME, então o score só faz sentido para ela.
    # O valor Holdout é o f1_score original do top10_mlp_configs_holdout.csv

    df_holdout_scores = pd.DataFrame(index=[base_holdout_name])
    for config_key_str, score in holdout_scores_map.items():
        df_holdout_scores.loc[base_holdout_name, config_key_str] = score

    df_holdout_scores.index.name = "Base"
    df_holdout_scores["Tipo_Avaliacao"] = "Holdout 70/30"

    # Adiciona a linha de avaliação K-Fold (para todas as 15 bases)
    df_pivot.index.name = "Base"
    df_pivot["Tipo_Avaliacao"] = "10Fold CV"

    # Junta as linhas
    df_final_report = pd.concat(
        [df_holdout_scores, df_pivot.reset_index()], ignore_index=True
    )

    # Remove as linhas Holdout de bases que não são a BEST_BASE_NAME
    df_final_report = df_final_report.drop_duplicates(subset=["Base", "Tipo_Avaliacao"])

    # Reordena as colunas
    cols = ["Base", "Tipo_Avaliacao"] + [
        col for col in df_final_report.columns if col not in ["Base", "Tipo_Avaliacao"]
    ]
    df_final_report = df_final_report[cols]

    # 5. Salva a Tabela Final
    FINAL_REPORT_CSV = "Relatorio_MLP_Final_15Bases.csv"
    df_final_report.to_csv(FINAL_REPORT_CSV, index=False, sep=";", decimal=",")

    print(f"\n✅ Relatório final pivotado salvo em: {FINAL_REPORT_CSV}")
    print("\n" + df_final_report.head(30).to_string(index=False))

else:
    print("\n⚠️ Nenhuma base foi processada. Verifique os caminhos dos arquivos.")
