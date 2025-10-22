import pandas as pd
import numpy as np
import os
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from joblib import Parallel, delayed
from tqdm import tqdm

# ==============================================================================
#                 CONFIGURAÇÃO GERAL
# ==============================================================================

# --- Configuração de Caminho Robusta ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "bases_geradas")
RESULTS_DIR = SCRIPT_DIR

# --- Caminho do Arquivo de Bases KNN ---
KNN_BEST_BASES_FILE = os.path.join(
    SCRIPT_DIR,
    "..",
    "Relatorio_Bases_Ordenadas_KNN.csv",
)



# Nomes de Arquivo de Entrada
TOP_10_CONFIGS_FILE = os.path.join(RESULTS_DIR, "dtree_top_10_configs.csv")

# Configurações
TEST_SIZE = 0.3
N_SPLITS_CV = 10
RANDOM_STATE = 42
N_JOBS = -1

# Colunas a ignorar
LABEL_COLUMN = "label"
COLUMNS_TO_IGNORE = [LABEL_COLUMN, "filename", "animal", "race", "nome_arquivo"]

# Nome do Arquivo de Saída
OUTPUT_COMPARATIVE_FILE = os.path.join(
    RESULTS_DIR, "dtree_comparativo_15_bases_top10_config.csv"
)

# ==============================================================================
#                 FUNÇÕES DE SUPORTE
# ==============================================================================


def load_data_and_configs():
    """Carrega as 15 bases e as 10 melhores configurações."""

    if not os.path.exists(KNN_BEST_BASES_FILE):
        raise FileNotFoundError(
            f"Arquivo de bases não encontrado: {KNN_BEST_BASES_FILE}"
        )

    # IMPORTANTE: Garante o separador correto para carregar o CSV de bases.
    df_bases = pd.read_csv(KNN_BEST_BASES_FILE, sep=";", decimal=",")
    
    df_bases = df_bases.head(15)  # Seleciona as 15 primeiras linhas


    if "Base" not in df_bases.columns:
        raise ValueError(f"Coluna 'Base' não encontrada em {KNN_BEST_BASES_FILE}.")

    base_names = df_bases["Base"].tolist()
    print(f"✅ 15 melhores bases carregadas: {len(base_names)} bases.")

    if not os.path.exists(TOP_10_CONFIGS_FILE):
        raise FileNotFoundError(
            f"Arquivo de configurações não encontrado: {TOP_10_CONFIGS_FILE}."
        )

    # IMPORTANTE: Garante o separador correto para carregar o CSV de configs.
    # O seu arquivo de configs usa ';'.
    df_configs = pd.read_csv(TOP_10_CONFIGS_FILE, sep=";", decimal=",")

    # Cria a lista de configs e o DataFrame de configs
    configs_list = df_configs[["Criterion", "Max_Depth"]].to_dict("records")
    print(
        f"✅ Top 10 Configurações DTree carregadas: {len(configs_list)} configurações."
    )

    return base_names, configs_list, df_configs


def load_base_data(base_name):
    # ... (função load_base_data permanece a mesma) ...
    file_path = os.path.join(BASE_DIR, base_name)

    if not os.path.exists(file_path):
        print(f"⚠️ Base '{base_name}' não encontrada em {file_path}. Pulando.")
        return None, None

    try:
        df = pd.read_csv(file_path, sep=",")
    except Exception:
        df = pd.read_csv(file_path, sep=";")

    if LABEL_COLUMN not in df.columns:
        print(
            f"⚠️ Coluna de label '{LABEL_COLUMN}' não encontrada em {base_name}. Pulando."
        )
        return None, None
    y = df[LABEL_COLUMN]

    cols_to_drop = [col for col in COLUMNS_TO_IGNORE if col in df.columns]
    X_temp = df.drop(columns=cols_to_drop, errors="ignore")
    X = X_temp.select_dtypes(include=np.number)
    X.dropna(axis=1, how="all", inplace=True)

    if X.shape[1] == 0:
        print(f"⚠️ Nenhuma feature numérica válida encontrada em {base_name}. Pulando.")
        return None, None

    return X, y


# ❗ FUNÇÃO DE AVALIAÇÃO CORRIGIDA PARA PIVOTAGEM
def evaluate_config(X, y, base_name, config_rank, config):
    """
    Avalia uma configuração DTree e retorna resultados em formato "longo"
    (Score e Config_ID) para facilitar a pivotagem.
    """
    criterion = config["Criterion"]
    max_depth = config["Max_Depth"]
    config_id = f"Config_{config_rank}"

    clf = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, random_state=RANDOM_STATE
    )

    # 1. Avaliação Holdout (70/30)
    accuracy_holdout = np.nan
    if len(y.unique()) > 1 and all(y.value_counts() >= 2):
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
            )
            clf.fit(X_train, y_train)
            y_pred = clf.predict(X_test)
            accuracy_holdout = accuracy_score(y_test, y_pred)
        except Exception:
            pass

    result_holdout = {
        "Base": base_name,
        "Metodologia": "Holdout (70/30)",
        "Config_ID": config_id,  # COLUNA DE CHAVE PARA PIVOTAGEM
        "Score": accuracy_holdout,  # COLUNA DE VALOR
        "Criterion": criterion,
        "Max_Depth": max_depth,
    }

    # 2. Avaliação Cross-Validation (10-Fold)
    accuracy_cv = np.nan
    try:
        if len(y) >= N_SPLITS_CV and all(y.value_counts() >= N_SPLITS_CV):
            cv_scores = cross_val_score(
                clf,
                X,
                y,
                cv=KFold(n_splits=N_SPLITS_CV, shuffle=True, random_state=RANDOM_STATE),
                scoring="accuracy",
                n_jobs=1,
            )
            accuracy_cv = np.mean(cv_scores)
    except Exception:
        pass

    result_cv = {
        "Base": base_name,
        "Metodologia": f"{N_SPLITS_CV}-Fold CV",
        "Config_ID": config_id,  # COLUNA DE CHAVE PARA PIVOTAGEM
        "Score": accuracy_cv,  # COLUNA DE VALOR
        "Criterion": criterion,
        "Max_Depth": max_depth,
    }

    return result_holdout, result_cv


# ==============================================================================
#                 PROCESSO PRINCIPAL
# ==============================================================================


def run_comparative_dtree_evaluation():

    print("--- INICIANDO AVALIAÇÃO COMPARATIVA DTREE ---")

    try:
        base_names, configs_list, df_configs = load_data_and_configs()
    except Exception as e:
        print(f"❌ ERRO CRÍTICO AO CARREGAR ARQUIVOS: {e}")
        return

    all_results = []

    for base_name in tqdm(base_names, desc="Processando Bases"):
        X, y = load_base_data(base_name)
        if X is None:
            continue

        tasks = []
        for rank, config in enumerate(configs_list, 1):
            tasks.append(delayed(evaluate_config)(X, y, base_name, rank, config))

        results_for_base = Parallel(n_jobs=N_JOBS, verbose=0)(tasks)

        for result_holdout, result_cv in results_for_base:
            all_results.append(result_holdout)
            all_results.append(result_cv)

    if not all_results:
        print(
            "❌ Nenhuma avaliação foi concluída com sucesso. O arquivo de saída ficará vazio."
        )
        return

    # 3. CONSTRUÇÃO DA TABELA FINAL

    df_all = pd.DataFrame(all_results)

    # ❗ PIVOTAGEM CORRIGIDA
    df_pivot = df_all.pivot_table(
        index=["Base", "Metodologia"],
        columns="Config_ID",  # Coluna Config_ID (e.g., 'Config_1')
        values="Score",  # Coluna Score (valor da acurácia)
        aggfunc="first",  # Usa o primeiro (e único) score
    )

    if df_pivot.empty:
        print(
            "❌ ERRO: O DataFrame pivotado está vazio após a pivotagem. Verifique os dados brutos."
        )
        return

    # Reordena as colunas
    config_cols = [f"Config_{i}" for i in range(1, len(configs_list) + 1)]
    df_final = df_pivot.reset_index()

    # Garante que todas as 10 colunas existam (adiciona NaNs se alguma faltar)
    for col in config_cols:
        if col not in df_final.columns:
            df_final[col] = np.nan

    df_final = df_final[["Base", "Metodologia"] + config_cols]

    # Ordena as linhas para manter a ordem original das 15 bases
    base_order = {base: i for i, base in enumerate(base_names)}
    df_final["Order"] = df_final["Base"].map(base_order)
    df_final.sort_values(by=["Order", "Metodologia"], inplace=True)
    df_final.drop(columns=["Order"], inplace=True)

    # 4. SALVAR A TABELA FINAL

    # Mapeamento para o novo cabeçalho formatado
    df_config_map = df_configs.set_index(pd.RangeIndex(1, len(df_configs) + 1))[
        ["Criterion", "Max_Depth"]
    ].T.to_dict("dict")

    new_columns = {"Base": "Base", "Metodologia": "Metodologia"}
    for i in range(1, len(configs_list) + 1):
        conf_col = f"Config_{i}"
        if i in df_config_map:
            conf = df_config_map[i]
            # Formato: Config_1 (gini/13)
            new_columns[conf_col] = (
                f"{conf_col} ({conf['Criterion']}/{conf['Max_Depth']})"
            )
        else:
            new_columns[conf_col] = conf_col

    df_final.rename(columns=new_columns, inplace=True)

    # Salva o resultado final
    df_final.to_csv(OUTPUT_COMPARATIVE_FILE, index=False, sep=";", decimal=",")

    print("\n" + "=" * 100)
    print(f"✅ AVALIAÇÃO CONCLUÍDA. Relatório salvo em: {OUTPUT_COMPARATIVE_FILE}")
    print("--- Exemplo das primeiras linhas do resultado final ---")

    # Exibe o resultado formatado
    df_display = df_final.copy()
    for col in df_display.columns:
        if "Config" in col and df_display[col].dtype in ["float64", "float32"]:
            df_display[col] = df_display[col].round(4)

    print(df_display.head(30).to_string(index=False))
    print("=" * 100)


if __name__ == "__main__":
    run_comparative_dtree_evaluation()
