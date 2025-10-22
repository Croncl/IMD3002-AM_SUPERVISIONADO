import pandas as pd
import numpy as np
import os
import joblib
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score
from itertools import product
from joblib import Parallel, delayed

# ==============================================================================
#                 CONFIGURAÇÃO GERAL
# ==============================================================================

# --- Configuração de Caminho Robusta ---
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "bases_geradas")
RESULTS_DIR = SCRIPT_DIR

# Configurações do DTree
CRITERIONS = ["gini", "entropy", "log_loss"]
MAX_DEPTHS = list(range(2, 16))  # 2, 3, ..., 15
TEST_SIZE = 0.3  # Holdout 70% treino, 30% teste
RANDOM_STATE = 42
N_JOBS = -1  # Usa todos os núcleos do processador para paralelismo

# Nomes de Arquivo
BEST_KNN_BASE_NAME = "hog_128_32x32_pca90.csv"
LABEL_COLUMN = "label"
# **NOVAS COLUNAS A IGNORAR:** Adicione todos os nomes de colunas que não são features.
COLUMNS_TO_IGNORE = [LABEL_COLUMN, "filename", "animal", "race", "nome_arquivo"]

# Nomes de Arquivo de Saída
LOG_FILE = os.path.join(RESULTS_DIR, "dtree_best_configs_log.csv")
TOP_10_FILE = os.path.join(RESULTS_DIR, "dtree_top_10_configs.csv")
KNN_BEST_BASES_FILE = os.path.join(
    RESULTS_DIR, "Relatorio_Bases_Ordenadas_15_melhores_KNN.csv"
)


# ==============================================================================
#                 FUNÇÕES DE SUPORTE
# ==============================================================================


def load_data(base_name):
    """Carrega os dados da base específica e limpa colunas não-numéricas."""

    file_path = os.path.join(BASE_DIR, base_name)
    print(f"\nCarregando dados de: {file_path}")

    if not os.path.exists(file_path):
        print(f"Diretório Base Calculado: {BASE_DIR}")
        raise FileNotFoundError(f"Arquivo da base NÃO ENCONTRADO em: {file_path}")

    # Tenta carregar com o separador mais comum (vírgula)
    try:
        df = pd.read_csv(file_path, sep=",")
    except Exception:
        # Tenta carregar com ponto e vírgula se o primeiro falhar
        df = pd.read_csv(file_path, sep=";")

    # Separa Labels (y)
    if LABEL_COLUMN not in df.columns:
        raise ValueError(f"Coluna de label '{LABEL_COLUMN}' não encontrada.")
    y = df[LABEL_COLUMN]

    # **PRINCIPAL CORREÇÃO:** Remove colunas de identificação e seleciona apenas numéricas.
    # 1. Remove as colunas de identificação (filename, animal, etc.)
    cols_to_drop = [col for col in COLUMNS_TO_IGNORE if col in df.columns]
    X_temp = df.drop(columns=cols_to_drop, errors="ignore")

    # 2. Garante que todas as colunas de features sejam numéricas
    # Isso impede que qualquer outra coluna de string cause o erro ValueError.
    X = X_temp.select_dtypes(include=np.number)

    # Remove colunas que podem ter ficado totalmente NaN após a conversão, se necessário.
    X.dropna(axis=1, how="all", inplace=True)

    if X.shape[1] == 0:
        raise ValueError(
            "Nenhuma feature numérica válida encontrada após o pré-processamento."
        )

    print(
        f"Dados carregados: {X.shape[0]} amostras, {X.shape[1]} features (numéricas)."
    )
    return X, y


def load_log():
    """Carrega o arquivo de log incremental ou cria um novo."""
    if os.path.exists(LOG_FILE):
        df_log = pd.read_csv(LOG_FILE, sep=";", decimal=",")
        print(f"Log incremental carregado. {len(df_log)} combinações já testadas.")
        return df_log
    else:
        print("Log incremental não encontrado. Iniciando novo log.")
        return pd.DataFrame(
            columns=["Base", "Criterion", "Max_Depth", "Accuracy", "Tested"]
        )


def save_log(df_log):
    """Salva o log incremental."""
    df_log.to_csv(LOG_FILE, index=False, sep=";", decimal=",")


def train_and_evaluate(
    X_train, X_test, y_train, y_test, base_name, criterion, max_depth
):
    """Treina e avalia a DTree para uma combinação específica."""

    # O seu código do Colab usava f1_score, mas o script atual usa accuracy_score.
    # Vou manter accuracy_score para consistência com o script anterior, mas farei a observação.

    # Treinamento
    dtree = DecisionTreeClassifier(
        criterion=criterion, max_depth=max_depth, random_state=RANDOM_STATE
    )
    dtree.fit(X_train, y_train)

    # Predição e Avaliação (Usando Accuracy para manter consistência)
    y_pred = dtree.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)

    # NOTA: Se preferir usar F1-Score como no seu exemplo do Colab, descomente as linhas abaixo
    # from sklearn.metrics import f1_score
    # score = f1_score(y_test, y_pred, average='macro')
    # e renomeie 'Accuracy' para 'F1_Score' abaixo.

    result = {
        "Base": base_name,
        "Criterion": criterion,
        "Max_Depth": max_depth,
        "Accuracy": accuracy,
        "Tested": 1,
    }

    return result


# ==============================================================================
#                 PROCESSO PRINCIPAL
# ==============================================================================


def run_dtree_grid_search():

    # 1. CARREGAR DADOS
    try:
        X, y = load_data(BEST_KNN_BASE_NAME)
    except Exception as e:
        print(f"❌ Falha crítica ao carregar os dados: {e}")
        return

    # Holdout: Separação 70% treino / 30% teste
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )
    print(
        f"Holdout 70/30: Treino={len(X_train)} amostras, Teste={len(X_test)} amostras."
    )

    # 2. CARREGAR LOG E IDENTIFICAR COMBINAÇÕES PENDENTES
    df_log = load_log()
    all_combinations = list(product(CRITERIONS, MAX_DEPTHS))

    # Cria uma chave única para cada combinação
    df_log["Key"] = df_log["Criterion"] + "_" + df_log["Max_Depth"].astype(str)
    tested_keys = set(df_log[df_log["Base"] == BEST_KNN_BASE_NAME]["Key"].tolist())

    pending_combinations = []
    for criterion, depth in all_combinations:
        key = f"{criterion}_{depth}"
        if key not in tested_keys:
            pending_combinations.append((criterion, depth))

    print(f"Total de combinações a testar: {len(all_combinations)}")
    print(
        f"Combinações pendentes para {BEST_KNN_BASE_NAME}: {len(pending_combinations)}"
    )

    if not pending_combinations:
        print("✅ Todas as combinações já foram testadas. Pulando a execução.")
        new_results = []
    else:
        # 3. EXECUÇÃO PARALELA DAS COMBINAÇÕES PENDENTES
        print(
            f"Iniciando {len(pending_combinations)} testes em paralelo com {N_JOBS} núcleos..."
        )

        # O joblib paraleliza a função de treinamento e avaliação
        new_results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(train_and_evaluate)(
                X_train, X_test, y_train, y_test, BEST_KNN_BASE_NAME, criterion, depth
            )
            for criterion, depth in pending_combinations
        )

    # 4. ATUALIZAR E SALVAR O LOG INCREMENTAL
    if new_results:
        df_new_results = pd.DataFrame(new_results)

        # Concatena os novos resultados
        df_log = pd.concat(
            [df_log.drop(columns=["Key"], errors="ignore"), df_new_results]
        )

        # Remove duplicatas e salva o log atualizado
        df_log.drop_duplicates(
            subset=["Base", "Criterion", "Max_Depth"], keep="last", inplace=True
        )

        save_log(df_log.drop(columns=["Key"], errors="ignore"))
        print(f"\n✅ Log incremental atualizado. Total de testes no log: {len(df_log)}")

    # 5. ORDENAR, SELECIONAR AS TOP 10 E SALVAR ARQUIVO FINAL

    # Filtra apenas os resultados da melhor base do KNN
    df_base_results = df_log[df_log["Base"] == BEST_KNN_BASE_NAME].copy()

    if len(df_base_results) == 0:
        print(
            "\n⚠️ Nenhum resultado encontrado para a base. Não é possível gerar o TOP 10."
        )
        return

    # Ordena pela Acurácia (descendente)
    df_top_10 = (
        df_base_results.sort_values(by="Accuracy", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    # Adicionar coluna de Rank
    df_top_10.index = df_top_10.index + 1
    df_top_10 = df_top_10.rename_axis("Rank").reset_index()

    # Salva o arquivo final das 10 melhores configurações
    df_top_10.to_csv(TOP_10_FILE, index=False, sep=";", decimal=",")

    print("\n" + "=" * 80)
    print(f"🏆 TOP 10 CONFIGURAÇÕES DTree para a Base: {BEST_KNN_BASE_NAME}")
    print("=" * 80)

    # Exibe as 10 melhores configurações
    df_display = df_top_10[["Rank", "Criterion", "Max_Depth", "Accuracy"]].copy()
    df_display["Accuracy"] = df_display["Accuracy"].round(4)
    print(df_display.to_string(index=False))

    print(f"\n✅ Top 10 configurações salvas em: {TOP_10_FILE}")


if __name__ == "__main__":
    run_dtree_grid_search()
