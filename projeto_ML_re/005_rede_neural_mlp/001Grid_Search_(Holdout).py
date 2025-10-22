import pandas as pd
import numpy as np
import os
import ast
import time
from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split, PredefinedSplit
from sklearn.metrics import f1_score
from joblib import Parallel, delayed
from tqdm import tqdm

# ==============================================================================
#                 CONFIGURAÇÃO DE CAMINHOS E DADOS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
# Base usada nos testes anteriores (assumida como a melhor)
BEST_BASE_NAME = "hog_128_32x32_pca90.csv"
# Caminho para o diretório onde estão os arquivos CSV (geralmente '../bases_geradas')
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "bases_geradas")
BEST_BASE_PATH = os.path.join(BASE_DIR, BEST_BASE_NAME)

# Arquivo de log incremental e de saída (Top 10)
LOG_FILE = "mlp_holdout_gridsearch_log.csv"
TOP10_OUTPUT_FILE = "top10_mlp_configs_holdout.csv"

RANDOM_STATE = 42
TEST_SIZE = 0.3  # Holdout 70% treino, 30% teste
N_JOBS = -1  # Usa todos os núcleos

# Colunas a ignorar (as mesmas usadas nos scripts anteriores)
LABEL_COLUMN = "label"
COLUMNS_TO_IGNORE = [LABEL_COLUMN, "filename", "animal", "race", "nome_arquivo"]

# ==============================================================================
#                 FUNÇÕES DE SUPORTE
# ==============================================================================


def load_data():
    """Carrega os dados da melhor base e prepara para o Holdout."""
    if not os.path.exists(BEST_BASE_PATH):
        raise FileNotFoundError(f"Base não encontrada: {BEST_BASE_PATH}")

    try:
        df = pd.read_csv(BEST_BASE_PATH, sep=",")
    except Exception:
        df = pd.read_csv(BEST_BASE_PATH, sep=";")

    # 1. Preparar X e y
    y = df[LABEL_COLUMN]
    cols_to_drop = [col for col in COLUMNS_TO_IGNORE if col in df.columns]
    X_temp = df.drop(columns=cols_to_drop, errors="ignore")
    X = X_temp.select_dtypes(include=np.number)
    X.dropna(axis=1, how="all", inplace=True)

    if X.shape[1] == 0:
        raise ValueError("Nenhuma feature numérica válida encontrada.")

    # 2. Split Holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Preparar para PredefinedSplit (como no seu código)
    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)

    # -1 para treino, 0 para teste
    test_fold = np.concatenate(
        (-1 * np.ones(len(X_train), dtype=int), 0 * np.ones(len(X_test), dtype=int))
    )
    ps = PredefinedSplit(test_fold)

    print(f"✅ Base '{BEST_BASE_NAME}' carregada.")
    print(
        f"Dados: {X.shape[0]} amostras, {X.shape[1]} features, {len(np.unique(y))} classes."
    )
    return X_full, y_full, ps, X_train.shape[1], len(np.unique(y_train))


def generate_param_grid(n_features, n_classes):
    """Gera a lista de todas as combinações de parâmetros."""
    hidden_layer_configs = [
        (n_features,),
        (n_features, n_features),
        (n_features, n_features, n_features),
        ((n_features + 1) // 2, (n_features + 1) // 2),
        ((n_features + 2) // 3, (n_features + 2), (n_features + 2)),
        (
            (n_features + 3) // 4,
            (n_features + 3) // 4,
            (n_features + 3) // 4,
            (n_features + 3) // 4,
        ),
        (n_features + n_classes,),
        ((n_features + n_classes) // 2,),
        (100,),
        (100, 50),
        (100, 100, 50),
        (100, 100, 100, 50),
    ]

    param_grid = []
    for hls in hidden_layer_configs:
        for activation in ["identity", "logistic", "tanh", "relu"]:
            for solver in ["sgd", "adam"]:
                for lr in [0.0001, 0.001, 0.01, 0.1]:
                    for max_iter in [500, 1000, 1500, 2000, 2500, 3000]:
                        param_grid.append(
                            {
                                "hidden_layer_sizes": hls,
                                "activation": activation,
                                "solver": solver,
                                "learning_rate_init": lr,
                                "max_iter": max_iter,
                                "early_stopping": False,  # Desliga o early stopping para consistência
                                "random_state": RANDOM_STATE,
                            }
                        )
    return param_grid

def evaluate_config(params, X_full, y_full, ps):
    """Função para ser executada em paralelo."""

    modelo = MLPClassifier(**params)

    try:
        # Apenas um split no PredefinedSplit (Treino no -1, Teste no 0)
        for train_idx, test_idx in ps.split():
            # Treinamento
            modelo.fit(X_full[train_idx], y_full[train_idx])

            # Predição e Score
            y_pred = modelo.predict(X_full[test_idx])
            score = f1_score(y_full[test_idx], y_pred, average="weighted")

            return {"params": params, "f1_score": score, "status": "OK"}

    except Exception as e:
        return {
            "params": params,
            "f1_score": None,
            "status": "ERRO",
            "erro_msg": str(e),
        }


# ==============================================================================
#                 PROCESSO PRINCIPAL (Ajustado)
# ==============================================================================


def run_mlp_grid_search():
    try:
        X_full, y_full, ps, n_features, n_classes = load_data()
    except Exception as e:
        print(f"❌ Erro Crítico ao carregar dados: {e}")
        return

    param_grid = generate_param_grid(n_features, n_classes)
    total_configs = len(param_grid)
    print(f"🔧 Total de combinações a testar: {total_configs}")

    # 1. Carregar Log Incremental (Carrega como string para evitar o erro inicial)
    df_log = pd.DataFrame()  # Inicia um DF vazio
    log_file_exists = os.path.exists(LOG_FILE)

    if log_file_exists:
        # Tenta carregar o log, mas sem conversão automática de 'params' (mantendo como string)
        # O ast.literal_eval falhava ao carregar o log. Vamos carregar a string falha
        try:
            # Força a leitura de todas as colunas como string, exceto as numéricas
            df_log = pd.read_csv(
                LOG_FILE,
                sep=";",
                decimal=",",
                dtype={
                    "f1_score": float,
                    "params": str,
                    "status": str,
                    "erro_msg": str,
                },
            )
            print(
                f"💾 Log incremental carregado com {len(df_log)} resultados (params como string)."
            )
        except Exception:
            df_log = pd.DataFrame()
            print("⚠️ Log incremental vazio/malformado. Iniciando novo log.")
    else:
        print("🆕 Iniciando novo log incremental.")

    # Flag para saber se precisamos rodar o loop (Se todas já foram rodadas, pulamos 2 e 3)
    skip_execution = len(df_log) == total_configs

    # 2. Identificar Configurações Pendentes
    configs_pendentes = []

    if not skip_execution:
        # Se for um log antigo e for falho, a conversão de params pode falhar aqui.
        # Por isso, na Etapa 1, carregamos 'params' como string para evitar o erro.

        tested_params_str = set()

        if not df_log.empty and "params" in df_log.columns:
            # Tenta converter o 'params' para o dicionário para a comparação, usando a conversão que falhou antes
            # Se falhar aqui, o usuário deve ser instruído a apagar o log
            try:
                # É mais seguro recarregar a coluna 'params' do log como dicionário APENAS para a lógica de 'pendentes'
                df_temp = df_log.copy()
                df_temp["params"] = df_temp["params"].apply(
                    ast.literal_eval
                )  # Aqui é onde falhava
                tested_params_str = set(df_temp["params"].astype(str))
            except Exception:
                print(
                    "❌ ERRO: O arquivo de log antigo está em um formato que o AST não consegue ler."
                )
                print(
                    "Por favor, **apague o arquivo 'mlp_holdout_gridsearch_log.csv'** e execute novamente."
                )
                return

        for params in param_grid:
            if str(params) not in tested_params_str:
                configs_pendentes.append(params)

    num_pendentes = len(configs_pendentes)

    if skip_execution or num_pendentes == 0:
        print("⏩ Todas as configurações já foram testadas. Pulando a execução.")
        df_new_results = pd.DataFrame()  # DataFrame vazio
    else:
        # 3. Execução Paralela
        print(
            f"🚀 Iniciando avaliação de {num_pendentes} combinações pendentes com {N_JOBS} núcleos..."
        )

        start_time = time.time()

        results = Parallel(n_jobs=N_JOBS, verbose=10)(
            delayed(evaluate_config)(params, X_full, y_full, ps)
            for params in configs_pendentes
        )

        end_time = time.time()
        print(f"✅ Execução paralela concluída. ⏱️ Tempo: {end_time - start_time:.2f}s")

        df_new_results = pd.DataFrame(results)

    # 4. Consolidar e Salvar Log
    if not df_new_results.empty:
        # Converte o dicionário de parâmetros para string para salvar no CSV
        df_new_results["params"] = df_new_results["params"].astype(str)

        # Concatena com o log existente.
        if df_log.empty:
            df_log = df_new_results.copy()
        else:
            # df_log.params já é string (do carregamento forçado)
            df_log = pd.concat([df_log, df_new_results], ignore_index=True)

        df_log.to_csv(LOG_FILE, index=False, sep=";", decimal=",")
        print(f"💾 Log incremental atualizado. Total de testes no log: {len(df_log)}")

    # 5. Selecionar Top 10 e Salvar (Onde o ast.literal_eval é necessário)
    df_final = df_log.copy()

    # Se o log está vazio (por exemplo, erro na execução e log vazio), não há nada a fazer
    if df_final.empty:
        print("⚠️ Log de resultados vazio. Não é possível gerar o Top 10.")
        return

    # Garante que 'params' seja um dicionário para a ordenação
    if df_final["params"].dtype == object:
        try:
            # ❗ CORREÇÃO: Tenta converter o 'params' para o dicionário AQUI
            df_final["params"] = df_final["params"].apply(ast.literal_eval)
        except Exception:
            print(
                "❌ ERRO NO TOP 10: O arquivo de log contém strings de parâmetros em formato inválido para ast.literal_eval."
            )
            print(
                "Por favor, **apague o arquivo 'mlp_holdout_gridsearch_log.csv'** e execute novamente. Não é possível prosseguir com o log corrompido."
            )
            return

    # Filtra, ordena e seleciona o Top 10
    df_validos = df_final[df_final["f1_score"].notnull()]
    top10 = (
        df_validos.sort_values(by="f1_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    # Salva o Top 10 para o próximo script
    top10_save = top10.copy()
    top10_save["params"] = top10_save["params"].astype(str)
    top10_save.to_csv(TOP10_OUTPUT_FILE, index=False)

    print("\n" + "=" * 50)
    print(f"📋 Top 10 configurações salvas em: {TOP10_OUTPUT_FILE}")

    # Formata para exibição
    df_display = top10.copy()

    # Adiciona uma representação simplificada dos parâmetros
    df_display["HLS"] = df_display["params"].apply(
        lambda x: str(x["hidden_layer_sizes"])
    )
    df_display["Act"] = df_display["params"].apply(lambda x: x["activation"])
    df_display["LR"] = df_display["params"].apply(lambda x: x["learning_rate_init"])
    df_display["Iter"] = df_display["params"].apply(lambda x: x["max_iter"])
    df_display["solver"] = df_display["params"].apply(lambda x: x["solver"])

    # Exibe as colunas mais importantes
    print(
        df_display[["f1_score", "HLS", "Act", "LR", "Iter", "solver"]]
        .head(10)
        .to_string(index=False)
    )
    print("=" * 50)


if __name__ == "__main__":
    run_mlp_grid_search()
