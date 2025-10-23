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
import math

# ==============================================================================
#                      CONFIGURAÇÃO DE CAMINHOS E DADOS
# ==============================================================================

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))

# Diretório onde estão os arquivos CSV das bases
BASE_DIR = os.path.join(SCRIPT_DIR, "..", "bases_geradas")

# Diretório de saída organizado com nome descritivo
RESULTS_DIR = os.path.join(SCRIPT_DIR, "resultados_004_gridsearch_varias_bases")
os.makedirs(RESULTS_DIR, exist_ok=True)  # Cria a pasta se não existir

# Templates de arquivos de saída dentro da pasta de resultados
LOG_FILE_TEMPLATE = os.path.join(
    RESULTS_DIR, "mlp_holdout_gridsearch_log_{BASE_NAME}.csv"
)
TOP10_OUTPUT_FILE_TEMPLATE = os.path.join(
    RESULTS_DIR, "top10_mlp_configs_holdout_{BASE_NAME}.csv"
)
FINAL_SUMMARY_FILE = os.path.join(RESULTS_DIR, "comparacao_top10_todas_bases.csv")

# Configurações gerais
RANDOM_STATE = 42
TEST_SIZE = 0.3  # Holdout 70% treino, 30% teste
N_JOBS = -1  # Usa todos os núcleos

# Colunas a ignorar (as mesmas usadas nos scripts anteriores)
LABEL_COLUMN = "label"
COLUMNS_TO_IGNORE = [LABEL_COLUMN, "filename", "animal", "race", "nome_arquivo"]


# ==============================================================================
#                      FUNÇÕES DE SUPORTE
# ==============================================================================


def get_base_files():
    """Retorna uma lista de todos os arquivos CSV no diretório BASE_DIR."""
    print(f"📁 Acessando diretório: {BASE_DIR}")

    if not os.path.exists(BASE_DIR):
        raise FileNotFoundError(f"Diretório de bases não encontrado: {BASE_DIR}")

    # Filtra apenas arquivos .csv
    base_files = [f for f in os.listdir(BASE_DIR) if f.endswith(".csv")]
    if not base_files:
        raise ValueError(f"Nenhum arquivo CSV encontrado em {BASE_DIR}")

    return base_files


def load_data(base_name):
    """Carrega os dados da base especificada e prepara para o Holdout."""
    base_path = os.path.join(BASE_DIR, base_name)

    if not os.path.exists(base_path):
        raise FileNotFoundError(f"Base não encontrada: {base_path}")

    try:
        df = pd.read_csv(base_path, sep=",")
    except Exception:
        # Tenta com separador ';' se falhar
        df = pd.read_csv(base_path, sep=";")

    # 1. Preparar X e y
    y = df[LABEL_COLUMN]
    cols_to_drop = [col for col in COLUMNS_TO_IGNORE if col in df.columns]
    X_temp = df.drop(columns=cols_to_drop, errors="ignore")
    X = X_temp.select_dtypes(include=np.number)
    X.dropna(axis=1, how="all", inplace=True)
    X.fillna(0, inplace=True)  # Preenche NaNs restantes com 0, se houver.

    if X.shape[1] == 0:
        raise ValueError("Nenhuma feature numérica válida encontrada.")

    # Validação de classes (necessária para stratify)
    if len(np.unique(y)) < 2:
        raise ValueError("A base possui menos de 2 classes na coluna 'label'.")

    # 2. Split Holdout
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Preparar para PredefinedSplit
    X_full = np.concatenate((X_train, X_test), axis=0)
    y_full = np.concatenate((y_train, y_test), axis=0)

    # -1 para treino, 0 para teste
    test_fold = np.concatenate(
        (-1 * np.ones(len(X_train), dtype=int), 0 * np.ones(len(X_test), dtype=int))
    )
    ps = PredefinedSplit(test_fold)

    print(f"✅ Base '{base_name}' carregada.")
    print(
        f"Dados: {X.shape[0]} amostras, {X.shape[1]} features, {len(np.unique(y))} classes."
    )
    return X_full, y_full, ps, X.shape[1], len(np.unique(y_train))


def generate_param_grid(n_features, n_classes):
    """Gera combinações de parâmetros adaptadas ao número de features."""

    # Faixa de max_iter proporcional ao número de features, com até 10 variações
    def get_max_iter_range(n):
        step = max(1, int(math.log2(n) * 75))  # Ajusta o passo conforme n_features
        return [500 + i * step for i in range(5)]

    # Configurações de camadas ocultas
    hidden_layer_configs = [
        (n_features,),
        (n_features, n_features),
        (n_features // 2, n_features // 2),
        (n_features, n_features, n_features),
        (n_features // 2, n_features // 2, n_features // 2),
        ((n_features + 2) // 3, (n_features + 2) // 3, (n_features + 2) // 3),
        (n_features, n_features, n_features, n_features),
        (
            (n_features + 1) // 2,
            (n_features + 1) // 2,
            (n_features + 1) // 2,
            (n_features + 1) // 2,
        ),
        (
            (n_features + 2) // 3,
            (n_features + 2) // 3,
            (n_features + 2) // 3,
            (n_features + 2) // 3,
        ),
        (2 * n_features,),
        (2 * n_features, 2 * n_features),
        (2 * n_features, 2 * n_features, 2 * n_features),
        (2 * n_features, 2 * n_features, 2 * n_features, 2 * n_features),
    ]

    if n_classes > 1:
        hidden_layer_configs.append((n_classes * 2, n_classes * 2))

    max_iter_values = get_max_iter_range(n_features)

    param_grid = []
    for hls in hidden_layer_configs:
        for activation in [
            "logistic",
            "relu",
        ]:  # ["identity", "logistic", "tanh", "relu"]:
            for solver in ["sgd", "adam"]:
                for lr in [0.001, 0.01, 0.1]:  # [0.0001, 0.001, 0.01, 0.1]:
                    for max_iter in max_iter_values:
                        config = {
                            "hidden_layer_sizes": hls,
                            "activation": activation,
                            "solver": solver,
                            "learning_rate_init": lr,
                            "max_iter": max_iter,
                            "early_stopping": False,
                            "random_state": RANDOM_STATE,
                        }
                        if solver == "adam":
                            config["epsilon"] = 1e-08
                        param_grid.append(config)

    print(
        f"🔧 Gerado grid com {len(param_grid)} combinações para {n_features} features."
    )
    return param_grid


def evaluate_config(params, X_full, y_full, ps):
    """Função para ser executada em paralelo."""
    # O random_state já está em params

    modelo = MLPClassifier(**params)

    try:
        # Apenas um split no PredefinedSplit (Treino no -1, Teste no 0)
        for train_idx, test_idx in ps.split():
            # Treinamento
            modelo.fit(X_full[train_idx], y_full[train_idx])

            # Predição e Score
            y_pred = modelo.predict(X_full[test_idx])
            # Se houver apenas uma classe, 'weighted' pode falhar. Usa 'f1_macro' (mais robusto)
            # Mas, se load_data validar as classes, 'weighted' deve ser o correto.
            score = f1_score(
                y_full[test_idx], y_pred, average="weighted", zero_division=0
            )

            return {"params": params, "f1_score": score, "status": "OK"}

    except Exception as e:
        return {
            "params": params,
            "f1_score": None,
            "status": "ERRO",
            "erro_msg": str(e),
        }


def run_grid_search_for_base(base_name, n_jobs=N_JOBS):
    """Executa o Grid Search para uma única base e salva o Top 10 com colunas detalhadas."""
    log_file = LOG_FILE_TEMPLATE.format(BASE_NAME=base_name)
    top10_output_file = TOP10_OUTPUT_FILE_TEMPLATE.format(BASE_NAME=base_name)

    print("\n" + "#" * 50)
    print(f"🔄 INICIANDO PROCESSAMENTO PARA A BASE: {base_name}")
    print("#" * 50)

    try:
        X_full, y_full, ps, n_features, n_classes = load_data(base_name)
    except Exception as e:
        print(f"❌ Erro Crítico ao carregar dados da base {base_name}: {e}")
        return pd.DataFrame(), None

    param_grid = generate_param_grid(n_features, n_classes)
    total_configs = len(param_grid)
    print(f"🔧 Total de combinações a testar: {total_configs}")

    # 1. Carregar Log Incremental
    df_log = pd.DataFrame()
    tested_params_str = set()

    if os.path.exists(log_file):
        try:
            df_log = pd.read_csv(log_file, sep=";", decimal=",", dtype={"params": str})
            df_temp = df_log.copy()
            df_temp["params"] = df_temp["params"].apply(ast.literal_eval)
            tested_params_str = set(df_temp["params"].astype(str))
            print(
                f"💾 Log incremental carregado com {len(df_log)} resultados para {base_name}."
            )
        except Exception as e:
            print(
                f"❌ ERRO ao ler o log {log_file} ({e}). Apague o arquivo e execute novamente."
            )
            return pd.DataFrame(), None
    else:
        print("🆕 Iniciando novo log incremental.")

    # 2. Identificar Configurações Pendentes
    configs_pendentes = [p for p in param_grid if str(p) not in tested_params_str]
    num_pendentes = len(configs_pendentes)

    if num_pendentes == 0:
        print("⏩ Todas as configurações já foram testadas. Pulando a execução.")
        df_new_results = pd.DataFrame()
    else:
        print(f"🚀 Avaliando {num_pendentes} combinações com {n_jobs} núcleos...")
        start_time = time.time()

        results = Parallel(n_jobs=n_jobs, verbose=10)(
            delayed(evaluate_config)(params, X_full, y_full, ps)
            for params in configs_pendentes
        )

        end_time = time.time()
        print(f"✅ Execução concluída em {end_time - start_time:.2f}s")
        df_new_results = pd.DataFrame(results)

    # 3. Consolidar e Salvar Log
    if not df_new_results.empty:
        df_new_results["params"] = df_new_results["params"].astype(str)
        df_new_results["n_features"] = n_features
        df_new_results["n_classes"] = n_classes
        df_new_results["base_name"] = base_name

        df_log = pd.concat([df_log, df_new_results], ignore_index=True)
        df_log.to_csv(log_file, index=False, sep=";", decimal=",")
        print(f"💾 Log atualizado com {len(df_log)} entradas.")

    # 4. Gerar Top 10
    df_final = df_log.copy()
    if df_final.empty:
        print(f"⚠️ Log vazio para {base_name}.")
        return pd.DataFrame(), None

    try:
        df_final["params_dict"] = df_final["params"].apply(ast.literal_eval)
    except Exception as e:
        print(f"❌ Erro ao converter parâmetros: {e}")
        return pd.DataFrame(), None

    df_validos = df_final[df_final["f1_score"].notnull()]
    if df_validos.empty:
        print(f"⚠️ Nenhum resultado válido para Top 10.")
        return pd.DataFrame(), None

    top10 = (
        df_validos.sort_values(by="f1_score", ascending=False)
        .head(10)
        .reset_index(drop=True)
    )

    # Adiciona colunas detalhadas
    top10["base_name"] = base_name
    top10["n_features"] = n_features
    top10["n_classes"] = n_classes
    top10["hidden_layer_sizes"] = top10["params_dict"].apply(
        lambda x: str(x["hidden_layer_sizes"])
    )
    top10["activation"] = top10["params_dict"].apply(lambda x: x["activation"])
    top10["solver"] = top10["params_dict"].apply(lambda x: x["solver"])
    top10["learning_rate_init"] = top10["params_dict"].apply(
        lambda x: x["learning_rate_init"]
    )
    top10["max_iter"] = top10["params_dict"].apply(lambda x: x["max_iter"])

    # Colunas finais para salvar
    cols_to_save = [
        "base_name",
        "n_features",
        "n_classes",
        "f1_score",
        "hidden_layer_sizes",
        "activation",
        "solver",
        "learning_rate_init",
        "max_iter",
        "params",
    ]

    top10_save = top10[cols_to_save].copy()
    top10_save.to_csv(top10_output_file, index=False, sep=";", decimal=",")
    print(f"📋 Top 10 salvo em: {top10_output_file}")

    return top10_save, top10


# ==============================================================================
#                      PROCESSO PRINCIPAL
# ==============================================================================


def run_all_bases_comparison():
    """Roda o Grid Search em todas as bases e gera um resumo final."""
    try:
        base_files = get_base_files()
    except Exception as e:
        print(f"❌ Erro ao listar arquivos: {e}")
        return

    print(f"Bases CSV encontradas para processamento: {base_files}")

    all_top10_results = []

    for base_name in tqdm(base_files, desc="Processando Bases"):
        # Executa o grid search e salva o CSV por base
        top10_base, _ = run_grid_search_for_base(base_name)

        if not top10_base.empty:
            all_top10_results.append(top10_base)

    if all_top10_results:
        # 1. Consolidar todos os Top 10 em um único arquivo
        df_summary = pd.concat(all_top10_results, ignore_index=True)

        # 2. Salvar o resumo final
        df_summary.to_csv(FINAL_SUMMARY_FILE, index=False, sep=";", decimal=",")
        print("\n" + "=" * 60)
        print(f"🏆 RESUMO FINAL SALVO EM: {FINAL_SUMMARY_FILE}")
        print(f"Este arquivo contém o Top 10 de CADA base para análise de relações.")
        print("=" * 60)

        # 3. Exibir o melhor de todos
        df_best = df_summary.sort_values(by="f1_score", ascending=False).iloc[0]
        print("\n⭐ MELHOR CONFIGURAÇÃO ENCONTRADA (GERAL):")
        print(f"Base: {df_best['base_name']}")
        print(f"F1-Score: {df_best['f1_score']:.4f}")
        print(f"Features: {df_best['n_features']}")
        print(
            f"Config: {df_best['hidden_layer_sizes']} (Act: {df_best['activation']}, LR: {df_best['learning_rate_init']}, Iter: {df_best['max_iter']})"
        )

    else:
        print("⚠️ Nenhum resultado de Top 10 válido gerado. Verifique os logs de erro.")


if __name__ == "__main__":
    run_all_bases_comparison()
