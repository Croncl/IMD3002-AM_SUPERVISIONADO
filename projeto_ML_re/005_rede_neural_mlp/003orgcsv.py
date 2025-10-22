import os
import pandas as pd
import ast

# ==============================================================================
#                 CONFIGURAÇÃO
# ==============================================================================

# Defina o nome do arquivo de log que contém os resultados do K-Fold CV
CAMINHO_CSV = "resultados_mlp_15bases_10fold_cv_inc.csv"
OUTPUT_CSV_FINAL = "tabela_resultados_mlp_kfold_organizada.csv"


# ==============================================================================
#                 FUNÇÃO DE NORMALIZAÇÃO
# ==============================================================================


def normalize_config_string(config_dict):
    """
    Normaliza o dicionário de configuração, removendo chaves não essenciais
    e garantindo a ordem das chaves para uma string consistente.
    """
    if not isinstance(config_dict, dict):
        return str(config_dict)

    cfg_copy = config_dict.copy()

    # 1. Remove chaves que não definem unicamente a configuração (se estiverem presentes)
    cfg_copy.pop("random_state", None)

    # 2. Garante que 'early_stopping' seja False para consistência (como forçado no script de avaliação)
    if "early_stopping" not in cfg_copy:
        cfg_copy["early_stopping"] = False

    # 3. Garante que 'hidden_layer_sizes' seja uma tupla (para string consistente)
    hls = cfg_copy.get("hidden_layer_sizes")
    if isinstance(hls, list):
        cfg_copy["hidden_layer_sizes"] = tuple(hls)

    # Ordena as chaves para garantir que a string final seja SEMPRE a mesma
    # para a mesma combinação de hiperparâmetros.
    sorted_keys = sorted(cfg_copy.keys())

    # Gera a string final
    normalized_str = str({k: cfg_copy[k] for k in sorted_keys})
    return normalized_str


# ==============================================================================
#                 PROCESSO PRINCIPAL
# ==============================================================================

# 📥 1. Carregar resultados
# 📥 1. Carregar resultados
if os.path.exists(CAMINHO_CSV):
    df_kfold = None

    # 1. Tenta carregar com o separador esperado (ponto-e-vírgula)
    try:
        df_kfold = pd.read_csv(CAMINHO_CSV, sep=";")
        print(f"✔️ Arquivo '{CAMINHO_CSV}' carregado com separador ';'.")
    except Exception:
        pass  # Ignora o erro e tenta a próxima opção

    # 2. Se a primeira tentativa falhou, tenta carregar com o separador padrão (vírgula)
    if df_kfold is None or "config" not in df_kfold.columns:
        try:
            df_kfold = pd.read_csv(CAMINHO_CSV, sep=",")
            print(f"✔️ Arquivo '{CAMINHO_CSV}' carregado com separador ','.")
        except Exception as e:
            print(f"❌ Erro ao ler o CSV com ',' ou ';'. Erro de leitura: {e}")
            exit()

    # 3. Verifica se a coluna 'config' finalmente existe
    if "config" not in df_kfold.columns:
        print(
            f"❌ A coluna 'config' não foi encontrada após tentar os separadores ',' e ';'."
        )
        print("As colunas disponíveis são:", df_kfold.columns.tolist())
        exit()

    try:
        # 4. Converter a coluna 'config' de string para dicionário
        df_kfold["config_dict"] = df_kfold["config"].apply(ast.literal_eval)
        print(f"✔️ Coluna 'config' convertida para dicionário.")
    except Exception as e:
        print(
            f"❌ Erro fatal ao converter a coluna 'config' para dicionário. Dados mal formados. Erro: {e}"
        )
        exit()
else:
    print(f"⚠️ Arquivo '{CAMINHO_CSV}' não encontrado. Verifique o caminho.")
    exit()


# 📊 3. Preparar e Pivotar
if not df_kfold.empty:

    # 4. Normaliza a configuração para a chave do pivô
    df_kfold["config_key"] = df_kfold["config_dict"].apply(normalize_config_string)

    # 5. Pivota a tabela: 'base' nas linhas e 'config_key' nas colunas,
    #    usando 'f1_score_medio' como valor
    tabela_kfold = df_kfold.pivot_table(
        index="base",
        columns="config_key",
        values="f1_score_medio",
        aggfunc="first",  # Pega o primeiro (e único) score médio
    )

    print("\n📊 Tabela organizada (KFold):")
    # Mostra as primeiras linhas e as colunas (configurações)
    print(tabela_kfold.head())

    # 6. Salva o resultado final no formato CSV
    # Usa sep=";" e decimal="," para consistência
    tabela_kfold.to_csv(OUTPUT_CSV_FINAL, sep=";", decimal=",")
    print(f"\n💾 Tabela KFold salva em '{OUTPUT_CSV_FINAL}'")
else:
    print("⚠️ Nenhum dado disponível para KFold.")
