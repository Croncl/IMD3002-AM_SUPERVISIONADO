import pandas as pd
import os

# ==============================================================================
#                 CONFIGURAÇÃO DE ENTRADA E SAÍDA
# ==============================================================================

# **MANTIDO:** Nome do arquivo CSV de resultados que você especificou
INPUT_CSV_FILE = "comparativo_bases_matriz_completa_k_x_metodologia_KNN.csv"
INPUT_DIR = "."  # Pasta atual (onde o script e o CSV estão localizados)

# Número de bases a selecionar e exibir
NUM_TOP_BASES = 15

# ==============================================================================
#                 FUNÇÃO PRINCIPAL DE ANÁLISE E SELEÇÃO
# ==============================================================================


def analyze_and_select_bases(input_file, num_bases):
    """
    Carrega o CSV de resultados, faz o "melt" das colunas k=1 a k=10,
    calcula Média e Desvio Padrão por Base, seleciona e ordena as top N bases.
    """
    input_path = os.path.join(INPUT_DIR, input_file)

    if not os.path.exists(input_path):
        print(f"❌ ERRO: Arquivo de resultados não encontrado em: {input_path}")
        return None

    try:
        # Carregar o CSV. Usando 'sep=;' conforme o padrão de CSVs gerados anteriormente.
        df_results = pd.read_csv(input_path, sep=";")
    except Exception as e:
        print(f"❌ ERRO ao carregar o arquivo CSV: {e}")
        return None

    print(
        f"✅ Arquivo de resultados '{input_file}' carregado. Total de {len(df_results)} linhas iniciais."
    )

    # 1. IDENTIFICAR E DERRETER (MELT) AS COLUNAS DE K

    # Identifica todas as colunas que começam com 'k='
    k_columns = [col for col in df_results.columns if col.startswith("k=")]

    if not k_columns:
        print(
            "❌ ERRO: Colunas 'k=1' a 'k=10' não encontradas. Verifique o cabeçalho do CSV."
        )
        print(f"Colunas encontradas: {df_results.columns.tolist()}")
        return None

    # Derreter o DataFrame: transforma as 10 colunas 'k=' em duas colunas longas:
    # 'k' (com o nome da coluna k) e 'Score' (com o valor)
    df_melted = pd.melt(
        df_results,
        id_vars=["Base", "Metodologia"],  # Colunas que serão mantidas
        value_vars=k_columns,  # Colunas que serão derretidas
        var_name="k",  # Novo nome para a coluna de k (ex: 'k=1')
        value_name="Score",  # Novo nome para a coluna de valores
    )

    print(
        f"   → Derretimento (Melt) concluído. Novo total de execuções: {len(df_melted)}"
    )

    # 2. TRATAMENTO E GARANTIA DE DADOS NUMÉRICOS
    # O CSV está usando vírgula (,) como separador decimal. Precisamos corrigir.
    if df_melted["Score"].dtype == "object":
        df_melted["Score"] = df_melted["Score"].str.replace(",", ".", regex=False)

    df_melted["Score"] = pd.to_numeric(df_melted["Score"], errors="coerce")

    # Remove execuções inválidas onde o Score não foi numérico
    df_melted.dropna(subset=["Score"], inplace=True)

    # 3. AGRUPAR E CALCULAR ESTATÍSTICAS POR BASE
    # Calcula Média, Desvio Padrão e o número total de testes para cada Base.
    df_stats = (
        df_melted.groupby("Base")["Score"]
        .agg(Media_Score=("mean"), Desvio_Padrao_Score=("std"), Total_Testes=("count"))
        .reset_index()
    )

    # Preenche o DP com 0 para bases que só tiveram 1 teste (Desvio Padrão é NaN neste caso)
    df_stats["Desvio_Padrao_Score"] = df_stats["Desvio_Padrao_Score"].fillna(0)

    # 4. CRIAÇÃO DO SCORE PONDERADO PARA ORDENAÇÃO
    # Score Ponderado = Média - Desvio Padrão (Penaliza a inconsistência)
    df_stats["Score_Ponderado_Final"] = (
        df_stats["Media_Score"] - df_stats["Desvio_Padrao_Score"]
    )

    # 5. ORDENAÇÃO E SELEÇÃO
    df_sorted = df_stats.sort_values(
        by="Score_Ponderado_Final", ascending=False
    ).reset_index(drop=True)

    # Adicionar a coluna de Rank/Posição
    df_sorted.index = df_sorted.index + 1
    df_sorted = df_sorted.rename_axis("Rank").reset_index()

    df_top_bases = df_sorted.head(num_bases)

    # 6. EXIBIÇÃO E SALVAMENTO DO RELATÓRIO
    print("\n" + "=" * 80)
    print(f"🏆 CLASSIFICAÇÃO GERAL DAS BASES ({len(df_sorted)} BASES TOTAIS)")
    print(f"   (Ordenado pelo Score Ponderado: Média - Desvio Padrão)")
    print("=" * 80)

    # Exibir a tabela das top N bases formatada
    # Arredondando os scores para melhor visualização
    df_display = df_top_bases.copy()
    for col in df_display.columns:
        if "Score" in col:
            df_display[col] = df_display[col].round(4)

    print(df_display.to_string(index=False))

    # Salvar o relatório completo ordenado
    output_top_csv = f"Relatorio_Bases_Ordenadas_KNN.csv"

    # Salvando o arquivo com o separador decimal como vírgula (,)
    df_sorted.to_csv(output_top_csv, index=False, sep=";", decimal=",")
    print("\n" + "=" * 80)
    print(
        f"✅ Relatório completo com todas as bases ordenadas salvo em: {output_top_csv}"
    )


# ==============================================================================
#                 EXECUÇÃO
# ==============================================================================

analyze_and_select_bases(INPUT_CSV_FILE, NUM_TOP_BASES)
