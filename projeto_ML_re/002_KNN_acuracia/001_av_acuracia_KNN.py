import pandas as pd
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import train_test_split, KFold, cross_val_score
from sklearn.metrics import accuracy_score, f1_score, make_scorer
from joblib import Parallel, delayed
import os
import matplotlib.pyplot as plt

# ==============================================================================
#                 CONFIGURAÇÕES DE CAMINHO E PARÂMETROS
# ==============================================================================

# Caminho para a pasta que contém os CSVs gerados (RELATIVO ao local do script)
input_dir_bases = os.path.abspath(os.path.join(os.path.dirname(__file__),  "..", "bases_geradas"))


if not os.path.exists(input_dir_bases):
    raise FileNotFoundError(f"❌ ERRO FATAL: Pasta não encontrada: '{input_dir_bases}'\nVerifique se o primeiro script gerou os arquivos corretamente.")

print("📂 Pasta de entrada das bases:", input_dir_bases)


output_csv_final = "comparativo_bases_matriz_completa_k_x_metodologia_KNN.csv"  # NOVO NOME

# Checkpoint (Cache Único) para salvar resultados detalhados
checkpoint_full_analysis = "checkpoint_full_matriz_k_x_metodologia_KNN.pkl"

# Parâmetros de Avaliação
percentuais_holdout = [0.1, 0.2, 0.3, 0.4]  # Todos serão avaliados
k_vizinhos_range = range(1, 11)  # k=1 a k=10, em COLUNAS
splits_kfold = [5, 10]
metric_knn = "euclidean"
scorer_f1 = make_scorer(f1_score, average="macro", zero_division=0)
N_JOBS = -1  # Usa todos os núcleos da CPU para o paralelismo

# ==============================================================================
#                 LISTAGEM DE BASES LOCAIS
# ==============================================================================

try:
    bases_locais = sorted(
        [f for f in os.listdir(input_dir_bases) if f.lower().endswith(".csv")]
    )
    if not bases_locais:
        raise FileNotFoundError("Nenhuma base CSV encontrada.")
    print(f"✅ {len(bases_locais)} bases CSVs locais encontradas em: {input_dir_bases}")
except FileNotFoundError as e:
    print(f"❌ ERRO FATAL: {e}")
    print(
        "Verifique se o caminho 'input_dir_bases' está correto e se o primeiro script gerou os arquivos."
    )
    exit()

# ==============================================================================
#                 FUNÇÃO AUXILIAR DE CARREGAMENTO
# ==============================================================================

def load_and_preprocess_base(base_name):
    """Carrega uma base CSV localmente, trata colunas e retorna X, y."""
    file_path = os.path.join(input_dir_bases, base_name)
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception:
        return None, None

    X = df.drop(columns=["filename", "animal", "label", "race"], errors="ignore").copy()
    if "label" not in df.columns:
        return None, None
    y = df["label"].copy()

    for col in X.columns:
        X[col] = pd.to_numeric(X[col], errors="coerce")
    X.dropna(axis=1, how="all", inplace=True)

    if X.shape[1] == 0:
        return None, None
    return X, y


# ==============================================================================
#                 FUNÇÃO UNIFICADA DE AVALIAÇÃO (MATRIZ COMPLETA)
# ==============================================================================


def run_full_matrix_evaluation(
    base_name, percentuais_holdout, k_range, splits_kfold, metric_knn, scorer_f1
):
    """Executa Holdout (10, 20, 30, 40%) e K-Fold (5 e 10) para todos os k (1-10)."""
    X, y = load_and_preprocess_base(base_name)
    if X is None:
        return []

    resultados = []

    # --- 1. Avaliação Holdout (Acurácia) para k=1 a k=10, em todos os percentuais ---
    for pct in percentuais_holdout:
        metodologia = f"Holdout ({int(pct*100)}%) [Acurácia]"

        if X.shape[0] * (1 - pct) < 2 or X.shape[0] * pct < 2:
            continue

        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=pct, random_state=1, stratify=y
            )
            for k in k_range:
                knn = KNeighborsClassifier(n_neighbors=k, metric=metric_knn)
                knn.fit(X_train, y_train)
                y_pred = knn.predict(X_test)
                score = accuracy_score(y_test, y_pred)

                resultados.append(
                    {
                        "Base": base_name,
                        "Metodologia": metodologia,
                        "k": k,
                        "Score": round(score, 4),
                    }
                )
        except Exception:
            pass

    # --- 2. Avaliação K-Fold (F1 Score) para k=1 a k=10, em 5 e 10 folds ---
    for n_splits in splits_kfold:
        metodologia = f"{n_splits}-Fold [F1 Média]"

        if X.shape[0] < n_splits:
            continue

        kf = KFold(n_splits=n_splits, shuffle=True, random_state=1)

        for k in k_range:
            # Verifica se k é maior que o número de amostras
            if X.shape[0] < k:
                continue

            knn = KNeighborsClassifier(n_neighbors=k, metric=metric_knn)

            try:
                # n_jobs=1 na cross_val_score para evitar conflitos com o Parallel externo
                scores = cross_val_score(knn, X, y, cv=kf, scoring=scorer_f1, n_jobs=1)

                resultados.append(
                    {
                        "Base": base_name,
                        "Metodologia": metodologia,
                        "k": k,
                        "Score": round(np.mean(scores), 4),
                    }
                )
            except Exception:
                pass

    return resultados



#                 EXECUÇÃO COM PARALELISMO E CHECKPOINT

print("\n--- INICIANDO ANÁLISE MATRIZ COMPLETA (Paralelismo e Checkpoint) ---")

# Tenta carregar resultados anteriores
try:
    df_full_results = pd.read_pickle(checkpoint_full_analysis)
    existing_combinations = (
        df_full_results[["Base", "Metodologia", "k"]]
        .apply(lambda x: tuple(x), axis=1)
        .tolist()
    )

    print(
        f"✅ Encontrados {len(df_full_results['Base'].unique())} bases no checkpoint."
    )
except:
    df_full_results = pd.DataFrame()
    existing_combinations = []
    print("⚠️ Nenhum checkpoint encontrado. Iniciando do zero.")


# cuidará das execuções parciais se o script for interrompido.
bases_to_run = bases_locais


# Execução Paralela
# A lista de resultados é uma lista de listas (uma para cada base)
list_of_results = Parallel(n_jobs=N_JOBS, verbose=10)(
    delayed(run_full_matrix_evaluation)(
        base, percentuais_holdout, k_vizinhos_range, splits_kfold, metric_knn, scorer_f1
    )
    for base in bases_to_run
)

# Consolidação e Salvar Checkpoint
new_results = [item for sublist in list_of_results for item in sublist]
df_new_results = pd.DataFrame(new_results)

# Combina resultados antigos e novos
if not df_new_results.empty:
    df_full_results = pd.concat([df_full_results, df_new_results], ignore_index=True)

# Garante que não há duplicatas (Base + Metodologia + k)
df_full_results.drop_duplicates(
    subset=["Base", "Metodologia", "k"], keep="last", inplace=True
)

df_full_results.to_pickle(checkpoint_full_analysis)
print(
    f"💾 Checkpoint atualizado com {len(df_full_results['Base'].unique())} bases totais."
)


# ==============================================================================
#                 PIVOTAMENTO, ORDENAÇÃO E GERAÇÃO DE CSV
# ==============================================================================

if df_full_results.empty:
    print("\n❌ Não há resultados válidos para gerar a tabela final.")
    exit()

# 1. Pivotear para colocar 'k' como colunas
df_pivot = df_full_results.pivot_table(
    index=["Base", "Metodologia"], columns="k", values="Score"
).reset_index()

# 2. Renomear as colunas k (de 1 a 10)
df_pivot.columns = ["Base", "Metodologia"] + [
    f"k={int(c)}" for c in df_pivot.columns[2:]
]

# 3. Definir a ordem das linhas (Metodologia)
metodologia_order = [
    "Holdout (10%) [Acurácia]",
    "Holdout (20%) [Acurácia]",
    "Holdout (30%) [Acurácia]",
    "Holdout (40%) [Acurácia]",
    "5-Fold [F1 Média]",
    "10-Fold [F1 Média]",
]
df_pivot["Metodologia_Ordenada"] = pd.Categorical(
    df_pivot["Metodologia"], categories=metodologia_order, ordered=True
)

# 4. Criar chave de ordenação primária: Média do F1 (k=1 a 10) do 10-Fold
f1_cols = [col for col in df_pivot.columns if col.startswith("k=")]
df_pivot["Media_F1_10Fold"] = df_pivot.apply(
    lambda row: (
        row[f1_cols].mean() if row["Metodologia"] == "10-Fold [F1 Média]" else np.nan
    ),
    axis=1,
)

ordenacao_10fold = (
    df_pivot[df_pivot["Metodologia"] == "10-Fold [F1 Média]"]
    .sort_values("Media_F1_10Fold", ascending=False)["Base"]
    .tolist()
)

base_cat_type = pd.CategoricalDtype(categories=ordenacao_10fold, ordered=True)
df_pivot["Base_Ordenada"] = df_pivot["Base"].astype(base_cat_type)

# 5. Ordenar: 1º Base (pelo melhor 10-Fold F1), 2º Metodologia (ordem lógica)
df_final_ordenado = (
    df_pivot.sort_values(by=["Base_Ordenada", "Metodologia_Ordenada"])
    .drop(columns=["Metodologia_Ordenada", "Media_F1_10Fold", "Base_Ordenada"])
    .reset_index(drop=True)
)

# 6. Geração do CSV
df_final_ordenado.to_csv(output_csv_final, index=False, sep=";", encoding="utf-8")

# ==============================================================================
#                 VISUALIZAÇÃO E RELATÓRIO
# ==============================================================================

print("\n" + "=" * 100)
print("✅ RESULTADO FINAL CONSOLIDADO (MATRIZ COMPLETA K x METODOLOGIA)")
print("=" * 100)
print(f"📁 Tabela comparativa salva como: '{output_csv_final}' (separador ';')")

# Imprimir as primeiras 18 linhas (3 metodologias * 6 bases ou 6 metodologias * 3 bases)
print("\n📊 Exemplo de Tabela Final (Top 3 Bases, todas as 6 Metodologias):")
print(df_final_ordenado.head(18).to_string(index=False))
print("-" * 100)
print("💡 Parâmetros Gerais Usados:")
print(f"   - k avaliado: De 1 a 10 (em colunas)")
print(f"   - Metodologias Holdout (Acurácia): 10%, 20%, 30%, 40%")
print(f"   - Metodologias k-Fold (F1 Média): 5-Fold, 10-Fold")
print(f"   - Ordenação: Bases ordenadas pela Média F1 do 10-Fold.")
print("-" * 100)


# ... (O código anterior, que termina na Geração do CSV, permanece intacto)

# ==============================================================================
#                 PREPARAÇÃO DE DADOS PARA VISUALIZAÇÃO
# ==============================================================================

# Calcular o melhor k e o melhor score para cada Base e Metodologia
df_best_scores = df_full_results.loc[
    df_full_results.groupby(["Base", "Metodologia"])["Score"].idxmax()
].reset_index(drop=True)

# 1. Identificar Top 5 Bases (pelo melhor score do 10-Fold)
top_5_bases_list = (
    df_best_scores[df_best_scores["Metodologia"] == "10-Fold [F1 Média]"]
    .sort_values(by="Score", ascending=False)["Base"]
    .head(5)
    .tolist()
)

df_plot = df_best_scores[df_best_scores["Base"].isin(top_5_bases_list)].copy()

if top_5_bases_list:

    print("\n" + "=" * 100)
    print("✅ GERANDO VISUALIZAÇÕES PARA AS TOP 5 BASES")
    print("=" * 100)

    # ----------------------------------------------------------------------
    # 📊 GRÁFICO 1: LINHA DO TEMPO (HOLDOUT ACROSS PERCENTAGES)
    # ----------------------------------------------------------------------

    df_holdout_plot = df_plot[df_plot["Metodologia"].str.contains("Holdout")].copy()

    # Extrair o percentual para o eixo X
    df_holdout_plot["Percentual"] = (
        df_holdout_plot["Metodologia"].str.extract(r"\((\d+)%\)").astype(int)
    )

    plt.figure(figsize=(10, 6))

    for base in top_5_bases_list:
        data = df_holdout_plot[df_holdout_plot["Base"] == base].sort_values(
            "Percentual"
        )
        plt.plot(
            data["Percentual"],
            data["Score"],
            marker="o",
            label=base.replace(".csv", ""),
        )

    plt.title(
        "Acurácia Holdout: Performance em função do percentual de Teste (Top 5 Bases)"
    )
    plt.xlabel("Percentual de Dados de Teste (%)")
    plt.ylabel("Melhor Acurácia (k=1 a 10)")
    plt.xticks(percentuais_holdout * 100)
    plt.legend(title="Base", bbox_to_anchor=(1.05, 1), loc="upper left")
    plt.grid(axis="y", linestyle="--", alpha=0.7)
    plt.tight_layout()
    plt.show()

    # ----------------------------------------------------------------------
    # 📊 GRÁFICO 2: BARRAS (MELHOR HOLD-OUT vs MELHOR K-FOLD)
    # ----------------------------------------------------------------------

    # 1. Encontrar o MELHOR Holdout (maior Acurácia entre 10, 20, 30, 40%)
    df_best_holdout = df_holdout_plot.loc[
        df_holdout_plot.groupby("Base")["Score"].idxmax()
    ][["Base", "Score"]].rename(columns={"Score": "Melhor Holdout"})

    # 2. Encontrar o MELHOR K-Fold (maior F1 entre 5-Fold e 10-Fold)
    df_kfold_plot = df_plot[df_plot["Metodologia"].str.contains("Fold")].copy()
    df_best_kfold = df_kfold_plot.loc[df_kfold_plot.groupby("Base")["Score"].idxmax()][
        ["Base", "Score"]
    ].rename(columns={"Score": "Melhor K-Fold"})

    # 3. Consolidar para o gráfico de barras
    df_bar_plot = pd.merge(df_best_holdout, df_best_kfold, on="Base", how="inner")

    # Garantir a ordenação (pelo Melhor K-Fold, a métrica mais robusta)
    df_bar_plot = df_bar_plot.sort_values("Melhor K-Fold", ascending=False)

    plt.figure(figsize=(12, 7))

    n_bases = len(df_bar_plot)
    r = np.arange(n_bases)
    width = 0.35

    base_names = [b.replace(".csv", "") for b in df_bar_plot["Base"]]

    plt.bar(
        r - width / 2,
        df_bar_plot["Melhor Holdout"],
        width,
        label="Melhor Holdout (Acurácia)",
        color="#4CAF50",
    )
    plt.bar(
        r + width / 2,
        df_bar_plot["Melhor K-Fold"],
        width,
        label="Melhor K-Fold (F1 Média)",
        color="#2196F3",
    )

    plt.ylabel("Score")
    plt.title(
        f"Comparação de Performance (Top {n_bases} Bases) - Melhor Holdout vs Melhor k-Fold"
    )
    plt.xticks(r, base_names, rotation=45, ha="right")
    plt.legend()
    plt.grid(axis="y", linestyle="--", alpha=0.6)
    plt.tight_layout()
    plt.show()

else:
    print("\n❌ Não há bases válidas para gerar a visualização.")


# ==============================================================================
#                 RELATÓRIO FINAL DO DATAFRAME
# ==============================================================================

print("\n" + "=" * 100)
print("✅ RESULTADO FINAL CONSOLIDADO (MATRIZ COMPLETA K x METODOLOGIA)")
print("=" * 100)
print(f"📁 Tabela comparativa salva como: '{output_csv_final}' (separador ';')")

# Imprimir as primeiras 18 linhas (6 metodologias * 3 bases)
print("\n📊 Exemplo de Tabela Final (Top 3 Bases, todas as 6 Metodologias):")
print(df_final_ordenado.head(18).to_string(index=False))
print("-" * 100)
print("💡 Parâmetros Gerais Usados:")
print(f"   - k avaliado: De 1 a 10 (em colunas)")
print(f"   - Metodologias Holdout (Acurácia): 10%, 20%, 30%, 40%")
print(f"   - Metodologias k-Fold (F1 Média): 5-Fold, 10-Fold")
print(f"   - Ordenação: Bases ordenadas pela Média F1 do 10-Fold.")
print("-" * 100)
