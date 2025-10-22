import numpy as np
import pandas as pd
from PIL import Image
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm  # Usado para ambiente local/terminal
import warnings
import os
import cv2  # Necessário para a rotação no LBP Multi-Orientation
import random

warnings.filterwarnings("ignore")

# ==============================================================================
#                 CONFIGURAÇÕES E PARÂMETROS
# ==============================================================================

# --- Defina o caminho das imagens e da pasta de saída ---
# ATENÇÃO: Adapte estes caminhos para a sua estrutura de pastas local
# O CAMINHO DEVE SER ABSOLUTO OU RELATIVO, E COM BARRAS DUPLAS (Windows) ou BARRA SIMPLES (Linux/Mac)
# Exemplo adaptado para um caminho local:
# O script está em: /projeto_ML_cat&dogs/001_extrator_de_caractecisticas/app.py

caminho_imagens = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "imagens"))
output_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "bases_geradas"))

# Criar a pasta de saída se ela não existir
os.makedirs(output_dir, exist_ok=True)

print("Pasta de imagens:", caminho_imagens)
print("Pasta de saída (CSVs):", output_dir)

# extensões de imagem aceitas
IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")

# Parâmetros para LBP
lbp_image_sizes = [(256, 256), (128, 128), (64, 64)]  # Adicionado 128x128 e 64x64
lbp_radii = [3, 6, 9, 12]
lbp_methods = ["uniform", "ror"]
multi_orientation_angles = [0, 45, 90, 135]

# Parâmetros para HOG
hog_image_sizes = [(256, 256), (128, 128), (64, 64)]  # Adicionado 64x64
pixels_per_cells = [(32, 32), (16, 16), (8, 8)]
orientations = 9
cells_per_block = (2, 2)
pca_variances = [0.90, 0.75]  # manter 90% e 75% da variância

# ==============================================================================
#                 FUNÇÕES AUXILIARES
# ==============================================================================


def is_image_file(fname):
    """Verifica se o nome de arquivo tem extensão de imagem aceita."""
    return fname.lower().endswith(IMG_EXTS)


def parse_labels_from_filename(fname):
    """Parser de rótulos pelo nome do arquivo (case-insensitive) - Lógica do projeto."""
    fname_low = fname.lower()
    # Detecção da 'raça'
    if "siamese" in fname_low:
        race = "Siamese"
    elif "bengal" in fname_low:
        race = "Bengal"
    elif "shiba" in fname_low:
        race = "Shiba_inu"
    elif "american" in fname_low:
        race = "american_bulldog"
    else:
        race = "Unknown"
    # Mapa raça -> animal
    if race in ["Siamese", "Bengal"]:
        animal = "cat"
        label = 0
    elif race in ["Shiba_inu", "american_bulldog"]:
        animal = "dog"
        label = 1
    else:
        animal = "unknown"
        label = -1
    return animal, label, race


def load_and_preprocess_image(path, size=(256, 256)):
    """Carrega e preprocessa imagem (RGB -> resized float image)."""
    img = Image.open(path).convert("RGB")
    img = np.array(img)
    img_resized = resize(img, size, anti_aliasing=True)  # retorna float [0..1]
    gray = rgb2gray(img_resized)  # float [0..1], shape (H,W)
    return img_resized, gray


def remove_zero_columns(df, prefix="feature"):
    """Remove colunas completamente zeradas de um DataFrame."""
    zero_mask = (df == 0).all(axis=0)
    zero_features = zero_mask[zero_mask].index.tolist()
    if zero_features:
        print(f"   → Removendo {len(zero_features)} {prefix} zeradas...")
        df = df.loc[:, ~zero_mask]
    return df


def remove_near_zero_columns(df, threshold=0.99, verbose=True):
    """
    Remove colunas com pouca variação usando threshold dinâmico baseado na amplitude de cada coluna.
    Mantido para LBP Multi-Orientation, conforme a lógica original.
    """
    cols_removed = []
    cols_kept = []
    for col in df.columns:
        col_data = df[col]
        # Threshold dinâmico: 1% do valor máximo da coluna (mínimo 1e-6)
        dynamic_threshold = max(1e-6, col_data.max() * 0.01)
        # Calcula a proporção de valores abaixo do threshold
        near_zero_ratio = (col_data < dynamic_threshold).mean()

        if near_zero_ratio > threshold:
            cols_removed.append((col, near_zero_ratio, col_data.max()))
        else:
            cols_kept.append((col, near_zero_ratio, col_data.max()))

    if cols_removed and verbose:
        print(
            f"   → Removendo {len(cols_removed)} colunas quase zeradas (LBP Multi-O.)..."
        )

    cols_to_keep = [col for col, _, _ in cols_kept]
    return df[cols_to_keep]


# ==============================================================================
#                 LISTAR ARQUIVOS E CONTAR RÓTULOS
# ==============================================================================

files = sorted([f for f in os.listdir(caminho_imagens) if is_image_file(f)])
print(f"Total de arquivos de imagem encontrados: {len(files)}")

# Contagem por rótulo
from collections import Counter

counts = Counter()
for f in files:
    _, _, race = parse_labels_from_filename(f)
    counts[race] += 1
print("Contagem por raça (detectada pelo nome do arquivo):")
for k, v in counts.items():
    print(f"  {k}: {v}")

if not files:
    print("ERRO: Nenhuma imagem encontrada. Verifique o caminho das imagens.")
    exit()

# ==============================================================================
#                 EXTRAÇÃO DE LBP PADRÃO (uniform e ror)
# ==============================================================================

for size_tuple in lbp_image_sizes:
    size_str = size_tuple[0]
    for lbp_method in lbp_methods:
        print(f"\n=== Extraindo LBP '{lbp_method}' para size={size_str}x{size_str} ===")
        for radius in lbp_radii:
            n_points = 8 * radius
            n_bins = n_points + 2
            csv_name = f"lbp_{size_str}_r{radius}_{lbp_method}.csv"
            csv_path = os.path.join(output_dir, csv_name)

            if os.path.exists(csv_path):
                print(f"Arquivo já existe, pulando: {csv_path}")
                continue

            print(f"--- Radius = {radius} (n_points={n_points}) ---")
            features = []
            meta = []

            for fname in tqdm(
                files, desc=f"LBP {lbp_method} r={radius} size={size_str}"
            ):
                filepath = os.path.join(caminho_imagens, fname)
                try:
                    _, gray = load_and_preprocess_image(filepath, size=size_tuple)
                    lbp = local_binary_pattern(
                        gray, n_points, radius, method=lbp_method
                    )
                    hist, _ = np.histogram(
                        lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
                    )
                    features.append(hist)

                    animal, label, race = parse_labels_from_filename(fname)
                    meta.append((fname, animal, label, race))
                except Exception as e:
                    print(f"Erro ao processar {fname}: {e}")

            colnames = [f"lbp_r{radius}_bin{i+1}" for i in range(n_bins)]
            df_feats = pd.DataFrame(features, columns=colnames)
            df_feats = remove_zero_columns(
                df_feats, prefix=f"LBP radius {radius} ({lbp_method})"
            )
            df_meta = pd.DataFrame(
                meta, columns=["filename", "animal", "label", "race"]
            )
            df_out = pd.concat([df_meta, df_feats], axis=1)

            df_out.to_csv(csv_path, index=False)
            print(f"Salvo: {csv_path}  (shape: {df_out.shape})")

# ==============================================================================
#                 EXTRAÇÃO DE LBP MULTI-ORIENTATION
# ==============================================================================

for size_tuple in lbp_image_sizes:
    size_str = size_tuple[0]
    print(f"\n=== Extraindo LBP Multi-Orientation para size={size_str}x{size_str} ===")
    for radius in lbp_radii:
        n_points = 8 * radius
        n_bins = n_points + 2
        csv_name = f"lbp_{size_str}_r{radius}_multiorientation.csv"
        csv_path = os.path.join(output_dir, csv_name)

        if os.path.exists(csv_path):
            print(f"Arquivo já existe, pulando: {csv_path}")
            continue

        print(f"--- Radius = {radius} (n_points={n_points}) ---")
        features = []
        meta = []
        H, W = size_tuple

        for fname in tqdm(files, desc=f"LBP Multi-O. r={radius} size={size_str}"):
            filepath = os.path.join(caminho_imagens, fname)
            try:
                _, gray_orig = load_and_preprocess_image(filepath, size=size_tuple)
                hist_concat = []

                for angle in multi_orientation_angles:
                    if angle != 0:
                        center = (W // 2, H // 2)
                        M = cv2.getRotationMatrix2D(center, angle, 1.0)
                        gray = cv2.warpAffine(
                            gray_orig, M, (W, H), flags=cv2.INTER_LINEAR
                        )
                    else:
                        gray = gray_orig

                    lbp = local_binary_pattern(gray, n_points, radius, method="uniform")
                    hist, _ = np.histogram(
                        lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
                    )
                    hist_concat.extend(hist)

                features.append(hist_concat)
                animal, label, race = parse_labels_from_filename(fname)
                meta.append((fname, animal, label, race))

            except Exception as e:
                print(f"Erro ao processar {fname}: {e}")

        # Monta DataFrame com os nomes das colunas
        colnames = []
        for angle in multi_orientation_angles:
            colnames += [f"lbp_r{radius}_a{angle}_bin{i+1}" for i in range(n_bins)]

        df_feats = pd.DataFrame(features, columns=colnames)
        df_feats = remove_near_zero_columns(df_feats, threshold=0.99, verbose=True)
        print(f"   → Shape final de features: {df_feats.shape}")

        df_meta = pd.DataFrame(meta, columns=["filename", "animal", "label", "race"])
        df_out = pd.concat([df_meta, df_feats], axis=1)

        df_out.to_csv(csv_path, index=False)
        print(f"Salvo: {csv_path}  (shape: {df_out.shape})")

# ==============================================================================
#                 EXTRAÇÃO DE HOG COM E SEM PCA
# ==============================================================================

for size_tuple in hog_image_sizes:
    size_str = size_tuple[0]
    for ppc in pixels_per_cells:
        ppc_str = f"{ppc[0]}x{ppc[1]}"
        print(f"\n=== Extraindo HOG: size={size_str} | pixels_per_cell={ppc_str} ===")

        # Checa se o arquivo sem PCA já existe
        csv_name_no_pca = f"hog_{size_str}_{ppc_str}_noPCA.csv"
        csv_path_no_pca = os.path.join(output_dir, csv_name_no_pca)

        # Checa se TODOS os arquivos PCA já existem para esta combinação
        pca_files_exist = all(
            os.path.exists(
                os.path.join(
                    output_dir, f"hog_{size_str}_{ppc_str}_pca{int(var*100)}.csv"
                )
            )
            for var in pca_variances
        )

        if os.path.exists(csv_path_no_pca) and pca_files_exist:
            print(
                f"Todos os arquivos HOG/PCA para size={size_str} e ppc={ppc_str} já existem. Pulando."
            )
            continue

        # --- Etapa de Extração HOG ---
        features = []
        meta = []
        for fname in tqdm(files, desc=f"HOG size={size_str} ppc={ppc_str}"):
            filepath = os.path.join(caminho_imagens, fname)
            try:
                _, gray = load_and_preprocess_image(filepath, size=size_tuple)
                fd = hog(
                    gray,
                    orientations=orientations,
                    pixels_per_cell=ppc,
                    cells_per_block=cells_per_block,
                    block_norm="L2-Hys",
                    visualize=False,
                    feature_vector=True,
                )
                features.append(fd)
                animal, label, race = parse_labels_from_filename(fname)
                meta.append((fname, animal, label, race))
            except Exception as e:
                print(f"Erro ao processar {fname}: {e}")

        X = np.vstack(features)
        df_meta = pd.DataFrame(meta, columns=["filename", "animal", "label", "race"])

        # --- 1) Salvar versão sem PCA ---
        if not os.path.exists(csv_path_no_pca):
            os.makedirs(os.path.dirname(csv_path_no_pca), exist_ok=True)

            df_feats = pd.DataFrame(
                X, columns=[f"hog_f{i+1}" for i in range(X.shape[1])]
            )
            df_feats = remove_zero_columns(df_feats, prefix="features HOG")
            df_out = pd.concat([df_meta, df_feats], axis=1)
            df_out.to_csv(csv_path_no_pca, index=False)
            file_size = os.path.getsize(csv_path_no_pca) / 1024
            print(
                f"✅ Salvo (sem PCA): {csv_path_no_pca}  (shape: {df_out.shape}, size: {file_size:.1f} KB)"
            )
        else:
            print(f"Arquivo HOG sem PCA já existe: {csv_path_no_pca}")

        # --- 2) Normalizar antes de aplicar PCA ---
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
        original_features = X_scaled.shape[1]

        # --- 3) PCA com 90% e 75% ---
        for var in pca_variances:
            pca_int = int(var * 100)
            csv_name_pca = f"hog_{size_str}_{ppc_str}_pca{pca_int}.csv"
            csv_path_pca = os.path.join(output_dir, csv_name_pca)

            if os.path.exists(csv_path_pca):
                print(f"Arquivo PCA {var:.0%} já existe, pulando: {csv_path_pca}")
                continue

            try:
                pca = PCA(n_components=var)
                X_pca = pca.fit_transform(X_scaled)
                n_components = pca.n_components_
                explained_variance = pca.explained_variance_ratio_.sum()

                print(
                    f"PCA {var:.0%} variância: {n_components}/{original_features} componentes"
                )
                print(f"Variância total explicada: {explained_variance:.3%}")

                df_pcs = pd.DataFrame(
                    X_pca, columns=[f"pc{i+1}" for i in range(n_components)]
                )
                df_pcs = remove_zero_columns(df_pcs, prefix="PCs")

                df_out = pd.concat([df_meta, df_pcs], axis=1)
                df_out.to_csv(csv_path_pca, index=False)
                print(f"Salvo: {csv_path_pca}  (shape: {df_out.shape})")

            except Exception as e:
                print(
                    f"Falha aplicando PCA var={var} para size={size_str} ppc={ppc_str}: {e}"
                )

# ==============================================================================
#                 VISUALIZAÇÃO DE AMOSTRAS (APÓS EXTRAÇÃO)
# ==============================================================================

print("\n--- Visualização de Amostras de Imagem, LBP e HOG ---")
sample_files = random.sample(files, min(6, len(files)))

fig, axes = plt.subplots(len(sample_files), 3, figsize=(12, 4 * len(sample_files)))
if (
    len(sample_files) == 1
):  # Garante que 'axes' seja bidimensional mesmo com uma amostra
    axes = np.expand_dims(axes, axis=0)

for i, fname in enumerate(sample_files):
    path = os.path.join(caminho_imagens, fname)
    img_rgb, gray = load_and_preprocess_image(path, size=(256, 256))

    # LBP de exemplo (radius 6)
    n_points = 8 * 6
    lbp = local_binary_pattern(gray, n_points, 6, method="uniform")

    # HOG de exemplo (size 256, ppc 16x16)
    fd, hog_image = hog(
        gray,
        orientations=9,
        pixels_per_cell=(16, 16),
        cells_per_block=(2, 2),
        visualize=True,
        feature_vector=True,
    )

    ax0 = axes[i, 0]
    ax1 = axes[i, 1]
    ax2 = axes[i, 2]
    ax0.imshow(img_rgb)
    ax0.set_title(f"{fname}\n(orig)")
    ax0.axis("off")
    ax1.imshow(lbp, cmap="gray")
    ax1.set_title("LBP (r=6)")
    ax1.axis("off")
    ax2.imshow(hog_image, cmap="gray")
    ax2.set_title("HOG (ppc=16x16)")
    ax2.axis("off")
plt.tight_layout()
plt.show()

# ==============================================================================
#                 LISTAGEM DE ARQUIVOS GERADOS
# ==============================================================================

generated = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(".csv")])
print(f"\nTotal de CSVs gerados na pasta {output_dir}: {len(generated)}")
for fn in generated:
    print(" -", fn)
