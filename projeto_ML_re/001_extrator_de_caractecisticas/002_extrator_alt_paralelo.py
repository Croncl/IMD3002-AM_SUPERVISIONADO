import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from skimage.color import rgb2gray
from skimage.transform import resize
from skimage.feature import local_binary_pattern, hog
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from tqdm import tqdm
import warnings
import os
import cv2
import random
from multiprocessing import Pool, cpu_count 
from collections import Counter

warnings.filterwarnings("ignore")

# ==============================================================================
#                       FUNÇÕES AUXILIARES DE PROCESSAMENTO (ESCOPO GLOBAL)
#    <--- MOVIMENTADAS PARA FORA DO if __name__ == '__main__': PARA CORRIGIR ERRO MP
# ==============================================================================

def is_image_file(fname):
    """Verifica se o nome de arquivo tem extensão de imagem aceita."""
    IMG_EXTS = (".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff")
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
    """
    Carrega e preprocessa imagem usando OpenCV (cv2).
    Retorna a imagem RGB redimensionada ([0..1]) e a imagem em tons de cinza ([0..1]).
    """
    img = cv2.imread(path, cv2.IMREAD_COLOR) 
    
    if img is None:
        raise FileNotFoundError(f"Erro ao carregar a imagem: {path}")

    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    img_float = img_rgb.astype(float) / 255.0
    # Redimensionamento usa scikit-image
    img_resized = resize(img_float, size, anti_aliasing=True) 
    
    # Conversão para tons de cinza usa scikit-image
    gray = rgb2gray(img_resized) 
    
    return img_resized, gray

# ==============================================================================
#                       FUNÇÕES PARA PROCESSAMENTO PARALELO (ESCOPO GLOBAL)
# ==============================================================================

def _extract_lbp_single(args):
    """Extrai LBP Padrão para uma única imagem (usado em Pool.starmap)."""
    fname, filepath, size_tuple, n_points, radius, lbp_method, n_bins = args
    try:
        # Usa a função global
        _, gray = load_and_preprocess_image(filepath, size=size_tuple)
        lbp = local_binary_pattern(
            gray, n_points, radius, method=lbp_method
        )
        hist, _ = np.histogram(
            lbp.ravel(), bins=n_bins, range=(0, n_bins), density=True
        )
        
        # Usa a função global
        animal, label, race = parse_labels_from_filename(fname)
        meta = (fname, animal, label, race)
        
        return (meta, hist)
    except Exception as e:
        # Retorna None em caso de erro para que o processo principal possa lidar
        print(f"Erro no worker LBP ao processar {fname}: {e}")
        return None

def _extract_lbp_multi_single(args):
    """Extrai LBP Multi-Orientation para uma única imagem (usado em Pool.starmap)."""
    fname, filepath, size_tuple, n_points, radius, n_bins, multi_orientation_angles = args
    try:
        H, W = size_tuple
        # Usa a função global
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
        
        # Usa a função global
        animal, label, race = parse_labels_from_filename(fname)
        meta = (fname, animal, label, race)
        
        return (meta, hist_concat)
    except Exception as e:
        print(f"Erro no worker LBP Multi-O ao processar {fname}: {e}")
        return None

def _extract_hog_single(args):
    """Extrai HOG para uma única imagem (usado em Pool.starmap)."""
    fname, filepath, size_tuple, orientations, ppc, cpb = args
    try:
        # Usa a função global
        _, gray = load_and_preprocess_image(filepath, size=size_tuple)
        fd = hog(
            gray,
            orientations=orientations,
            pixels_per_cell=ppc,
            cells_per_block=cpb,
            block_norm="L2-Hys",
            visualize=False,
            feature_vector=True,
        )
        
        # Usa a função global
        animal, label, race = parse_labels_from_filename(fname)
        meta = (fname, animal, label, race)
        
        return (meta, fd)
    except Exception as e:
        print(f"Erro no worker HOG ao processar {fname}: {e}")
        return None

# ==============================================================================
#                       FUNÇÕES AUXILIARES DE LIMPEZA (ESCOPO GLOBAL)
# ==============================================================================

def remove_zero_columns(df, prefix="feature"):
    """Remove colunas completamente zeradas de um DataFrame."""
    # Assume que as 4 primeiras colunas são metadados
    if df.shape[1] <= 4:
        return df
        
    df_meta = df.iloc[:, :4]
    df_features = df.iloc[:, 4:]
    
    # Máscara de colunas zeradas
    zero_mask = (df_features == 0).all(axis=0)
    zero_features = zero_mask[zero_mask].index.tolist()
    
    if zero_features:
        print(f"    → Removendo {len(zero_features)} {prefix} zeradas...")
        # Remove as colunas zeradas apenas do DataFrame de features
        df_features = df_features.loc[:, ~zero_mask]
        
        # Reconcatena com as colunas de metadados
        return pd.concat([df_meta, df_features], axis=1)
    
    return df


def remove_near_zero_columns(df, threshold=0.99, verbose=True):
    """
    Remove colunas com pouca variação usando threshold dinâmico.
    Aplica a lógica apenas nas colunas de features (após as 4 primeiras colunas de metadados).
    """
    if df.shape[1] <= 4:
        return df
        
    df_meta = df.iloc[:, :4]
    df_features = df.iloc[:, 4:]
    
    cols_removed = []
    cols_to_keep = []
    
    for col in df_features.columns:
        col_data = df_features[col]
        # Dynamic threshold based on 1% of the max value, minimum 1e-6
        dynamic_threshold = max(1e-6, col_data.max() * 0.01)
        # Ratio of values near zero
        near_zero_ratio = (col_data < dynamic_threshold).mean() 

        if near_zero_ratio > threshold:
            cols_removed.append((col, near_zero_ratio, col_data.max()))
        else:
            cols_to_keep.append(col)

    if cols_removed and verbose:
        print(
            f"    → Removendo {len(cols_removed)} colunas quase zeradas (LBP Multi-O.)..."
        )

    df_features = df_features[cols_to_keep]
    return pd.concat([df_meta, df_features], axis=1)


# ==============================================================================
#                      BLOCO PRINCIPAL DE EXECUÇÃO (CORRIGIDO)
# ==============================================================================

if __name__ == '__main__':

    # ==============================================================================
    #                       CONFIGURAÇÕES E PARÂMETROS
    # ==============================================================================

    # --- Defina o caminho das imagens e da pasta de saída ---
    
    SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
    caminho_imagens = os.path.join(SCRIPT_DIR, "imagensCD")
    output_dir = os.path.join(SCRIPT_DIR, "bases_geradas")

    N_CORES = cpu_count()
    print(f"Detectado {N_CORES} núcleos de CPU para processamento paralelo.")

    os.makedirs(output_dir, exist_ok=True)

    print("Pasta do Script:", SCRIPT_DIR)
    print("Pasta de imagens:", caminho_imagens)
    print("Pasta de saída (CSVs):", output_dir)

    # --- Combinações de Parâmetros ---

    # Parâmetros para LBP
    lbp_image_sizes = [(256, 256), (128, 128), (64, 64)]
    lbp_radii = [6, 9, 12]
    lbp_methods = ["uniform"] 
    lbp_n_points_modes = ["dynamic_8r", "fixed_8"] 

    multi_orientation_angles = [0, 45, 90, 135]

    # Parâmetros para HOG
    hog_image_sizes = [(256, 256), (128, 128), (64, 64)]
    pixels_per_cells = [(32, 32), (16, 16), (8, 8)]
    hog_orientations = [9, 12, 18]
    cells_per_block_list = [(2, 2), (3, 3)] 
    pca_variances = [0.90, 0.75] 

    # ==============================================================================
    #                       LISTAR ARQUIVOS E CONTAR RÓTULOS
    # ==============================================================================

    # is_image_file está no escopo global
    files = sorted([f for f in os.listdir(caminho_imagens) if is_image_file(f)])
    print(f"Total de arquivos de imagem encontrados: {len(files)}")

    counts = Counter()
    for f in files:
        # parse_labels_from_filename está no escopo global
        _, _, race = parse_labels_from_filename(f)
        counts[race] += 1
    print("Contagem por raça (detectada pelo nome do arquivo):")
    for k, v in counts.items():
        print(f"  {k}: {v}")

    if not files:
        print("ERRO: Nenhuma imagem encontrada. Verifique o caminho das imagens.")
        pass 
    else:
        # ==============================================================================
        #                       EXTRAÇÃO DE LBP PADRÃO (com processamento paralelo)
        # ==============================================================================

        for size_tuple in lbp_image_sizes:
            size_str = size_tuple[0]
            for lbp_method in lbp_methods:
                for n_points_mode in lbp_n_points_modes:
                    print(f"\n=== Extraindo LBP '{lbp_method}' para size={size_str}x{size_str} (Mode: {n_points_mode}) ===")
                    for radius in lbp_radii:
                        
                        if n_points_mode == "dynamic_8r":
                            n_points = 8 * radius
                            csv_suffix = f"r{radius}_dynamicP"
                        elif n_points_mode == "fixed_8":
                            n_points = 8 
                            csv_suffix = f"r{radius}_P8"
                            
                        n_bins = n_points + 2
                        
                        csv_name = f"lbp_{size_str}_{csv_suffix}_{lbp_method}.csv"
                        csv_path = os.path.join(output_dir, csv_name)

                        if os.path.exists(csv_path):
                            print(f"Arquivo já existe, pulando: {csv_path}")
                            continue

                        print(f"--- Radius = {radius} (n_points={n_points}) ---")
                        
                        # --- Preparação para Paralelismo ---
                        tasks = []
                        for fname in files:
                            filepath = os.path.join(caminho_imagens, fname)
                            tasks.append((fname, filepath, size_tuple, n_points, radius, lbp_method, n_bins))

                        # --- Execução Paralela ---
                        # _extract_lbp_single está no escopo global
                        with Pool(N_CORES) as pool:
                            results = list(tqdm(
                                pool.imap(_extract_lbp_single, tasks),
                                total=len(tasks),
                                desc=f"LBP {lbp_method} r={radius} P={n_points} size={size_str} (Paralelo)"
                            ))
                        
                        # --- Coleta e Salvamento ---
                        features = []
                        meta = []
                        for res in results:
                            if res is not None:
                                meta.append(res[0])
                                features.append(res[1])

                        colnames = [f"lbp_{csv_suffix}_bin{i+1}" for i in range(n_bins)]
                        df_feats = pd.DataFrame(features, columns=colnames)
                        df_meta = pd.DataFrame(
                            meta, columns=["filename", "animal", "label", "race"]
                        )
                        df_out = pd.concat([df_meta, df_feats], axis=1)
                        
                        # remove_zero_columns está no escopo global
                        df_out = remove_zero_columns(
                            df_out, prefix=f"LBP {csv_suffix} ({lbp_method})"
                        )
                        
                        df_out.to_csv(csv_path, index=False)
                        print(f"Salvo: {csv_path}  (shape: {df_out.shape})")

        # ==============================================================================
        #                       EXTRAÇÃO DE LBP MULTI-ORIENTATION (com processamento paralelo)
        # ==============================================================================

        for size_tuple in lbp_image_sizes:
            size_str = size_tuple[0]
            print(f"\n=== Extraindo LBP Multi-Orientation para size={size_str}x{size_str} (Paralelo) ===")
            for radius in lbp_radii:
                n_points = 8 * radius
                n_bins = n_points + 2
                csv_name = f"lbp_{size_str}_r{radius}_multiorientation.csv"
                csv_path = os.path.join(output_dir, csv_name)

                if os.path.exists(csv_path):
                    print(f"Arquivo já existe, pulando: {csv_path}")
                    continue

                print(f"--- Radius = {radius} (n_points={n_points}) ---")

                # --- Preparação para Paralelismo ---
                tasks = []
                for fname in files:
                    filepath = os.path.join(caminho_imagens, fname)
                    tasks.append((fname, filepath, size_tuple, n_points, radius, n_bins, multi_orientation_angles))

                # --- Execução Paralela ---
                # _extract_lbp_multi_single está no escopo global
                with Pool(N_CORES) as pool:
                    results = list(tqdm(
                        pool.imap(_extract_lbp_multi_single, tasks),
                        total=len(tasks),
                        desc=f"LBP Multi-O. r={radius} size={size_str} (Paralelo)"
                    ))

                # --- Coleta e Salvamento ---
                features = []
                meta = []
                for res in results:
                    if res is not None:
                        meta.append(res[0])
                        features.append(res[1])
                
                colnames = []
                for angle in multi_orientation_angles:
                    colnames += [f"lbp_r{radius}_a{angle}_bin{i+1}" for i in range(n_bins)]

                df_feats = pd.DataFrame(features, columns=colnames)
                df_meta = pd.DataFrame(meta, columns=["filename", "animal", "label", "race"])
                df_out = pd.concat([df_meta, df_feats], axis=1)

                # remove_near_zero_columns está no escopo global
                df_out = remove_near_zero_columns(df_out, threshold=0.99, verbose=True)
                print(f"    → Shape final de features após limpeza: {df_out.shape}")

                df_out.to_csv(csv_path, index=False)
                print(f"Salvo: {csv_path}  (shape: {df_out.shape})")

        # ==============================================================================
        #                       EXTRAÇÃO DE HOG COM E SEM PCA (com processamento paralelo)
        # ==============================================================================

        for size_tuple in hog_image_sizes:
            size_str = size_tuple[0]
            for n_orientations in hog_orientations:
                for ppc in pixels_per_cells:
                    ppc_str = f"{ppc[0]}x{ppc[1]}"
                    for cpb in cells_per_block_list:
                        cpb_str = f"{cpb[0]}x{cpb[1]}"
                        
                        print(f"\n=== Extraindo HOG: size={size_str} | ori={n_orientations} | ppc={ppc_str} | cpb={cpb_str} (Paralelo) ===")

                        base_name = f"hog_{size_str}_ori{n_orientations}_{ppc_str}_cpb{cpb_str}"
                        csv_name_no_pca = f"{base_name}_noPCA.csv"
                        csv_path_no_pca = os.path.join(output_dir, csv_name_no_pca)

                        pca_files_exist = all(
                            os.path.exists(
                                os.path.join(
                                    output_dir, f"{base_name}_pca{int(var*100)}.csv"
                                )
                            )
                            for var in pca_variances
                        )

                        if os.path.exists(csv_path_no_pca) and pca_files_exist:
                            print(
                                f"Todos os arquivos HOG/PCA para size={size_str}, ori={n_orientations}, ppc={ppc_str} e cpb={cpb_str} já existem. Pulando."
                            )
                            continue
                            
                        # --- Preparação para Paralelismo ---
                        tasks = []
                        for fname in files:
                            filepath = os.path.join(caminho_imagens, fname)
                            tasks.append((fname, filepath, size_tuple, n_orientations, ppc, cpb))

                        # --- Execução Paralela ---
                        # _extract_hog_single está no escopo global
                        with Pool(N_CORES) as pool:
                            results = list(tqdm(
                                pool.imap(_extract_hog_single, tasks),
                                total=len(tasks),
                                desc=f"HOG size={size_str} ori={n_orientations} ppc={ppc_str} cpb={cpb_str} (Paralelo)"
                            ))

                        # --- Coleta de Resultados ---
                        features = []
                        meta = []
                        for res in results:
                            if res is not None:
                                meta.append(res[0])
                                features.append(res[1])

                        X = np.vstack(features)
                        df_meta = pd.DataFrame(meta, columns=["filename", "animal", "label", "race"])

                        # --- 1) Salvar versão sem PCA ---
                        if not os.path.exists(csv_path_no_pca):
                            os.makedirs(os.path.dirname(csv_path_no_pca), exist_ok=True)

                            df_feats = pd.DataFrame(
                                X, columns=[f"hog_f{i+1}" for i in range(X.shape[1])]
                            )
                            df_out = pd.concat([df_meta, df_feats], axis=1)
                            
                            # remove_zero_columns está no escopo global
                            df_out = remove_zero_columns(df_out, prefix="features HOG")
                            
                            df_out.to_csv(csv_path_no_pca, index=False)
                            file_size = os.path.getsize(csv_path_no_pca) / 1024
                            print(
                                f"✅ Salvo (sem PCA): {csv_path_no_pca}  (shape: {df_out.shape}, size: {file_size:.1f} KB)"
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
                            csv_name_pca = f"{base_name}_pca{pca_int}.csv"
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
                                df_out = pd.concat([df_meta, df_pcs], axis=1)
                                
                                df_out = remove_zero_columns(df_out, prefix="PCs")
                                
                                df_out.to_csv(csv_path_pca, index=False)
                                print(f"Salvo: {csv_path_pca}  (shape: {df_out.shape})")

                            except Exception as e:
                                print(
                                    f"Falha aplicando PCA var={var} para size={size_str} ppc={ppc_str} cpb={cpb_str}: {e}"
                                )

        # ==============================================================================
        #                       VISUALIZAÇÃO DE AMOSTRAS (APÓS EXTRAÇÃO)
        # ==============================================================================

        print("\n--- Visualização de Amostras de Imagem, LBP e HOG ---")
        
        sample_files = random.sample(files, min(6, len(files)))

        fig, axes = plt.subplots(len(sample_files), 3, figsize=(12, 4 * len(sample_files)))
        if (
            len(sample_files) == 1
        ):
            axes = np.expand_dims(axes, axis=0)

        EX_ORIENTATIONS = hog_orientations[0] 

        for i, fname in enumerate(sample_files):
            path = os.path.join(caminho_imagens, fname)
            # load_and_preprocess_image está no escopo global
            img_rgb, gray = load_and_preprocess_image(path, size=(256, 256))

            n_points = 8 * 6
            lbp = local_binary_pattern(gray, n_points, 6, method="uniform")

            fd, hog_image = hog(
                gray,
                orientations=EX_ORIENTATIONS,
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
            ax1.set_title("LBP (r=6, P=48)")
            ax1.axis("off")
            
            ax2.imshow(hog_image, cmap="gray")
            ax2.set_title(f"HOG (ori={EX_ORIENTATIONS}, ppc=16x16, cpb=2x2)")
            ax2.axis("off")
            
        plt.tight_layout()
        plt.show() 

        # ==============================================================================
        #                       LISTAGEM DE ARQUIVOS GERADOS
        # ==============================================================================

        generated = sorted([f for f in os.listdir(output_dir) if f.lower().endswith(".csv")])
        print(f"\nTotal de CSVs gerados na pasta {output_dir}: {len(generated)}")
        for fn in generated:
            print(" -", fn)
