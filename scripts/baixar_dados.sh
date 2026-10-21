#!/usr/bin/env bash
# =====================================================================
# Baixa as imagens COLORIDAS do PlantVillage usadas no PjBL2
# Fonte pública: https://github.com/spMohanty/PlantVillage-Dataset
# (pasta raw/color, mesma base de Hughes & Salathé, 2015)
# Usa "sparse checkout" para baixar só as pastas das plantas pedidas.
#
# Uso:  bash scripts/baixar_dados.sh                 # batata, milho e tomate
#       bash scripts/baixar_dados.sh batata milho    # só algumas plantas
# Resultado: dados/plantvillage/color/<classe>/   (uma pasta por classe)
# As imagens NÃO são versionadas (ver .gitignore).
# Tamanhos aproximados: batata ~40 MB, milho ~70 MB, tomate ~330 MB.
# =====================================================================
set -euo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
DESTINO="$RAIZ/dados/plantvillage/color"
COMMIT="7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"   # commit fixo para reprodutibilidade

BATATA=("Potato___Early_blight" "Potato___Late_blight" "Potato___healthy")
MILHO=("Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot" "Corn_(maize)___Common_rust_"
       "Corn_(maize)___Northern_Leaf_Blight" "Corn_(maize)___healthy")
TOMATE=("Tomato___Bacterial_spot" "Tomato___Early_blight" "Tomato___Late_blight" "Tomato___Leaf_Mold"
        "Tomato___Septoria_leaf_spot" "Tomato___Spider_mites Two-spotted_spider_mite" "Tomato___Target_Spot"
        "Tomato___Tomato_Yellow_Leaf_Curl_Virus" "Tomato___Tomato_mosaic_virus" "Tomato___healthy")

PLANTAS=("$@"); [ ${#PLANTAS[@]} -eq 0 ] && PLANTAS=(batata milho tomate)
CLASSES=()
for p in "${PLANTAS[@]}"; do
    case "$p" in
        batata) CLASSES+=("${BATATA[@]}") ;;
        milho)  CLASSES+=("${MILHO[@]}") ;;
        tomate) CLASSES+=("${TOMATE[@]}") ;;
        *) echo "Planta desconhecida: $p (use batata, milho ou tomate)"; exit 1 ;;
    esac
done

TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT
git clone -q --filter=blob:none --sparse --no-checkout \
    https://github.com/spMohanty/PlantVillage-Dataset.git "$TMP/pv"
cd "$TMP/pv"
PASTAS=(); for c in "${CLASSES[@]}"; do PASTAS+=("raw/color/$c"); done
git sparse-checkout set "${PASTAS[@]}"
git checkout -q "$COMMIT"

mkdir -p "$DESTINO"
for c in "${CLASSES[@]}"; do
    rm -rf "${DESTINO:?}/$c"
    cp -r "raw/color/$c" "$DESTINO/$c"
    echo "$c: $(ls "$DESTINO/$c" | wc -l) imagens"
done
echo "Imagens salvas em $DESTINO"
