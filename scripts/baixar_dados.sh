#!/usr/bin/env bash
# =====================================================================
# Baixa as imagens COLORIDAS de batata do PlantVillage (PjBL2)
# Fonte pública: https://github.com/spMohanty/PlantVillage-Dataset
# (pasta raw/color, mesma base de Hughes & Salathé, 2015)
# Usa "sparse checkout" para baixar só as 3 pastas de batata (~40 MB).
# Uso: bash scripts/baixar_dados.sh
# Resultado: dados/plantvillage/color/Potato___{Early_blight,Late_blight,healthy}/
# As imagens NÃO são versionadas (ver .gitignore).
# =====================================================================
set -euo pipefail
RAIZ="$(cd "$(dirname "$0")/.." && pwd)"
DESTINO="$RAIZ/dados/plantvillage/color"
COMMIT="7f7ecc7e1eaca78107e3affe7cb5abd9427e139a"   # commit fixo para reprodutibilidade
TMP="$(mktemp -d)"
trap 'rm -rf "$TMP"' EXIT

git clone -q --filter=blob:none --sparse --no-checkout \
    https://github.com/spMohanty/PlantVillage-Dataset.git "$TMP/pv"
cd "$TMP/pv"
git sparse-checkout set "raw/color/Potato___Early_blight" \
                        "raw/color/Potato___Late_blight" \
                        "raw/color/Potato___healthy"
git checkout -q "$COMMIT"

mkdir -p "$DESTINO"
for c in Potato___Early_blight Potato___Late_blight Potato___healthy; do
    rm -rf "${DESTINO:?}/$c"
    cp -r "raw/color/$c" "$DESTINO/$c"
    echo "$c: $(ls "$DESTINO/$c" | wc -l) imagens"
done
echo "Imagens salvas em $DESTINO"
