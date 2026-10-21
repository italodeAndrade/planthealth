"""
Preparação e divisão dos dados – PjBL2 (batata)
================================================
Protocolo experimental (Seção 4.1 do plano de projeto):
  * Divisão ESTRATIFICADA 70% treino / 15% validação / 15% teste.
  * Semente fixa (42) para que qualquer pessoa obtenha exatamente a mesma divisão.
  * A divisão é feita ANTES de qualquer aumento de dados (data augmentation):
    o aumento só é aplicado, depois, em memória e apenas no conjunto de treino.
  * Imagens idênticas (mesmo hash MD5) são mantidas juntas no mesmo conjunto,
    para evitar vazamento entre treino e teste.

Uso:  python scripts/preparar_dados.py
Entrada: dados/plantvillage/color/<classe>/*.JPG   (ver scripts/baixar_dados.sh)
Saída:   dados/splits/{treino,validacao,teste}.csv  (caminho,classe)
         dados/splits/contagem.csv e contagem.json  (imagens por classe e conjunto)
"""
import hashlib
import json
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEMENTE = 42
RAIZ = Path(__file__).resolve().parents[1]
ORIGEM = RAIZ / "dados" / "plantvillage" / "color"
SAIDA = RAIZ / "dados" / "splits"
# Ordem das classes = ordem alfabética = ordem esperada por app/inferencia.py
CLASSES = ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"]


def md5(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def main() -> None:
    linhas = []
    for classe in CLASSES:
        arquivos = sorted(p for p in (ORIGEM / classe).iterdir()
                          if p.suffix.lower() in {".jpg", ".jpeg", ".png"})
        if not arquivos:
            raise SystemExit(f"Nenhuma imagem em {ORIGEM / classe}. Rode scripts/baixar_dados.sh")
        for p in arquivos:
            linhas.append({"caminho": p.relative_to(RAIZ).as_posix(), "classe": classe, "md5": md5(p)})
    df = pd.DataFrame(linhas)

    # 1) Contagem por classe (verifica o desequilíbrio da base)
    print("Imagens por classe:\n", df["classe"].value_counts().reindex(CLASSES), "\n")

    # 2) Agrupa duplicatas exatas: a divisão é feita por grupo (hash), não por arquivo
    grupos = df.drop_duplicates("md5")[["md5", "classe"]]
    print(f"Arquivos: {len(df)} | imagens únicas: {len(grupos)}")

    # 3) Divisão estratificada em duas etapas: 70% treino / 30% resto; resto -> 50/50
    g_treino, g_resto = train_test_split(grupos, test_size=0.30, stratify=grupos["classe"], random_state=SEMENTE)
    g_val, g_teste = train_test_split(g_resto, test_size=0.50, stratify=g_resto["classe"], random_state=SEMENTE)

    SAIDA.mkdir(parents=True, exist_ok=True)
    conjuntos = {"treino": g_treino, "validacao": g_val, "teste": g_teste}
    contagem = {}
    for nome, g in conjuntos.items():
        parte = df[df["md5"].isin(g["md5"])].sort_values(["classe", "caminho"])
        parte[["caminho", "classe"]].to_csv(SAIDA / f"{nome}.csv", index=False)
        contagem[nome] = parte["classe"].value_counts().reindex(CLASSES).fillna(0).astype(int).to_dict()

    tabela = pd.DataFrame(contagem).loc[CLASSES]
    tabela["total"] = tabela.sum(axis=1)
    tabela.loc["total"] = tabela.sum()
    tabela.to_csv(SAIDA / "contagem.csv", index_label="classe")
    with open(SAIDA / "contagem.json", "w", encoding="utf-8") as f:
        json.dump({"semente": SEMENTE, "proporcao": {"treino": 0.70, "validacao": 0.15, "teste": 0.15},
                   "fonte": "PlantVillage (raw/color) – github.com/spMohanty/PlantVillage-Dataset",
                   "contagem": tabela.to_dict(orient="index")}, f, ensure_ascii=False, indent=2)
    print("\nContagem por classe e conjunto:\n", tabela)

    # Verificação: nenhum arquivo em mais de um conjunto
    vistos = [set(pd.read_csv(SAIDA / f"{n}.csv")["caminho"]) for n in conjuntos]
    assert not (vistos[0] & vistos[1] or vistos[0] & vistos[2] or vistos[1] & vistos[2])
    print("\nOK: conjuntos disjuntos. Arquivos salvos em", SAIDA)


if __name__ == "__main__":
    main()
