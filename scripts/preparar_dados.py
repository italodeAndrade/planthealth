"""
Preparação e divisão dos dados – PjBL2 (batata, milho e tomate)
================================================================
Protocolo experimental (Seção 4.1 do plano de projeto), aplicado a CADA planta separadamente:
  * Divisão ESTRATIFICADA 70% treino / 15% validação / 15% teste.
  * Semente fixa (42) para que qualquer pessoa obtenha exatamente a mesma divisão.
  * A divisão é feita ANTES de qualquer aumento de dados (data augmentation):
    o aumento só é aplicado, depois, em memória e apenas no conjunto de treino.
  * Imagens idênticas (mesmo hash MD5) são mantidas juntas no mesmo conjunto,
    para evitar vazamento entre treino e teste.
  * LIMITE DE TREINO (só tomate, por tempo de CPU): depois da divisão, sorteia-se
    (semente 42) no máximo N imagens de TREINO por classe -> treino_limitado.csv.
    Validação e teste NÃO são reduzidos: continuam com 15% + 15% de todas as imagens.

Uso:  python scripts/preparar_dados.py                 # todas as plantas
      python scripts/preparar_dados.py milho tomate    # só algumas
Entrada: dados/plantvillage/color/<classe>/*   (ver scripts/baixar_dados.sh)
Saída (por planta): dados/splits/<planta>/{treino,validacao,teste}.csv  (caminho,classe)
                    dados/splits/<planta>/treino_limitado.csv           (se houver limite)
                    dados/splits/<planta>/contagem.csv e contagem.json
         modelos/<planta>.classes.json  {"classes": [...]} na ordem das saídas do modelo
                                        (formato lido por app/inferencia.py)
"""
import hashlib
import json
import sys
from pathlib import Path

import pandas as pd
from sklearn.model_selection import train_test_split

SEMENTE = 42
RAIZ = Path(__file__).resolve().parents[1]
ORIGEM = RAIZ / "dados" / "plantvillage" / "color"

# Ordem das classes = ordem alfabética dos nomes das pastas (a mesma usada no treino e no ONNX).
# Para a batata, é a ordem de CLASSES_BATATA em app/inferencia.py.
PLANTAS = {
    "batata": {"classes": ["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"],
               "max_treino_por_classe": None},
    "milho": {"classes": ["Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot", "Corn_(maize)___Common_rust_",
                          "Corn_(maize)___Northern_Leaf_Blight", "Corn_(maize)___healthy"],
              "max_treino_por_classe": None},
    "tomate": {"classes": ["Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___Late_blight",
                           "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
                           "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
                           "Tomato___Tomato_Yellow_Leaf_Curl_Virus", "Tomato___Tomato_mosaic_virus",
                           "Tomato___healthy"],
               "max_treino_por_classe": 800},   # ~18 mil imagens: limite só no TREINO, por tempo de CPU
}


def md5(caminho: Path) -> str:
    return hashlib.md5(caminho.read_bytes()).hexdigest()


def preparar(planta: str) -> None:
    cfg = PLANTAS[planta]
    classes, limite = cfg["classes"], cfg["max_treino_por_classe"]
    saida = RAIZ / "dados" / "splits" / planta
    print(f"\n===== {planta.upper()} =====")

    linhas = []
    for classe in classes:
        pasta = ORIGEM / classe
        arquivos = sorted(p for p in pasta.iterdir() if p.suffix.lower() in {".jpg", ".jpeg", ".png"}) \
            if pasta.exists() else []
        if not arquivos:
            raise SystemExit(f"Nenhuma imagem em {pasta}. Rode: bash scripts/baixar_dados.sh {planta}")
        for p in arquivos:
            linhas.append({"caminho": p.relative_to(RAIZ).as_posix(), "classe": classe, "md5": md5(p)})
    df = pd.DataFrame(linhas)

    # 1) Contagem por classe (mostra o desequilíbrio da base)
    print("Imagens por classe:\n", df["classe"].value_counts().reindex(classes).to_string())

    # 2) Agrupa duplicatas exatas: a divisão é feita por grupo (hash), não por arquivo.
    #    Se o mesmo conteúdo aparecer em classes diferentes, fica com a primeira (caso raro).
    grupos = df.drop_duplicates("md5")[["md5", "classe"]]
    print(f"Arquivos: {len(df)} | imagens únicas (MD5): {len(grupos)}")

    # 3) Divisão estratificada em duas etapas: 70% treino / 30% resto; resto -> 50/50
    g_treino, g_resto = train_test_split(grupos, test_size=0.30, stratify=grupos["classe"], random_state=SEMENTE)
    g_val, g_teste = train_test_split(g_resto, test_size=0.50, stratify=g_resto["classe"], random_state=SEMENTE)

    saida.mkdir(parents=True, exist_ok=True)
    conjuntos = {"treino": g_treino, "validacao": g_val, "teste": g_teste}
    contagem, partes = {}, {}
    for nome, g in conjuntos.items():
        parte = df[df["md5"].isin(g["md5"])].sort_values(["classe", "caminho"])
        parte[["caminho", "classe"]].to_csv(saida / f"{nome}.csv", index=False)
        partes[nome] = parte
        contagem[nome] = parte["classe"].value_counts().reindex(classes).fillna(0).astype(int).to_dict()

    # 4) Limite opcional de imagens de TREINO por classe (sorteio com semente fixa)
    if limite:
        tr = partes["treino"]
        limitado = pd.concat([tr[tr["classe"] == c].sample(n=min((tr["classe"] == c).sum(), limite),
                                                           random_state=SEMENTE) for c in classes]
                             ).sort_values(["classe", "caminho"])
        limitado[["caminho", "classe"]].to_csv(saida / "treino_limitado.csv", index=False)
        contagem["treino_usado"] = limitado["classe"].value_counts().reindex(classes).fillna(0).astype(int).to_dict()

    tabela = pd.DataFrame(contagem).loc[classes]
    tabela["total"] = tabela[["treino", "validacao", "teste"]].sum(axis=1)
    tabela.loc["total"] = tabela.sum()
    tabela.to_csv(saida / "contagem.csv", index_label="classe")
    with open(saida / "contagem.json", "w", encoding="utf-8") as f:
        json.dump({"planta": planta, "semente": SEMENTE,
                   "proporcao": {"treino": 0.70, "validacao": 0.15, "teste": 0.15},
                   "max_treino_por_classe": limite,
                   "fonte": "PlantVillage (raw/color) – github.com/spMohanty/PlantVillage-Dataset",
                   "contagem": tabela.to_dict(orient="index")}, f, ensure_ascii=False, indent=2)
    print("Contagem por classe e conjunto:\n", tabela.to_string())

    # 5) Ordem das classes para o app (app/inferencia.py lê modelos/<planta>.classes.json)
    assert classes == sorted(classes), "a ordem das classes deve ser a alfabética (sorted) das pastas"
    (RAIZ / "modelos").mkdir(exist_ok=True)
    with open(RAIZ / "modelos" / f"{planta}.classes.json", "w", encoding="utf-8") as f:
        json.dump({"planta": planta, "classes": classes}, f, ensure_ascii=False, indent=2)
        f.write("\n")

    # Verificação: nenhum arquivo em mais de um conjunto
    vistos = [set(partes[n]["caminho"]) for n in conjuntos]
    assert not (vistos[0] & vistos[1] or vistos[0] & vistos[2] or vistos[1] & vistos[2])
    print("OK: conjuntos disjuntos. Arquivos salvos em", saida.relative_to(RAIZ))


if __name__ == "__main__":
    escolhidas = sys.argv[1:] or list(PLANTAS)
    for p in escolhidas:
        if p not in PLANTAS:
            raise SystemExit(f"Planta desconhecida: {p}. Opções: {', '.join(PLANTAS)}")
        preparar(p)
