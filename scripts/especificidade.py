"""
Especificidade por classe (um-contra-todos) – PjBL2
===================================================
Calcula a ESPECIFICIDADE de cada classe a partir da matriz de confusão de TESTE já salva
em modelos/metricas_<planta>.json (não retreina nada e não usa o conjunto de teste de novo).

Fórmula (slides da disciplina, Aula 03, p. 29):   Especificidade = VN / (VN + FP)

Num problema com várias classes, cada classe c é tratada como "positiva" e todas as outras
como "negativas" (um-contra-todos). Na matriz M (linhas = classe real, colunas = prevista):
    VP = M[c, c]
    FN = soma da linha c    - VP   (era c, o modelo disse outra classe)
    FP = soma da coluna c   - VP   (o modelo disse c, mas era outra classe)
    VN = total - VP - FN - FP      (não era c e o modelo não disse c)
A especificidade macro é a média simples entre as classes.

Uso:  python scripts/especificidade.py            # batata, milho e tomate
Saída: grava, em cada modelos/metricas_<planta>.json,
       teste.por_classe[<classe>].especificidade e teste.especificidade_macro
"""
import json
import sys
from pathlib import Path

import numpy as np

RAIZ = Path(__file__).resolve().parents[1]


def especificidades(matriz: np.ndarray) -> np.ndarray:
    total = matriz.sum()
    vp = np.diag(matriz)
    fn = matriz.sum(axis=1) - vp
    fp = matriz.sum(axis=0) - vp
    vn = total - vp - fn - fp
    return vn / (vn + fp)


def processar(planta: str) -> None:
    caminho = RAIZ / "modelos" / f"metricas_{planta}.json"
    with open(caminho, encoding="utf-8") as f:
        m = json.load(f)
    classes = m["matriz_confusao"]["classes"]
    matriz = np.array(m["matriz_confusao"]["valores"])
    assert matriz.shape == (len(classes), len(classes)) and classes == m["classes"]

    esp = especificidades(matriz)
    for c, e in zip(classes, esp):
        m["teste"]["por_classe"][c]["especificidade"] = float(e)
    m["teste"]["especificidade_macro"] = float(esp.mean())

    with open(caminho, "w", encoding="utf-8") as f:
        json.dump(m, f, ensure_ascii=False, indent=2)
    print(f"\n{planta}: especificidade macro = {esp.mean():.4f}")
    for c, e in zip(classes, esp):
        print(f"  {c:50s} {e:.4f}")


if __name__ == "__main__":
    for p in sys.argv[1:] or ["batata", "milho", "tomate"]:
        processar(p)
