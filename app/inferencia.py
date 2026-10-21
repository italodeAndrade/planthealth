"""
Inferência local de visão computacional com roteamento por cultura.

Cada cultura (batata, tomate, milho) tem seu próprio modelo treinado offline
(Google Colab) e salvo em modelos/:

    modelos/<planta>.onnx            pesos exportados (saída = logits)
    modelos/<planta>.classes.json    {"classes": [...]} na ordem do treino (gerado pelo notebook)

O backend nunca consulta o Colab: se o arquivo do modelo não existir, a
requisição falha com ModeloIndisponivelError (nada é simulado).
"""
import json
import os
import threading
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODELOS_DIR = os.path.join(BASE_DIR, "modelos")

TAMANHO_ENTRADA = (224, 224)
MEDIA = np.array([0.485, 0.456, 0.406], dtype=np.float32)   # ImageNet (igual ao treino)
DESVIO = np.array([0.229, 0.224, 0.225], dtype=np.float32)

PLANTAS_SUPORTADAS = ("batata", "tomate", "milho")

# Usado apenas se <planta>.classes.json não existir. A ordem é a alfabética das
# pastas do dataset, a mesma que o notebook usa ao treinar (sorted).
CLASSES_PADRAO: Dict[str, List[str]] = {
    "batata": sorted([
        "Potato___Early_blight", "Potato___Late_blight", "Potato___healthy",
    ]),
    "milho": sorted([
        "Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot",
        "Corn_(maize)___Common_rust_",
        "Corn_(maize)___healthy",
        "Corn_(maize)___Northern_Leaf_Blight",
    ]),
    "tomate": sorted([
        "Tomato___Bacterial_spot", "Tomato___Early_blight", "Tomato___healthy",
        "Tomato___Late_blight", "Tomato___Leaf_Mold", "Tomato___Septoria_leaf_spot",
        "Tomato___Spider_mites Two-spotted_spider_mite", "Tomato___Target_Spot",
        "Tomato___Tomato_mosaic_virus", "Tomato___Tomato_Yellow_Leaf_Curl_Virus",
    ]),
}


class ModeloIndisponivelError(Exception):
    """Modelo da cultura ausente, inválido ou incompatível com as classes."""


# Cache: planta -> (mtime_do_arquivo, sessão ONNX, classes)
_cache: Dict[str, Tuple[float, Any, List[str]]] = {}
_lock = threading.Lock()


def _normalizar_planta(planta: str) -> str:
    slug = (planta or "").strip().lower()
    if slug not in PLANTAS_SUPORTADAS:
        raise ModeloIndisponivelError(
            f"Cultura '{planta}' não suportada. Use: {', '.join(PLANTAS_SUPORTADAS)}."
        )
    return slug


def caminho_modelo(planta: str) -> str:
    return os.path.join(MODELOS_DIR, f"{planta}.onnx")


def _carregar_classes(planta: str) -> List[str]:
    caminho = os.path.join(MODELOS_DIR, f"{planta}.classes.json")
    if os.path.exists(caminho):
        with open(caminho, "r", encoding="utf-8") as f:
            return list(json.load(f)["classes"])
    return list(CLASSES_PADRAO[planta])


def _provedores_onnx() -> List[str]:
    import onnxruntime as ort
    disponiveis = ort.get_available_providers()
    preferidos = [p for p in ("CUDAExecutionProvider", "CPUExecutionProvider") if p in disponiveis]
    return preferidos or ["CPUExecutionProvider"]


def carregar_modelo(planta: str) -> Tuple[Any, List[str]]:
    """Carrega (e mantém em cache) a sessão ONNX da cultura. Recarrega se o arquivo mudar."""
    planta = _normalizar_planta(planta)
    caminho = caminho_modelo(planta)
    if not os.path.exists(caminho):
        raise ModeloIndisponivelError(
            f"Modelo da cultura '{planta}' não encontrado (esperado em modelos/{planta}.onnx)."
        )

    mtime = os.path.getmtime(caminho)
    with _lock:
        em_cache = _cache.get(planta)
        if em_cache and em_cache[0] == mtime:
            return em_cache[1], em_cache[2]

        try:
            import onnxruntime as ort
            sessao = ort.InferenceSession(caminho, providers=_provedores_onnx())
        except Exception as e:
            raise ModeloIndisponivelError(f"Falha ao abrir modelos/{planta}.onnx: {e}")

        classes = _carregar_classes(planta)
        saida = sessao.get_outputs()[0].shape
        n_saidas = saida[-1] if saida else None
        if isinstance(n_saidas, int) and n_saidas != len(classes):
            raise ModeloIndisponivelError(
                f"modelos/{planta}.onnx tem {n_saidas} saídas, mas há {len(classes)} classes "
                f"configuradas. Copie também modelos/{planta}.classes.json gerado pelo notebook."
            )

        _cache[planta] = (mtime, sessao, classes)
        print(f"[INFERÊNCIA] Modelo '{planta}' carregado ({len(classes)} classes).")
        return sessao, classes


def modelos_disponiveis() -> Dict[str, Dict[str, Any]]:
    """Situação do modelo de cada cultura (usado em /api/modelos e /api/health)."""
    situacao = {}
    for planta in PLANTAS_SUPORTADAS:
        existe = os.path.exists(caminho_modelo(planta))
        situacao[planta] = {
            "disponivel": existe,
            "arquivo": f"modelos/{planta}.onnx",
            "classes": _carregar_classes(planta),
        }
    return situacao


def preprocessar_imagem(caminho_imagem: str) -> np.ndarray:
    """Resize direto 224x224 (bilinear, como o torchvision no treino), normalização ImageNet, NCHW."""
    img = Image.open(caminho_imagem).convert("RGB")
    img = img.resize(TAMANHO_ENTRADA, Image.BILINEAR)
    arr = np.asarray(img, dtype=np.float32) / 255.0
    arr = (arr - MEDIA) / DESVIO
    arr = np.transpose(arr, (2, 0, 1))
    return np.expand_dims(arr, axis=0).astype(np.float32)


def softmax(x: np.ndarray) -> np.ndarray:
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=-1, keepdims=True)


def eh_saudavel(classe: str) -> bool:
    c = classe.lower()
    return "healthy" in c or "saudavel" in c


def nome_legivel(classe: str) -> str:
    """'Potato___Early_blight' -> 'Early blight' (para o prompt da LLM)."""
    return classe.split("___")[-1].replace("_", " ").strip()


def classificar_foto(caminho_imagem: str, ordem: int, planta: str) -> Dict[str, Any]:
    """Classifica uma foto com o modelo da cultura informada."""
    sessao, classes = carregar_modelo(planta)
    try:
        tensor = preprocessar_imagem(caminho_imagem)
    except Exception as e:
        raise ValueError(f"Imagem {ordem} inválida ou corrompida: {e}")

    entrada = sessao.get_inputs()[0].name
    logits = sessao.run(None, {entrada: tensor})[0][0]
    probs = softmax(logits)
    idx = int(np.argmax(probs))
    return {
        "ordem": ordem,
        "caminho": caminho_imagem,
        "classe_prevista": classes[idx],
        "confianca": round(float(probs[idx]), 4),
        "probabilidades": {classes[i]: round(float(probs[i]), 4) for i in range(len(classes))},
    }


def combinar_diagnosticos(
    fotos_classificadas: List[Dict[str, Any]],
    limiar: float = 0.70,
) -> Tuple[str, int, str, List[str]]:
    """
    Regra de decisão sobre as 3 folhas:
    - 'doenca':   ao menos uma folha com classe doente e confiança >= limiar
                  (folhas_doentes = quantas folhas estão nessa condição);
    - 'incerto':  nenhuma doença confirmada, mas alguma foto abaixo do limiar
                  (pede novas fotos em vez de arriscar um diagnóstico);
    - 'saudavel': todas as folhas saudáveis com confiança >= limiar.

    Retorna: (status, folhas_doentes, orientacao_adicional, doencas_detectadas)
    """
    doentes_confiaveis = []
    baixa_confianca = []

    for foto in fotos_classificadas:
        confiante = foto.get("confianca", 0.0) >= limiar
        if not confiante:
            baixa_confianca.append(foto["ordem"])
        elif not eh_saudavel(foto["classe_prevista"]):
            doentes_confiaveis.append(foto)

    if doentes_confiaveis:
        doencas = sorted({nome_legivel(f["classe_prevista"]) for f in doentes_confiaveis})
        orientacao = (
            f"{len(doentes_confiaveis)} de {len(fotos_classificadas)} folhas apresentaram indícios de "
            f"doença com confiança >= {limiar * 100:.0f}% ({', '.join(doencas)})."
        )
        if baixa_confianca:
            fotos_txt = ", ".join(f"Foto {o}" for o in baixa_confianca)
            orientacao += f" {fotos_txt}: confiança baixa, recomenda-se refazer a fotografia."
        return "doenca", len(doentes_confiaveis), orientacao, doencas

    if baixa_confianca:
        fotos_txt = ", ".join(f"Foto {o}" for o in baixa_confianca)
        return (
            "incerto",
            0,
            f"Confiança abaixo do limiar ({limiar * 100:.0f}%) em: {fotos_txt}. "
            "Envie novas fotografias com uma folha por imagem, fundo liso e boa iluminação.",
            [],
        )

    return "saudavel", 0, "Todas as folhas foram classificadas como saudáveis com alta confiança.", []
