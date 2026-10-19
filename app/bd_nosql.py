import os
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from bson import ObjectId

try:
    from pymongo import MongoClient
    from pymongo.errors import PyMongoError
except ImportError:
    MongoClient = None
    PyMongoError = Exception

MONGO_URI = os.getenv("MONGO_URI", "mongodb://localhost:27017")
DB_NAME = os.getenv("MONGO_DB", "agro_saude")


def obter_cliente():
    """Retorna cliente do MongoDB ou None se não configurado/instalado."""
    if MongoClient is None:
        return None
    try:
        client = MongoClient(MONGO_URI, serverSelectionTimeoutMS=2000)
        return client
    except Exception as e:
        print(f"[AVISO MONGO] Não foi possível criar cliente MongoDB: {e}")
        return None


def obter_bd():
    """Retorna o banco de dados agro_saude."""
    cliente = obter_cliente()
    if cliente is not None:
        return cliente[DB_NAME]
    return None


def salvar_atualizacao_nosql(dados: Dict[str, Any]) -> Optional[str]:
    """
    Registra a atualização no MongoDB conforme o schema validado em mongo_init.js.
    """
    bd = obter_bd()
    if bd is None:
        return None

    try:
        doc = {
            "plantio_id_sqlite": dados.get("plantio_id"),
            "enviado_em": datetime.now(timezone.utc),
            "semana": int(dados.get("semana", 0)),
            "fase_esperada": dados.get("fase_esperada"),
            "status": dados.get("status", "incerto"),
            "folhas_com_doenca": int(dados.get("folhas_com_doenca", 0)),
            "medidas": {
                "altura_cm": dados.get("altura_cm"),
                "n_folhas": dados.get("n_folhas"),
            },
            "modelo_id": None,
            "fotos": dados.get("fotos", []),
            "relatorio": dados.get("relatorio", ""),
            "llm_usada": bool(dados.get("llm_usada", True)),
        }
        res = bd.atualizacoes.insert_one(doc)
        return str(res.inserted_id)
    except Exception as e:
        print(f"[AVISO MONGO] Erro ao salvar atualização no NoSQL: {e}")
        return None


def salvar_mensagem_chat(
    plantio_id: Any,
    papel: str,
    texto: str,
    atualizacao_id: Optional[Any] = None,
) -> Optional[str]:
    """
    Insere mensagem de chat na coleção 'mensagens' do MongoDB (Seção 3.2 do ABNT).
    Salva 'plantio_id_sqlite' para amarrar a conversa à entidade relacional.
    """
    bd = obter_bd()
    if bd is None:
        return None

    try:
        doc = {
            "plantio_id_sqlite": int(plantio_id) if str(plantio_id).isdigit() else plantio_id,
            "atualizacao_id": atualizacao_id,
            "papel": papel,
            "texto": texto,
            "criado_em": datetime.now(timezone.utc),
        }
        res = bd.mensagens.insert_one(doc)
        return str(res.inserted_id)
    except Exception as e:
        print(f"[AVISO MONGO] Erro ao salvar mensagem no MongoDB: {e}")
        return None


def salvar_log_treino_colab(dados_treino: Dict[str, Any]) -> Optional[str]:
    """
    Salva artefatos densos na coleção 'colab_logs_modelos' (Seção 3.2 do ABNT):
    - hiperparâmetros
    - logs de épocas
    - métricas por classe (precisão, revocação, f1-score, acurácia)
    - matriz de confusão completa
    """
    bd = obter_bd()
    if bd is None:
        return None

    try:
        doc = {
            "planta": dados_treino.get("planta", "batata"),
            "versao": dados_treino.get("versao", "v1"),
            "arquitetura": dados_treino.get("arquitetura", "MobileNetV2"),
            "treinado_em": datetime.now(timezone.utc),
            "ambiente": dados_treino.get("ambiente", {"plataforma": "colab", "gpu": "T4"}),
            "hiperparametros": dados_treino.get("hiperparametros", {}),
            "divisao": dados_treino.get("divisao", {"treino": 0.70, "validacao": 0.15, "teste": 0.15, "semente": 42}),
            "classes": dados_treino.get("classes", []),
            "limiar": float(dados_treino.get("limiar", 0.70)),
            "metricas": dados_treino.get("metricas", {}),
            "matriz_confusao": dados_treino.get("matriz_confusao", {}),
            "logs_epocas": dados_treino.get("logs_epocas", []),
            "caminho": dados_treino.get("caminho", "modelos/batata.onnx"),
            "ativo": bool(dados_treino.get("ativo", True)),
        }
        res = bd.colab_logs_modelos.insert_one(doc)
        print(f"[MONGO] Log de treino do Colab registrado com sucesso: {res.inserted_id}")
        return str(res.inserted_id)
    except Exception as e:
        print(f"[AVISO MONGO] Erro ao salvar logs de treino do Colab: {e}")
        return None
