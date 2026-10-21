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


def salvar_plantio_nosql(dados: Dict[str, Any]) -> Optional[str]:
    """
    Registra um novo plantio/sessão de chat no MongoDB (coleção 'plantios').
    Schema: planta (str), apelido (str), data_plantio (date), criado_em (date), ativo (bool), plantio_id_sqlite (int).
    """
    bd = obter_bd()
    if bd is None:
        return None

    try:
        dt_plantio = dados.get("data_plantio")
        if isinstance(dt_plantio, str):
            try:
                dt_plantio = datetime.fromisoformat(dt_plantio.replace("Z", "+00:00"))
            except Exception:
                dt_plantio = datetime.now(timezone.utc)
        elif not isinstance(dt_plantio, datetime):
            dt_plantio = datetime.now(timezone.utc)

        doc = {
            "planta": str(dados.get("planta", "batata")).lower(),
            "apelido": str(dados.get("apelido") or f"Plantio {dados.get('id', '')}").strip(),
            "data_plantio": dt_plantio,
            "criado_em": datetime.now(timezone.utc),
            "ativo": bool(dados.get("ativo", True)),
            "plantio_id_sqlite": int(dados.get("id")) if dados.get("id") else None,
        }
        res = bd.plantios.insert_one(doc)
        return str(res.inserted_id)
    except Exception as e:
        print(f"[AVISO MONGO] Erro ao salvar plantio no NoSQL: {e}")
        return None


def sincronizar_plantios_sqlite_mongo(plantios_sqlite: List[Dict[str, Any]]) -> None:
    """
    Garante que todos os plantios catalogados no SQLite possuam um registro correspondente
    na coleção 'plantios' do MongoDB, unificando a identidade das conversas.
    """
    bd = obter_bd()
    if bd is None:
        return

    try:
        for p in plantios_sqlite:
            pid = p.get("id")
            if not pid:
                continue
            existente = bd.plantios.find_one({"plantio_id_sqlite": pid})
            if not existente:
                salvar_plantio_nosql(p)
                print(f"[MONGO] Plantio SQLite #{pid} ({p.get('apelido')}) sincronizado com sucesso no NoSQL.")
    except Exception as e:
        print(f"[AVISO MONGO] Erro ao sincronizar plantios SQLite com MongoDB: {e}")


def salvar_atualizacao_nosql(dados: Dict[str, Any]) -> Optional[str]:
    """
    Registra a atualização no MongoDB conforme o schema validado em mongo_init.js.
    """
    bd = obter_bd()
    if bd is None:
        return None

    try:
        doc = {
            "plantio_id": dados.get("plantio_id"),
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
    fotos: Optional[List[Dict[str, Any]]] = None,
) -> Optional[str]:
    """
    Insere mensagem de chat na coleção 'mensagens' do MongoDB (Seção 3.2 do ABNT).
    Salva 'plantio_id_sqlite' para amarrar a conversa à entidade relacional
    e armazena as 'fotos' analisadas diretamente no documento NoSQL.
    """
    bd = obter_bd()
    if bd is None:
        return None

    try:
        pid = int(plantio_id) if str(plantio_id).isdigit() else plantio_id
        doc = {
            "plantio_id": pid,
            "plantio_id_sqlite": pid,
            "atualizacao_id": atualizacao_id,
            "papel": papel,
            "texto": texto,
            "fotos": fotos or [],
            "criado_em": datetime.now(timezone.utc),
        }
        res = bd.mensagens.insert_one(doc)
        return str(res.inserted_id)
    except Exception as e:
        print(f"[AVISO MONGO] Erro ao salvar mensagem no MongoDB: {e}")
        return None


def listar_mensagens_chat(plantio_id: Any) -> List[Dict[str, Any]]:
    """
    Recupera todo o histórico de mensagens de chat de um plantio diretamente do MongoDB.
    Garante enriquecimento das fotos para mensagens de diagnóstico.
    """
    bd = obter_bd()
    if bd is None:
        return []

    try:
        pid = int(plantio_id) if str(plantio_id).isdigit() else plantio_id
        cursor = bd.mensagens.find(
            {"$or": [{"plantio_id": pid}, {"plantio_id_sqlite": pid}]}
        ).sort("criado_em", 1)

        mensagens = []
        for doc in cursor:
            criado_em = doc.get("criado_em")
            if isinstance(criado_em, datetime):
                data_str = criado_em.strftime("%Y-%m-%d %H:%M:%S")
            else:
                data_str = str(criado_em or "")

            fotos = doc.get("fotos") or []
            # Se a mensagem for de uma atualização e não tiver fotos salvas no doc,
            # busca na coleção atualizacoes do MongoDB
            if not fotos and doc.get("atualizacao_id"):
                at_id = doc.get("atualizacao_id")
                at_doc = bd.atualizacoes.find_one({
                    "$or": [
                        {"_id": at_id if isinstance(at_id, ObjectId) else None},
                        {"semana": at_id if isinstance(at_id, int) else None},
                        {"plantio_id": pid, "semana": at_id if isinstance(at_id, int) else None}
                    ]
                })
                if at_doc and at_doc.get("fotos"):
                    fotos = at_doc.get("fotos")

            mensagens.append({
                "id": str(doc["_id"]),
                "plantio_id": pid,
                "atualizacao_id": doc.get("atualizacao_id"),
                "papel": doc.get("papel"),
                "texto": doc.get("texto"),
                "criado_em": data_str,
                "fotos": fotos,
            })
        return mensagens
    except Exception as e:
        print(f"[AVISO MONGO] Erro ao listar mensagens no MongoDB: {e}")
        return []


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
