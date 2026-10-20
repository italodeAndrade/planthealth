import json
import os
import shutil
from contextlib import asynccontextmanager
from typing import Any, Dict, List, Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

import ollama
from app.bd_nosql import (
    DB_NAME,
    obter_cliente,
    salvar_atualizacao_nosql,
    salvar_log_treino_colab,
    salvar_mensagem_chat,
)
from app.bd_relacional import (
    criar_plantio,
    inicializar_banco,
    inserir_atualizacao,
    inserir_mensagem,
    listar_mensagens,
    listar_plantas,
    listar_plantios,
    obter_conexao,
    obter_fase_cultura,
    obter_historico,
    obter_plantio,
)
from app.inferencia import classificar_foto, combinar_diagnosticos
from app.llm import (
    OLLAMA_HOST,
    OLLAMA_MODEL,
    chat_generico_ollama,
    gerar_relatorio_agronomico,
    listar_modelos_ollama,
    obter_cliente_ollama,
    responder_chat_agronomico,
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
STATIC_DIR = os.path.join(BASE_DIR, "app", "static")
UPLOADS_DIR = os.path.join(BASE_DIR, "dados", "uploads")
os.makedirs(UPLOADS_DIR, exist_ok=True)


@asynccontextmanager
async def lifespan(app: FastAPI):
    # Inicializa tabelas do SQLite na subida da aplicação
    inicializar_banco()
    yield


app = FastAPI(
    title="AgroSaúde API",
    description="Sistema Inteligente de Monitoramento da Saúde de Plantas por Visão Computacional e Modelo de Linguagem (PjBL2/TDE2)",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Monta diretório estático e uploads
if os.path.exists(STATIC_DIR):
    app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

if os.path.exists(UPLOADS_DIR):
    app.mount("/uploads", StaticFiles(directory=UPLOADS_DIR), name="uploads")


# Schemas Pydantic
class NovoPlantio(BaseModel):
    planta: str = Field(..., example="batata")
    apelido: Optional[str] = Field(None, example="Canteiro da horta")
    data_plantio: str = Field(..., pattern=r"^\d{4}-\d{2}-\d{2}$", example="2026-10-01")


class AtualizacaoManual(BaseModel):
    planta: str = Field(..., example="batata")
    status: str = Field(..., example="doenca")
    folhas_doentes: int = Field(..., ge=0, le=3, example=2)
    semana: Optional[int] = Field(3, example=3)


class MensagemChat(BaseModel):
    plantio_id: int
    texto: str


class LogTreinoColab(BaseModel):
    planta: str = Field("batata", example="batata")
    versao: str = Field("v1", example="v1")
    arquitetura: str = Field("MobileNetV2", example="MobileNetV2")
    classes: List[str] = Field(..., example=["Potato___Early_blight", "Potato___Late_blight", "Potato___healthy"])
    limiar: float = Field(0.70, example=0.70)
    acuracia_teste: float = Field(..., example=0.945)
    f1_macro_teste: float = Field(..., example=0.941)
    metricas: Dict[str, Any] = Field(..., example={"precisao_macro": 0.943, "revocacao_macro": 0.940})
    matriz_confusao: Dict[str, Any] = Field(..., example={"classes": ["Early", "Late", "Healthy"], "valores": [[95, 3, 2], [2, 94, 4], [1, 2, 97]]})
    hiperparametros: Optional[Dict[str, Any]] = Field(default_factory=dict)
    caminho_pesos: Optional[str] = Field("modelos/batata.onnx", example="modelos/batata.onnx")


class OllamaChatRequest(BaseModel):
    prompt: Optional[str] = Field(None, example="Como prevenir requeima da batata sem produtos químicos?")
    mensagens: Optional[List[Dict[str, str]]] = Field(None, example=[{"role": "user", "content": "Olá assistente"}])
    modelo: Optional[str] = Field(None, example="llama3.2")


class OllamaPromptRequest(BaseModel):
    prompt: str = Field(..., example="Explique como fazer rotação de culturas na horta familiar.")
    modelo: Optional[str] = Field(None, example="llama3.2")
    system: Optional[str] = Field(None, example="Você é um agrônomo especialista do sistema AgroSaúde.")


# Rotas
@app.get("/")
async def raiz():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "ok", "mensagem": "API AgroSaúde em operação"}


@app.get("/api/health")
async def health_check():
    # 1. SQLite
    sqlite_ok = False
    plantas_total = 0
    try:
        plantas = listar_plantas()
        plantas_total = len(plantas)
        sqlite_ok = True
    except Exception as e:
        sqlite_erro = str(e)

    # 2. MongoDB
    mongo_ok = False
    mongo_colecoes = []
    mongo_erro = None
    try:
        cliente_mongo = obter_cliente()
        if cliente_mongo is not None:
            cliente_mongo.admin.command("ping")
            bd = cliente_mongo[DB_NAME]
            mongo_colecoes = bd.list_collection_names()
            mongo_ok = True
        else:
            mongo_erro = "Cliente MongoDB retornou None"
    except Exception as e:
        mongo_erro = str(e)

    # 3. Ollama (GPU local)
    ollama_ok = False
    modelos_ollama = []
    ollama_erro = None
    try:
        client = ollama.Client(host=OLLAMA_HOST) if OLLAMA_HOST else ollama
        resp = client.list()
        if hasattr(resp, "models"):
            modelos_ollama = [m.model for m in resp.models]
        elif isinstance(resp, dict) and "models" in resp:
            modelos_ollama = [m.get("name", m.get("model")) for m in resp["models"]]
        ollama_ok = True
    except Exception as e:
        ollama_erro = str(e)

    return {
        "status": "operacional" if (sqlite_ok and mongo_ok and ollama_ok) else "degradado",
        "sqlite": {
            "conectado": sqlite_ok,
            "plantas_cadastradas": plantas_total,
        },
        "mongodb": {
            "conectado": mongo_ok,
            "colecoes": mongo_colecoes,
            "erro": mongo_erro,
        },
        "ollama": {
            "conectado": ollama_ok,
            "host": OLLAMA_HOST or "default",
            "modelo_padrao": OLLAMA_MODEL,
            "modelos_disponiveis": modelos_ollama,
            "erro": ollama_erro,
        },
    }


@app.get("/api/plantas")
async def get_plantas():
    return listar_plantas()


@app.get("/api/plantios")
async def get_plantios():
    return listar_plantios()


@app.post("/api/plantios")
async def post_plantio(dados: NovoPlantio):
    plantio_id = criar_plantio(
        planta=dados.planta,
        apelido=dados.apelido,
        data_plantio=dados.data_plantio,
    )
    return {"sucesso": True, "plantio_id": plantio_id}


@app.get("/api/plantios/{plantio_id}")
async def get_detalhes_plantio(plantio_id: int):
    plantio = obter_plantio(plantio_id)
    if not plantio:
        raise HTTPException(status_code=404, detail="Plantio não encontrado")

    historico = obter_historico(plantio_id)
    mensagens = listar_mensagens(plantio_id)
    return {
        "plantio": plantio,
        "historico": historico,
        "mensagens": mensagens,
    }


@app.post("/api/atualizacoes/upload")
async def enviar_atualizacao_semanal(
    plantio_id: int = Form(...),
    semana: int = Form(...),
    altura_cm: Optional[float] = Form(None),
    n_folhas: Optional[int] = Form(None),
    limiar: float = Form(0.70),
    fotos: List[UploadFile] = File(...),
):
    """
    Recebe 3 fotos de folhas, roda inferência em cada uma,
    aplica a regra de combinação e limiar da Seção 4.4, consulta o calendário de fases da cultura,
    chama a LLM ética (Ollama) e persiste de forma poliglota em SQLite e MongoDB.
    """
    if len(fotos) != 3:
        raise HTTPException(status_code=400, detail="É obrigatório o envio de exatamente 3 fotos distintas de folhas.")

    plantio = obter_plantio(plantio_id)
    if not plantio:
        raise HTTPException(status_code=404, detail="Plantio não encontrado.")

    fotos_processadas = []

    for idx, upload_file in enumerate(fotos, start=1):
        ext = os.path.splitext(upload_file.filename or "")[1] or ".jpg"
        nome_arquivo = f"plantio_{plantio_id}_sem{semana}_f{idx}_{os.urandom(4).hex()}{ext}"
        caminho_disco = os.path.join(UPLOADS_DIR, nome_arquivo)

        with open(caminho_disco, "wb") as buffer:
            shutil.copyfileobj(upload_file.file, buffer)

        # Classificação da imagem
        resultado_foto = classificar_foto(caminho_disco, ordem=idx)
        resultado_foto["caminho"] = f"/uploads/{nome_arquivo}"
        fotos_processadas.append(resultado_foto)

    # 1. Regra de decisão das 3 folhas com limiar de confiança (Seção 4.4 do ABNT)
    status, folhas_doentes, orientacao_adicional = combinar_diagnosticos(fotos_processadas, limiar=limiar)

    # 2. Consulta da fase fenológica de referência agronômica (Seção 5 do ABNT)
    ref_fase = obter_fase_cultura(plantio["planta"], semana)
    fase_esperada = ref_fase["fase"] if ref_fase else None

    # 3. Geração do relatório com Ollama respeitando restrições éticas da Seção 7
    relatorio = gerar_relatorio_agronomico(
        planta=plantio["planta_nome"],
        status=status,
        folhas_doentes=folhas_doentes,
        semana=semana,
        fase_esperada=fase_esperada,
        orientacao_adicional=orientacao_adicional,
    )

    # 4. Persistência Relacional (SQLite - Core Estrutural)
    atualizacao_id = inserir_atualizacao(
        plantio_id=plantio_id,
        semana=semana,
        fase_esperada=fase_esperada,
        status=status,
        folhas_com_doenca=folhas_doentes,
        fotos=fotos_processadas,
        relatorio=relatorio,
        llm_usada=True,
        altura_cm=altura_cm,
        n_folhas=n_folhas,
    )

    # Inserção da mensagem no histórico de chat
    inserir_mensagem(
        plantio_id=plantio_id,
        papel="assistente",
        texto=f"🌱 **Atualização da Semana {semana} ({status.upper()}):**\n\n{relatorio}",
        atualizacao_id=atualizacao_id,
    )

    # 5. Persistência NoSQL (MongoDB - Coleção atualizacoes e mensagens)
    salvar_atualizacao_nosql({
        "plantio_id": plantio_id,
        "semana": semana,
        "fase_esperada": fase_esperada,
        "status": status,
        "folhas_com_doenca": folhas_doentes,
        "altura_cm": altura_cm,
        "n_folhas": n_folhas,
        "fotos": fotos_processadas,
        "relatorio": relatorio,
        "llm_usada": True,
    })
    salvar_mensagem_chat(
        plantio_id=plantio_id,
        papel="assistente",
        texto=relatorio,
        atualizacao_id=atualizacao_id,
    )

    return {
        "sucesso": True,
        "atualizacao_id": atualizacao_id,
        "status": status,
        "folhas_com_doenca": folhas_doentes,
        "fase_esperada": fase_esperada,
        "orientacao_adicional": orientacao_adicional,
        "fotos": fotos_processadas,
        "relatorio": relatorio,
    }


@app.post("/api/gerar-relatorio")
async def api_gerar_relatorio(dados: AtualizacaoManual):
    """Endpoint direto para teste pontual da LLM (TDE2 / PjBL2)."""
    ref_fase = obter_fase_cultura(dados.planta.lower(), dados.semana or 3)
    fase_esperada = ref_fase["fase"] if ref_fase else None

    relatorio = gerar_relatorio_agronomico(
        planta=dados.planta,
        status=dados.status,
        folhas_doentes=dados.folhas_doentes,
        semana=dados.semana,
        fase_esperada=fase_esperada,
    )
    return {
        "sucesso": True,
        "fase_esperada": fase_esperada,
        "relatorio_ia": relatorio,
    }


@app.get("/api/ollama/status")
async def get_ollama_status():
    """Verifica conectividade direta com o Ollama na GPU."""
    try:
        modelos = listar_modelos_ollama()
        return {
            "status": "online",
            "host": OLLAMA_HOST or "http://127.0.0.1:11434",
            "modelo_padrao": OLLAMA_MODEL,
            "modelos_disponiveis": modelos,
            "total_modelos": len(modelos),
        }
    except Exception as e:
        raise HTTPException(status_code=503, detail=f"Ollama inacessível: {e}")


@app.get("/api/ollama/modelos")
async def get_ollama_modelos():
    """Lista todos os modelos instalados no Ollama."""
    modelos = listar_modelos_ollama()
    return {"modelos": modelos}


@app.post("/api/ollama/chat")
async def post_ollama_chat(req: OllamaChatRequest):
    """Chat direto com a LLM do Ollama (passando prompt ou array de mensagens)."""
    mensagens = req.mensagens or []
    if req.prompt:
        mensagens.append({"role": "user", "content": req.prompt})

    if not mensagens:
        raise HTTPException(status_code=400, detail="Envie 'prompt' ou 'mensagens'.")

    try:
        resposta = chat_generico_ollama(mensagens=mensagens, modelo=req.modelo)
        return {
            "sucesso": True,
            "modelo_usado": req.modelo or OLLAMA_MODEL,
            "resposta": resposta,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao comunicar com o Ollama: {e}")


@app.post("/api/ollama/gerar")
async def post_ollama_gerar(req: OllamaPromptRequest):
    """Gera texto a partir de prompt com system opcional."""
    mensagens = []
    if req.system:
        mensagens.append({"role": "system", "content": req.system})
    mensagens.append({"role": "user", "content": req.prompt})

    try:
        resposta = chat_generico_ollama(mensagens=mensagens, modelo=req.modelo)
        return {
            "sucesso": True,
            "modelo_usado": req.modelo or OLLAMA_MODEL,
            "resposta": resposta,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Erro ao gerar com Ollama: {e}")


@app.post("/api/chat")
async def enviar_mensagem(dados: MensagemChat):
    """
    Chat com o assistente agronômico inteligente sobre o plantio:
    - Salva pergunta no SQLite e MongoDB.
    - Consulta histórico de mensagens do plantio.
    - Chama o Ollama (Llama 3.2 na GPU) com regras agronômicas éticas.
    - Salva resposta gerada no SQLite e MongoDB.
    """
    plantio = obter_plantio(dados.plantio_id)
    if not plantio:
        raise HTTPException(status_code=404, detail="Plantio não encontrado")

    # 1. Salva pergunta do usuário
    inserir_mensagem(dados.plantio_id, papel="usuario", texto=dados.texto)
    salvar_mensagem_chat(dados.plantio_id, papel="usuario", texto=dados.texto)

    # 2. Obtém histórico recente do plantio
    historico = listar_mensagens(dados.plantio_id)

    # 3. Gera resposta inteligente com Ollama
    resposta = responder_chat_agronomico(
        planta_nome=plantio.get("planta_nome") or plantio.get("planta", "cultura"),
        mensagem_usuario=dados.texto,
        historico_mensagens=historico,
    )

    # 4. Salva resposta no SQLite e MongoDB
    inserir_mensagem(dados.plantio_id, papel="assistente", texto=resposta)
    salvar_mensagem_chat(dados.plantio_id, papel="assistente", texto=resposta)

    return {"sucesso": True, "resposta": resposta}


@app.post("/api/colab/registrar-modelo")
async def registrar_modelo_colab(dados: LogTreinoColab):
    """
    Endpoint para o Google Colab registrar os artefatos de treino (Seção 3.2 do ABNT):
    - Salva logs densos e matriz de confusão no MongoDB (coleção colab_logs_modelos)
    - Registra a versão e caminho do arquivo .onnx no SQLite (tabela modelo)
    """
    # 1. MongoDB (Artefatos Densos)
    mongo_id = salvar_log_treino_colab(dados.model_dump())

    # 2. SQLite (Metadados Estruturais)
    with obter_conexao() as conn:
        conn.execute(
            """
            INSERT INTO modelo (
                planta, versao, arquitetura, treinado_em, classes, limiar,
                acuracia_teste, f1_macro_teste, metricas, caminho, ativo
            )
            VALUES (?, ?, ?, datetime('now'), ?, ?, ?, ?, ?, ?, 1)
            ON CONFLICT(planta, versao) DO UPDATE SET
                acuracia_teste = excluded.acuracia_teste,
                f1_macro_teste = excluded.f1_macro_teste,
                metricas = excluded.metricas,
                caminho = excluded.caminho,
                ativo = 1
            """,
            (
                dados.planta,
                dados.versao,
                dados.arquitetura,
                json.dumps(dados.classes),
                dados.limiar,
                dados.acuracia_teste,
                dados.f1_macro_teste,
                json.dumps(dados.metricas),
                dados.caminho_pesos or "modelos/batata.onnx",
            ),
        )
        conn.commit()

    return {
        "sucesso": True,
        "mensagem": f"Modelo {dados.planta} {dados.versao} registrado com sucesso!",
        "mongo_log_id": mongo_id,
    }
