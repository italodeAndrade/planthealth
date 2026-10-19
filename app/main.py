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

from app.bd_nosql import salvar_atualizacao_nosql, salvar_log_treino_colab, salvar_mensagem_chat
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
from app.llm import gerar_relatorio_agronomico

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


# Rotas
@app.get("/")
async def raiz():
    index_file = os.path.join(STATIC_DIR, "index.html")
    if os.path.exists(index_file):
        return FileResponse(index_file)
    return {"status": "ok", "mensagem": "API AgroSaúde em operação"}


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


@app.post("/api/chat")
async def enviar_mensagem(dados: MensagemChat):
    """Chat com o assistente sobre o plantio."""
    plantio = obter_plantio(dados.plantio_id)
    if not plantio:
        raise HTTPException(status_code=404, detail="Plantio não encontrado")

    inserir_mensagem(dados.plantio_id, papel="usuario", texto=dados.texto)
    salvar_mensagem_chat(dados.plantio_id, papel="usuario", texto=dados.texto)

    resposta = (
        f"Entendido sobre o seu plantio de {plantio['planta_nome']}. "
        "Mantenha o monitoramento semanal com fotos das folhas e siga boas práticas culturais. "
        "Não aplique defensivos químicos sem a receita e o diagnóstico presencial de um agrônomo."
    )
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
