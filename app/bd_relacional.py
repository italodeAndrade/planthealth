import json
import os
import sqlite3
from datetime import datetime
from typing import Any, Dict, List, Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DB_PATH = os.path.join(BASE_DIR, "banco", "agro_saude.db")
SCHEMA_PATH = os.path.join(BASE_DIR, "banco", "schema_sqlite.sql")


def obter_conexao() -> sqlite3.Connection:
    """Retorna uma conexão com o SQLite garantindo row_factory e foreign_keys=ON."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA foreign_keys = ON;")
    return conn


def inicializar_banco() -> None:
    """Aplica o schema_sqlite.sql caso o banco ainda não tenha sido populado."""
    if os.path.exists(SCHEMA_PATH):
        with obter_conexao() as conn:
            with open(SCHEMA_PATH, "r", encoding="utf-8") as f:
                script = f.read()
            conn.executescript(script)


def listar_plantas() -> List[Dict[str, Any]]:
    """Retorna todas as plantas cadastradas."""
    with obter_conexao() as conn:
        cursor = conn.execute("SELECT slug, nome, tem_classificador FROM planta ORDER BY nome ASC")
        return [dict(row) for row in cursor.fetchall()]


def obter_fase_cultura(planta: str, semana: int) -> Optional[Dict[str, Any]]:
    """Consulta a fase esperada, altura e folhas de referência para a cultura e semana."""
    with obter_conexao() as conn:
        cursor = conn.execute(
            """
            SELECT fase, altura_ref_cm, n_folhas_ref, fonte
            FROM fase_cultura
            WHERE planta = ? AND ? BETWEEN semana_inicio AND semana_fim
            """,
            (planta, semana),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def criar_plantio(planta: str, apelido: Optional[str], data_plantio: str) -> int:
    """Cria um novo plantio e retorna o ID gerado."""
    with obter_conexao() as conn:
        cursor = conn.execute(
            """
            INSERT INTO plantio (planta, apelido, data_plantio, criado_em, ativo)
            VALUES (?, ?, ?, datetime('now'), 1)
            """,
            (planta, apelido, data_plantio),
        )
        conn.commit()
        return cursor.lastrowid


def listar_plantios() -> List[Dict[str, Any]]:
    """Retorna os plantios ativos ordenados por data de criação desc."""
    with obter_conexao() as conn:
        cursor = conn.execute(
            """
            SELECT p.id, p.planta, pl.nome AS planta_nome, p.apelido, p.data_plantio, p.criado_em, p.ativo
            FROM plantio p
            JOIN planta pl ON pl.slug = p.planta
            ORDER BY p.criado_em DESC
            """
        )
        return [dict(row) for row in cursor.fetchall()]


def obter_plantio(plantio_id: int) -> Optional[Dict[str, Any]]:
    """Busca os detalhes de um plantio por ID."""
    with obter_conexao() as conn:
        cursor = conn.execute(
            """
            SELECT p.id, p.planta, pl.nome AS planta_nome, p.apelido, p.data_plantio, p.criado_em, p.ativo
            FROM plantio p
            JOIN planta pl ON pl.slug = p.planta
            WHERE p.id = ?
            """,
            (plantio_id,),
        )
        row = cursor.fetchone()
        return dict(row) if row else None


def inserir_atualizacao(
    plantio_id: int,
    semana: int,
    status: str,
    folhas_com_doenca: int,
    fotos: List[Dict[str, Any]],
    relatorio: str,
    llm_usada: bool = True,
    fase_esperada: Optional[str] = None,
    altura_cm: Optional[float] = None,
    n_folhas: Optional[int] = None,
    modelo_id: Optional[int] = None,
) -> int:
    """
    Insere uma atualização semanal com as 3 fotos associadas na mesma transação.
    """
    with obter_conexao() as conn:
        cursor = conn.cursor()
        cursor.execute(
            """
            INSERT INTO atualizacao (
                plantio_id, enviado_em, semana, fase_esperada, status,
                folhas_com_doenca, altura_cm, n_folhas, modelo_id, relatorio, llm_usada
            )
            VALUES (?, datetime('now'), ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                plantio_id,
                semana,
                fase_esperada,
                status,
                folhas_com_doenca,
                altura_cm,
                n_folhas,
                modelo_id,
                relatorio,
                1 if llm_usada else 0,
            ),
        )
        atualizacao_id = cursor.lastrowid

        for foto in fotos:
            cursor.execute(
                """
                INSERT INTO foto (atualizacao_id, ordem, caminho, classe_prevista, confianca, probabilidades)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    atualizacao_id,
                    foto["ordem"],
                    foto["caminho"],
                    foto["classe_prevista"],
                    foto["confianca"],
                    json.dumps(foto.get("probabilidades", {})),
                ),
            )

        conn.commit()
        return atualizacao_id


def obter_historico(plantio_id: int) -> List[Dict[str, Any]]:
    """Consulta a view v_historico para dados semanais."""
    with obter_conexao() as conn:
        cursor = conn.execute(
            """
            SELECT * FROM v_historico
            WHERE plantio_id = ?
            ORDER BY semana ASC
            """,
            (plantio_id,),
        )
        return [dict(row) for row in cursor.fetchall()]


def inserir_mensagem(
    plantio_id: int,
    papel: str,
    texto: str,
    atualizacao_id: Optional[int] = None,
) -> int:
    """[LEGADO] As mensagens de chat agora são armazenadas exclusivamente no MongoDB (app.bd_nosql)."""
    with obter_conexao() as conn:
        cursor = conn.execute(
            """
            INSERT INTO mensagem (plantio_id, atualizacao_id, papel, texto, criado_em)
            VALUES (?, ?, ?, ?, datetime('now'))
            """,
            (plantio_id, atualizacao_id, papel, texto),
        )
        conn.commit()
        return cursor.lastrowid


def listar_mensagens(plantio_id: int) -> List[Dict[str, Any]]:
    """[LEGADO] As mensagens de chat agora são recuperadas exclusivamente do MongoDB (app.bd_nosql)."""
    with obter_conexao() as conn:
        cursor = conn.execute(
            """
            SELECT id, plantio_id, atualizacao_id, papel, texto, criado_em
            FROM mensagem
            WHERE plantio_id = ?
            ORDER BY criado_em ASC
            """,
            (plantio_id,),
        )
        mensagens = [dict(row) for row in cursor.fetchall()]

        # Carrega as fotos vinculadas às atualizações
        for m in mensagens:
            if m.get("atualizacao_id"):
                c_fotos = conn.execute(
                    """
                    SELECT ordem, caminho, classe_prevista, confianca
                    FROM foto
                    WHERE atualizacao_id = ?
                    ORDER BY ordem ASC
                    """,
                    (m["atualizacao_id"],),
                )
                m["fotos"] = [dict(r) for r in c_fotos.fetchall()]
            else:
                m["fotos"] = []
        return mensagens


def obter_ultimas_fotos_plantio(plantio_id: int, limite_atualizacoes: int = 3) -> List[Dict[str, Any]]:
    """Retorna as fotos e diagnósticos dos envios mais recentes do plantio para contexto do Ollama."""
    with obter_conexao() as conn:
        cursor = conn.execute(
            """
            SELECT a.id AS atualizacao_id, a.semana, a.status, a.folhas_com_doenca, a.enviado_em,
                   f.ordem, f.caminho, f.classe_prevista, f.confianca
            FROM atualizacao a
            JOIN foto f ON f.atualizacao_id = a.id
            WHERE a.plantio_id = ?
            ORDER BY a.enviado_em DESC, f.ordem ASC
            """,
            (plantio_id,),
        )
        return [dict(row) for row in cursor.fetchall()]

