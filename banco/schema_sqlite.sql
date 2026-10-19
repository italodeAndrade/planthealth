-- =====================================================================
-- Modelo físico (SQLite) – monitoramento semanal da saúde de plantas
-- Uso:  sqlite3 agro_saude.db < schema_sqlite.sql
-- Observação: ative as chaves estrangeiras em TODA conexão:
--             PRAGMA foreign_keys = ON;
-- Datas em texto ISO 8601 (AAAA-MM-DD ou AAAA-MM-DD HH:MM:SS, em UTC).
-- =====================================================================
PRAGMA foreign_keys = ON;

-- Plantas atendidas pelo sistema (uma linha por planta) ---------------
CREATE TABLE IF NOT EXISTS planta (
    slug               TEXT PRIMARY KEY,                 -- 'batata', 'milho', ...
    nome               TEXT NOT NULL,
    tem_classificador  INTEGER NOT NULL DEFAULT 1 CHECK (tem_classificador IN (0, 1))
);

-- Versões dos classificadores treinados no Colab ----------------------
CREATE TABLE IF NOT EXISTS modelo (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    planta           TEXT NOT NULL REFERENCES planta(slug),
    versao           TEXT NOT NULL,                      -- ex.: 'v1'
    arquitetura      TEXT NOT NULL,                      -- ex.: 'MobileNetV2'
    treinado_em      TEXT NOT NULL,
    classes          TEXT NOT NULL,                      -- JSON: ["Potato___healthy", ...]
    limiar           REAL NOT NULL CHECK (limiar BETWEEN 0 AND 1),
    n_treino         INTEGER,
    n_validacao      INTEGER,
    n_teste          INTEGER,
    acuracia_teste   REAL CHECK (acuracia_teste BETWEEN 0 AND 1),
    f1_macro_teste   REAL CHECK (f1_macro_teste BETWEEN 0 AND 1),
    metricas         TEXT,                               -- JSON: métricas por classe e matriz de confusão
    caminho          TEXT NOT NULL,                      -- arquivo do modelo (.onnx, .pt, .keras)
    ativo            INTEGER NOT NULL DEFAULT 0 CHECK (ativo IN (0, 1)),
    UNIQUE (planta, versao)
);
-- no máximo um modelo ativo por planta
CREATE UNIQUE INDEX IF NOT EXISTS ux_modelo_ativo ON modelo(planta) WHERE ativo = 1;

-- Calendário de referência (fases esperadas por semana) ---------------
CREATE TABLE IF NOT EXISTS fase_cultura (
    planta           TEXT NOT NULL REFERENCES planta(slug),
    semana_inicio    INTEGER NOT NULL CHECK (semana_inicio >= 0),
    semana_fim       INTEGER NOT NULL,
    fase             TEXT NOT NULL,                      -- ex.: 'vegetativa'
    altura_ref_cm    REAL,
    n_folhas_ref     INTEGER,
    fonte            TEXT,                               -- referência agronômica usada
    PRIMARY KEY (planta, semana_inicio),
    CHECK (semana_fim >= semana_inicio)
);

-- Um plantio = um chat ------------------------------------------------
CREATE TABLE IF NOT EXISTS plantio (
    id               INTEGER PRIMARY KEY AUTOINCREMENT,
    planta           TEXT NOT NULL REFERENCES planta(slug),
    apelido          TEXT,                               -- ex.: 'Canteiro da varanda'
    data_plantio     TEXT NOT NULL
                     CHECK (data_plantio GLOB '[0-9][0-9][0-9][0-9]-[0-9][0-9]-[0-9][0-9]'),
    criado_em        TEXT NOT NULL DEFAULT (datetime('now')),
    ativo            INTEGER NOT NULL DEFAULT 1 CHECK (ativo IN (0, 1))
);
CREATE INDEX IF NOT EXISTS ix_plantio_planta ON plantio(planta, criado_em);

-- Envio semanal (3 fotos + resultado combinado + relatório) -----------
CREATE TABLE IF NOT EXISTS atualizacao (
    id                 INTEGER PRIMARY KEY AUTOINCREMENT,
    plantio_id         INTEGER NOT NULL REFERENCES plantio(id) ON DELETE CASCADE,
    enviado_em         TEXT NOT NULL DEFAULT (datetime('now')),
    semana             INTEGER NOT NULL CHECK (semana >= 0),   -- semanas desde o plantio
    fase_esperada      TEXT,
    status             TEXT NOT NULL CHECK (status IN ('saudavel', 'doenca', 'incerto')),
    folhas_com_doenca  INTEGER NOT NULL DEFAULT 0 CHECK (folhas_com_doenca BETWEEN 0 AND 3),
    altura_cm          REAL CHECK (altura_cm IS NULL OR altura_cm >= 0),
    n_folhas           INTEGER CHECK (n_folhas IS NULL OR n_folhas >= 0),
    modelo_id          INTEGER REFERENCES modelo(id),
    relatorio          TEXT,
    llm_usada          INTEGER NOT NULL DEFAULT 1 CHECK (llm_usada IN (0, 1))  -- 0 = texto de contingência
);
CREATE INDEX IF NOT EXISTS ix_atualizacao_plantio ON atualizacao(plantio_id, enviado_em);

-- As 3 fotos de cada atualização --------------------------------------
CREATE TABLE IF NOT EXISTS foto (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    atualizacao_id    INTEGER NOT NULL REFERENCES atualizacao(id) ON DELETE CASCADE,
    ordem             INTEGER NOT NULL CHECK (ordem BETWEEN 1 AND 3),
    caminho           TEXT NOT NULL,                     -- arquivo em disco (a imagem NÃO fica no banco)
    classe_prevista   TEXT NOT NULL,
    confianca         REAL NOT NULL CHECK (confianca BETWEEN 0 AND 1),
    probabilidades    TEXT NOT NULL,                     -- JSON: {"classe": prob, ...}
    UNIQUE (atualizacao_id, ordem)
);

-- Mensagens do chat do plantio ----------------------------------------
CREATE TABLE IF NOT EXISTS mensagem (
    id                INTEGER PRIMARY KEY AUTOINCREMENT,
    plantio_id        INTEGER NOT NULL REFERENCES plantio(id) ON DELETE CASCADE,
    atualizacao_id    INTEGER REFERENCES atualizacao(id) ON DELETE SET NULL,
    papel             TEXT NOT NULL CHECK (papel IN ('usuario', 'assistente', 'sistema')),
    texto             TEXT NOT NULL,
    criado_em         TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE INDEX IF NOT EXISTS ix_mensagem_plantio ON mensagem(plantio_id, criado_em);

-- Histórico pronto para o gráfico de evolução (real x referência) -----
DROP VIEW IF EXISTS v_historico;
CREATE VIEW v_historico AS
SELECT a.plantio_id,
       a.id              AS atualizacao_id,
       a.enviado_em,
       a.semana,
       a.status,
       a.folhas_com_doenca,
       a.altura_cm,
       a.n_folhas,
       fc.fase           AS fase_tabela,
       fc.altura_ref_cm,
       fc.n_folhas_ref
FROM atualizacao a
JOIN plantio p ON p.id = a.plantio_id
LEFT JOIN fase_cultura fc
       ON fc.planta = p.planta
      AND a.semana BETWEEN fc.semana_inicio AND fc.semana_fim;

-- Dados iniciais (seed padrão) ----------------------------------------
INSERT OR IGNORE INTO planta (slug, nome, tem_classificador) VALUES
    ('batata', 'Batata', 1),
    ('tomate', 'Tomate', 1),
    ('milho', 'Milho', 1);

INSERT OR IGNORE INTO fase_cultura (planta, semana_inicio, semana_fim, fase, altura_ref_cm, n_folhas_ref, fonte) VALUES
    ('batata', 0, 2, 'Emergência / Brotamento', 8.0, 4, 'Embrapa Hortaliças'),
    ('batata', 3, 6, 'Desenvolvimento Vegetativo', 25.0, 12, 'Embrapa Hortaliças'),
    ('batata', 7, 10, 'Iniciação / Enchimento de Tubérculos', 45.0, 20, 'Embrapa Hortaliças'),
    ('batata', 11, 14, 'Maturação / Senescência', 45.0, 18, 'Embrapa Hortaliças');
