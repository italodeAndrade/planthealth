// =====================================================================
// Modelo físico (MongoDB) – monitoramento semanal da saúde de plantas
// Uso:  mongosh "mongodb://localhost:27017" mongo_init.js
//       (ou a URI do Atlas, se o Colab também for gravar no banco)
// Convenções: nomes em snake_case, datas como Date (UTC), imagens em
// disco (o documento guarda só o caminho), ids de plantas como texto.
// =====================================================================
const bd = db.getSiblingDB("agro_saude");

function criar(nome, schema) {
  if (bd.getCollectionNames().includes(nome)) return;
  bd.createCollection(nome, {
    validator: { $jsonSchema: schema },
    validationLevel: "strict",
    validationAction: "error",
  });
}

const num = { bsonType: "number" };
const texto = { bsonType: "string" };
const nuloOuNum = { bsonType: ["number", "null"] };

// ---------------------------------------------------------------------
// plantas: referência (uma por planta atendida)
// { _id: "batata", nome: "Batata", tem_classificador: true }
// ---------------------------------------------------------------------
criar("plantas", {
  bsonType: "object",
  required: ["_id", "nome", "tem_classificador"],
  properties: { _id: texto, nome: texto, tem_classificador: { bsonType: "bool" } },
});

// ---------------------------------------------------------------------
// plantio: cada plantio é um chat
// ---------------------------------------------------------------------
criar("plantios", {
  bsonType: "object",
  required: ["planta", "data_plantio", "criado_em", "ativo"],
  properties: {
    planta: texto,                         // referencia plantas._id
    apelido: texto,
    data_plantio: { bsonType: "date" },
    criado_em: { bsonType: "date" },
    ativo: { bsonType: "bool" },
  },
});

// ---------------------------------------------------------------------
// atualizacoes: envio semanal; as 3 fotos ficam EMBUTIDAS no documento
// ---------------------------------------------------------------------
criar("atualizacoes", {
  bsonType: "object",
  required: ["plantio_id", "enviado_em", "semana", "status", "folhas_com_doenca", "fotos"],
  properties: {
    plantio_id: { bsonType: ["objectId", "number", "int", "string"] },
    plantio_id_sqlite: { bsonType: ["number", "int", "null"] },
    enviado_em: { bsonType: "date" },
    semana: { bsonType: "number", minimum: 0 },
    fase_esperada: { bsonType: ["string", "null"] },
    status: { enum: ["saudavel", "doenca", "incerto"] },
    folhas_com_doenca: { bsonType: "number", minimum: 0, maximum: 3 },
    medidas: {
      bsonType: "object",
      properties: { altura_cm: nuloOuNum, n_folhas: nuloOuNum },
    },
    modelo_id: { bsonType: ["objectId", "number", "int", "string", "null"] },
    fotos: {
      bsonType: "array",
      minItems: 0,
      maxItems: 10,
      items: {
        bsonType: "object",
        required: ["ordem", "caminho", "classe_prevista", "confianca"],
        properties: {
          ordem: { bsonType: "number", minimum: 1, maximum: 3 },
          caminho: texto,
          classe_prevista: texto,
          confianca: { bsonType: "number", minimum: 0, maximum: 1 },
          probabilidades: { bsonType: "object" },   // { "classe": prob }
        },
      },
    },
    relatorio: texto,
    llm_usada: { bsonType: "bool" },        // false = texto de contingência
  },
});

// ---------------------------------------------------------------------
// mensagens: chat. Coleção separada porque cresce sem limite
// (um documento tem teto de 16 MB) e é paginada por data.
// ---------------------------------------------------------------------
criar("mensagens", {
  bsonType: "object",
  required: ["plantio_id", "papel", "texto", "criado_em"],
  properties: {
    plantio_id: { bsonType: ["objectId", "number", "int", "string"] },
    plantio_id_sqlite: { bsonType: ["number", "int", "null"] },
    atualizacao_id: { bsonType: ["objectId", "number", "int", "string", "null"] },
    papel: { enum: ["usuario", "assistente", "sistema"] },
    texto: texto,
    criado_em: { bsonType: "date" },
  },
});

// ---------------------------------------------------------------------
// modelos: uma linha por versão treinada (o Colab pode gravar aqui)
// O arquivo do modelo NÃO vai para o banco: só o caminho/versão.
// ---------------------------------------------------------------------
criar("modelos", {
  bsonType: "object",
  required: ["planta", "versao", "arquitetura", "treinado_em", "classes", "limiar", "caminho", "ativo"],
  properties: {
    planta: texto,
    versao: texto,
    arquitetura: texto,
    treinado_em: { bsonType: "date" },
    ambiente: { bsonType: "object" },       // ex.: { plataforma: "colab", gpu: "T4" }
    hiperparametros: { bsonType: "object" },
    divisao: { bsonType: "object" },        // { treino, validacao, teste, semente }
    classes: { bsonType: "array", items: texto },
    limiar: { bsonType: "number", minimum: 0, maximum: 1 },
    metricas: { bsonType: "object" },       // acuracia, f1_macro, por_classe
    matriz_confusao: { bsonType: "object" },// { classes: [...], valores: [[...]] }
    caminho: texto,
    ativo: { bsonType: "bool" },
  },
});

// ---------------------------------------------------------------------
// colab_logs_modelos: artefatos densos vindos do Google Colab (Seção 3.2 ABNT)
// Hiperparâmetros, logs de épocas, métricas por classe e matriz de confusão.
// ---------------------------------------------------------------------
criar("colab_logs_modelos", {
  bsonType: "object",
  required: ["planta", "versao", "arquitetura", "treinado_em"],
  properties: {
    planta: texto,
    versao: texto,
    arquitetura: texto,
    treinado_em: { bsonType: "date" },
    ambiente: { bsonType: "object" },
    hiperparametros: { bsonType: "object" },
    divisao: { bsonType: "object" },
    classes: { bsonType: "array", items: texto },
    limiar: { bsonType: "number" },
    metricas: { bsonType: "object" },
    matriz_confusao: { bsonType: "object" },
    logs_epocas: { bsonType: "array" },
    caminho: texto,
    ativo: { bsonType: "bool" },
  },
});

// ---------------------------------------------------------------------
// fases_cultura: calendário de referência por semana
// ---------------------------------------------------------------------
criar("fases_cultura", {
  bsonType: "object",
  required: ["planta", "semana_inicio", "semana_fim", "fase"],
  properties: {
    planta: texto,
    semana_inicio: { bsonType: "number", minimum: 0 },
    semana_fim: num,
    fase: texto,
    altura_ref_cm: nuloOuNum,
    n_folhas_ref: nuloOuNum,
    fonte: texto,
  },
});


// ---------------------------------------------------------------------
// Índices (as consultas do app: histórico do plantio, chat paginado,
// modelo ativo da planta e fase esperada por semana)
// ---------------------------------------------------------------------
bd.plantios.createIndex({ planta: 1, criado_em: -1 });
bd.atualizacoes.createIndex({ plantio_id: 1, enviado_em: -1 });
bd.mensagens.createIndex({ plantio_id: 1, criado_em: 1 });
bd.modelos.createIndex({ planta: 1, versao: 1 }, { unique: true });
bd.modelos.createIndex(
  { planta: 1 },
  { unique: true, partialFilterExpression: { ativo: true } }   // um modelo ativo por planta
);
bd.fases_cultura.createIndex({ planta: 1, semana_inicio: 1 }, { unique: true });

// Carga inicial básica
if (bd.plantas.countDocuments() === 0) {
  bd.plantas.insertMany([
    { _id: "batata", nome: "Batata", tem_classificador: true },
    { _id: "tomate", nome: "Tomate", tem_classificador: true },
    { _id: "milho", nome: "Milho", tem_classificador: true }
  ]);
}

print("agro_saude: coleções e índices criados.");
