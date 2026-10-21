# Modelos de Visão Computacional – AgroSaúde

Esta pasta armazena os modelos convolucionais exportados no formato **ONNX** e seus mapeamentos de classes.

O backend FastAPI (`app/inferencia.py`) monitora esta pasta dinamicamente via bind-mount: assim que um arquivo `.onnx` é adicionado ou atualizado aqui, ele é carregado em tempo de execução sem reiniciar o servidor.

---

## 1. Batata (`batata.onnx`)
- **Status:** Disponível / Treinado
- **Formato:** Entrada `(batch, 3, 224, 224)` normalizada com ImageNet, Saída `(batch, 3)` logits
- **Classes (`modelos/batata.classes.json`):**
  0. `Potato___Early_blight` (pinta-preta)
  1. `Potato___Late_blight` (requeima)
  2. `Potato___healthy` (saudável)

---

## 2. Milho (`milho.onnx`)
- **Status:** Aguardando arquivo exportado do notebook
- **Formato:** Entrada `(batch, 3, 224, 224)` normalizada com ImageNet, Saída `(batch, 4)` logits
- **Classes (`modelos/milho.classes.json`):**
  0. `Corn_(maize)___Cercospora_leaf_spot Gray_leaf_spot`
  1. `Corn_(maize)___Common_rust_`
  2. `Corn_(maize)___Northern_Leaf_Blight`
  3. `Corn_(maize)___healthy`

---

## 3. Tomate (`tomate.onnx`)
- **Status:** Aguardando arquivo exportado do notebook
- **Formato:** Entrada `(batch, 3, 224, 224)` normalizada com ImageNet, Saída `(batch, 10)` logits
- **Classes (`modelos/tomate.classes.json`):**
  0. `Tomato___Bacterial_spot`
  1. `Tomato___Early_blight`
  2. `Tomato___Late_blight`
  3. `Tomato___Leaf_Mold`
  4. `Tomato___Septoria_leaf_spot`
  5. `Tomato___Spider_mites Two-spotted_spider_mite`
  6. `Tomato___Target_Spot`
  7. `Tomato___Tomato_Yellow_Leaf_Curl_Virus`
  8. `Tomato___Tomato_mosaic_virus`
  9. `Tomato___healthy`

---

## Como Adicionar os Modelos Faltantes
1. Execute o notebook `notebooks/01_treino_culturas.ipynb` no Google Colab ou localmente.
2. Ao final da execução, o download de `modelos_agrosaude.zip` será iniciado automaticamente.
3. Extraia o zip e copie `tomate.onnx` e `milho.onnx` diretamente para esta pasta (`modelos/`).
4. A interface web em `http://localhost:8000` atualizará os badges verdes instantaneamente.
