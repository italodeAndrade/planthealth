# PjBL2 – Classificador de doenças da batata

Entrega do PjBL2 (Sistemas de Informação Inteligentes, PUCPR). Aqui estão o modelo de visão computacional, a base de imagens, o protocolo experimental e as métricas. O restante do sistema (API, interface e LLM) pertence ao TDE2 e não foi alterado nesta entrega.

## 1. Base de imagens
- **Fonte:** PlantVillage (Hughes; Salathé, 2015), cópia pública em <https://github.com/spMohanty/PlantVillage-Dataset>, pasta `raw/color` (commit `7f7ecc7`).
- Usamos **só a pasta `color`**, que é a mais parecida com as fotos que o usuário vai enviar. As pastas `grayscale` e `segmented` não entram: são versões das mesmas folhas e poderiam vazar entre os conjuntos.
- **3 categorias**, todas com imagens de 256×256 px:

| Classe | Doença | Imagens |
|---|---|---|
| `Potato___Early_blight` | pinta-preta (*Alternaria solani*) | 1000 |
| `Potato___Late_blight` | requeima (*Phytophthora infestans*) | 1000 |
| `Potato___healthy` | saudável | 152 |
| **Total** | | **2152** |

As imagens **não são versionadas**: estão no `.gitignore` e são baixadas por `scripts/baixar_dados.sh`.

## 2. Protocolo experimental
O script é `scripts/preparar_dados.py` e as listas geradas ficam em `dados/splits/`.
- A divisão é **estratificada** em **70% treino, 15% validação e 15% teste**, com **semente 42**.
- Ela é feita **antes** de qualquer aumento de dados. O aumento é aplicado só em memória e só no treino.
- Imagens idênticas (mesmo MD5) ficariam no mesmo conjunto. A verificação não encontrou nenhuma duplicata.

| Classe | Treino | Validação | Teste |
|---|---|---|---|
| Early_blight | 700 | 150 | 150 |
| Late_blight | 700 | 150 | 150 |
| healthy | 106 | 23 | 23 |
| **Total** | **1506** | **323** | **323** |

Papel de cada conjunto:
- **Treino:** ajusta os pesos.
- **Validação:** escolhe a melhor época e calibra o limiar de confiança.
- **Teste:** usado **uma única vez**, no final.

## 3. Modelo
Notebook: `notebooks/01_treino_batata.ipynb`, já executado e com as saídas salvas. Framework: PyTorch.
- **Arquitetura:** MobileNetV2 pré-treinada na ImageNet (Sandler et al., 2018), com a última camada trocada por `Dropout(0.2)` e `Linear(1280 → 3)`.
- **Etapa 1:** backbone congelado, treino só da cabeça. 6 épocas, Adam com lr 1e-3.
- **Etapa 2:** ajuste fino leve. Descongela os últimos 5 blocos convolucionais (1,69 M de 2,23 M parâmetros). 6 épocas, Adam com lr 1e-4.
- **Desbalanceamento:** pesos de classe na entropia cruzada, inversos à frequência (healthy = 4,74; doenças = 0,72).
- **Aumento de dados (só no treino):** recorte/zoom (80–100%), espelhamento horizontal e vertical, rotação de ±20° e variação de brilho, contraste e saturação.
- **Pré-processamento:** idêntico ao de `app/inferencia.py` (`resize` para 224×224, /255, normalização da ImageNet, NCHW).
- **Outros hiperparâmetros:** lote 32, weight decay 1e-4, BatchNorm congelado.
- **Seleção do modelo:** a época com maior F1 macro na validação (época 12, F1 = 1,000).
- **Tempo:** cerca de 5 min em CPU (8 threads, sem GPU).

As curvas de treino estão em `modelos/curvas_treino_batata.png`.

## 4. Resultados no conjunto de teste (323 imagens)

| Classe | Precisão | Revocação | F1 | Suporte |
|---|---|---|---|---|
| Early_blight | 1,0000 | 1,0000 | 1,0000 | 150 |
| Late_blight | 1,0000 | 0,9933 | 0,9967 | 150 |
| healthy | 0,9583 | 1,0000 | 0,9787 | 23 |
| **Macro** | **0,9861** | **0,9978** | **0,9918** | 323 |

- **Acurácia: 0,9969.** Houve 1 erro: uma folha com requeima classificada como saudável, com confiança de 0,886.
- **Matriz de confusão** (`modelos/matriz_confusao_batata.png`, linhas = real, colunas = prevista):

![Matriz de confusão](../modelos/matriz_confusao_batata.png)

Todas as métricas, o histórico por época e a matriz estão em `modelos/metricas_batata.json`.

### Limiar de confiança (calibrado na validação)
- **Regra:** entre os limiares com cobertura ≥ 97% na validação, escolhe-se o de maior acurácia das fotos aceitas; em caso de empate, o maior. Resultado: **limiar = 0,98**.
- **No teste:** com 0,98, 96,6% das fotos são aceitas, com 100% de acurácia entre elas. O único erro é rejeitado (o sistema pediria uma nova foto).
- **Comparação com o app:** o app usa por padrão o limiar **0,70**. Com ele, 99,7% das fotos são aceitas, mas aquele erro passaria como "saudável".

## 5. Exportação ONNX
- **Arquivo:** `modelos/batata.onnx`, 8,9 MB, opset 17.
- **Entrada:** `input` com forma `(lote, 3, 224, 224)`, float32, normalizado pela ImageNet.
- **Saída:** `logits` com forma `(lote, 3)`. O app aplica o softmax.
- **Ordem das classes:** a mesma de `CLASSES_BATATA` em `app/inferencia.py` e de `modelos/README.md`.
- **Verificação:** rodei as 323 imagens de teste com o `preprocessar_imagem` do próprio app. As probabilidades do ONNX e do PyTorch diferem no máximo 5,7e-6 e a classe prevista é a mesma nas 323 imagens.

## 6. Limitações
- **Fotos de laboratório:** as imagens têm fundo uniforme e uma folha por foto. Em fotos de campo o desempenho costuma cair muito (Mohanty et al., 2016). **Ainda falta avaliar fotos reais externas** e reportar o resultado separadamente (sugestão de pasta: `dados/fotos_reais/`).
- **Mesma folha em várias fotos:** a base não identifica a folha de origem. Fotos parecidas da mesma folha podem ter caído em conjuntos diferentes, o que pode deixar as métricas otimistas.
- **Classe saudável pequena:** só 23 imagens no teste, então as métricas dessa classe têm mais incerteza.
- **Escopo:** o modelo só reconhece essas 3 condições. Pragas, deficiências nutricionais ou fotos que não sejam de folha serão classificadas, erroneamente, em uma delas.

## 7. Como reproduzir
```bash
python -m venv .venv && source .venv/bin/activate
pip install --index-url https://download.pytorch.org/whl/cpu torch torchvision   # ou com GPU
pip install -r requirements-treino.txt
bash scripts/baixar_dados.sh          # baixa as 3 pastas de batata (~40 MB)
python scripts/preparar_dados.py      # divisão 70/15/15, semente 42
cd notebooks && jupyter nbconvert --to notebook --execute --inplace 01_treino_batata.ipynb
```
Também funciona no Google Colab: rode os dois scripts e depois o notebook.

## Referências
- HUGHES, D. P.; SALATHÉ, M. *An open access repository of images on plant health…* arXiv:1511.08060, 2015.
- MOHANTY, S. P.; HUGHES, D. P.; SALATHÉ, M. *Using deep learning for image-based plant disease detection.* Frontiers in Plant Science, 7:1419, 2016.
- SANDLER, M. et al. *MobileNetV2: inverted residuals and linear bottlenecks.* CVPR, 2018.
