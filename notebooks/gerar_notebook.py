"""
Gera o notebook único de treinamento de visão computacional: notebooks/01_treino_culturas.ipynb
Treina em um único loop as 3 culturas (Batata, Milho e Tomate) com base na pasta escolha/.
Uso: python3 notebooks/gerar_notebook.py
"""
import json
import os

cells = []


def md(text):
    cells.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(True)})


def code(text):
    cells.append({
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.strip("\n").splitlines(True),
    })


# -------------------------------------------------------------------------
# CÉLULA 1: Cabeçalho e Mapeamento PjBL2
# -------------------------------------------------------------------------
md("""
# AgroSaúde – Treinamento Unificado de Visão Computacional (PjBL2)
### Pipeline Automatizado para as Três Culturas: Batata, Milho e Tomate

Este notebook treina sequencialmente os classificadores convolucionais das três culturas agrícolas a partir da pasta **`escolha/`**:
1. **Batata** (3 classes): `Early_blight`, `Late_blight`, `healthy` *(Entrega principal do PjBL2)*
2. **Milho** (4 classes): `Cercospora/Gray_leaf_spot`, `Common_rust`, `Northern_Leaf_Blight`, `healthy`
3. **Tomate** (10 classes): 9 patologias foliares + folha saudável

Ao final da execução, todos os modelos serão exportados para o formato **ONNX**, validados numericamente e compactados em um único arquivo **`modelos_agrosaude.zip`** para download direto.

---

### Correspondência com os Requisitos do PjBL2 (Prof. Dr. Joelton Deonei Gotz):
| Requisito do Enunciado | Implementação no Notebook |
|---|---|
| **1. Base de imagens com ≥ 3 categorias** | Dataset PlantVillage na pasta `escolha/` (Batata: 3, Milho: 4, Tomate: 10 classes) |
| **2. Protocolo experimental (treino/val/teste)** | Divisão estratificada **70% Treino / 15% Validação / 15% Teste** com semente fixa (`seed=42`) |
| **3. Modelo de visão computacional** | **MobileNetV2** com Transfer Learning em duas fases (cabeça linear + fine-tuning) |
| **4. Métricas e Matriz de Confusão** | Acurácia, F1-Score (macro e por classe), Precisão, Revocação e Matriz de Confusão anotada |

**Diferenciais AgroSaúde:**
- Checkpoint em disco durante o treino (`checkpoint_<cultura>_melhor.pt`)
- Análise de incerteza pelo limiar de confiança (0.70)
- Explicabilidade visual com **Grad-CAM** em folhas de teste
- Exportação para **ONNX** com validação estrita contra o PyTorch (erro < 1e-3)
- Limpeza sistemática de VRAM entre culturas para evitar estouro de memória
""")

# -------------------------------------------------------------------------
# CÉLULA 2: Instalação de Dependências
# -------------------------------------------------------------------------
md("## 1. Instalação de Dependências e Importações")
code("""
# Instalação das dependências necessárias no ambiente do Colab
!pip -q install onnx onnxruntime scikit-learn seaborn

import os
import gc
import glob
import json
import time
import zipfile
import random
import platform
import shutil
from collections import Counter
from typing import Dict, List, Tuple

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from PIL import Image

import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_recall_fscore_support,
    confusion_matrix,
    classification_report
)

import onnx
import onnxruntime as ort

print(f"PyTorch versão: {torch.__version__}")
print(f"Disponibilidade CUDA: {torch.cuda.is_available()}")
""")

# -------------------------------------------------------------------------
# CÉLULA 3: Google Drive e Reprodutibilidade
# -------------------------------------------------------------------------
md("""
## 2. Montagem do Google Drive (Opcional) e Sementes de Reprodutibilidade
Se você salvou sua pasta `escolha/` ou `escolha.zip` no seu Google Drive, execute a célula abaixo para montá-lo.
Caso vá fazer upload direto pela aba lateral de arquivos do Colab, basta deixar como está.
""")
code("""
# Descomente as duas linhas abaixo caso queira carregar diretamente do Google Drive:
# from google.colab import drive
# drive.mount('/content/drive')

def fixar_sementes(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

fixar_sementes(42)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
GPU_NOME = torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU"
print(f"Dispositivo de processamento: {DEVICE} ({GPU_NOME})")
if DEVICE.type == "cpu":
    print("ATENÇÃO: Recomenda-se ativar a GPU em 'Ambiente de execução -> Alterar tipo -> GPU T4'.")
""")

# -------------------------------------------------------------------------
# CÉLULA 4: Localização e Descompactação da pasta escolha/
# -------------------------------------------------------------------------
md("""
## 3. Localização da Base de Imagens (`escolha/`)
O código abaixo localiza a pasta `escolha/` onde estão as 3 culturas (`batata`, `milho`, `tomate`).
Se você fez upload de um arquivo `.zip` (como `escolha.zip`), ele será extraído automaticamente.
""")
code("""
def localizar_ou_extrair_dataset() -> str:
    \"\"\"Localiza a pasta 'escolha' ou descompacta arquivos zip encontrados.\"\"\"
    # 1. Se encontrar algum zip de escolha em /content ou no diretório atual, extrai
    zips_encontrados = glob.glob("/content/*.zip") + glob.glob("*.zip")
    for z in zips_encontrados:
        if "escolha" in z.lower() or "plant" in z.lower() or "dataset" in z.lower():
            print(f"Arquivo compactado detectado: {z}. Descompactando...")
            try:
                with zipfile.ZipFile(z, "r") as zf:
                    zf.extractall("/content")
                print("Extração concluída!")
            except Exception as e:
                print(f"Erro ao extrair {z}: {e}")

    # 2. Possíveis locais onde a pasta 'escolha' pode estar
    candidatos = [
        "/content/escolha",
        "./escolha",
        "../escolha",
        "/content/drive/MyDrive/escolha",
        "/content/drive/MyDrive/planthealth/escolha",
    ]

    for c in candidatos:
        if os.path.isdir(c):
            # Valida se contém pelo menos uma das culturas esperadas
            subpastas = [d.lower() for d in os.listdir(c) if os.path.isdir(os.path.join(c, d))]
            if any(cult in subpastas for cult in ["batata", "milho", "tomate"]):
                print(f"Base de dados 'escolha' confirmada em: {c}")
                return os.path.abspath(c)

    # 3. Se as pastas das culturas estiverem diretamente em /content
    if os.path.isdir("/content/batata") or os.path.isdir("./batata"):
        diretorio_pai = "/content" if os.path.isdir("/content/batata") else "."
        print(f"Pastas das culturas encontradas diretamente em: {diretorio_pai}")
        return os.path.abspath(diretorio_pai)

    raise FileNotFoundError(
        "A pasta 'escolha/' não foi encontrada!\\n"
        "Envie a pasta 'escolha' ou o arquivo 'escolha.zip' para os arquivos do Colab."
    )

PASTA_ESCOLHA = localizar_ou_extrair_dataset()

# Validação das 3 culturas
print("\\nVerificando subpastas das culturas em:", PASTA_ESCOLHA)
for c in ["batata", "milho", "tomate"]:
    caminho_c = os.path.join(PASTA_ESCOLHA, c)
    if os.path.isdir(caminho_c):
        classes_c = sorted([d for d in os.listdir(caminho_c) if os.path.isdir(os.path.join(caminho_c, d)) and not d.startswith(".")])
        print(f"  ✓ {c.capitalize():7s}: {len(classes_c)} classes encontradas: {classes_c}")
    else:
        print(f"  ✗ {c.capitalize():7s}: NÃO encontrada em {caminho_c}")
""")

# -------------------------------------------------------------------------
# CÉLULA 5: Hiperparâmetros Globais
# -------------------------------------------------------------------------
md("## 4. Configurações Globais do Treinamento")
code("""
# Lista das culturas a serem treinadas na ordem recomendada (mais rápida para a maior)
CULTURAS = ["batata", "milho", "tomate"]

SEED = 42
IMG_SIZE = 224
BATCH_SIZE = 32
EPOCAS_CABECA = 5       # Fase 1: apenas a camada linear de saída
EPOCAS_FINETUNE = 10    # Fase 2: fine-tuning de toda a rede neural
LR_CABECA = 1e-3
LR_FINETUNE = 1e-4
LIMIAR = 0.70           # Limiar mínimo de confiança do sistema (70%)
PACIENCIA = 4           # Early stopping se a acurácia de validação não melhorar

MEDIA = [0.485, 0.456, 0.406]   # Estatísticas oficiais do ImageNet
DESVIO = [0.229, 0.224, 0.225]

print("Configurações fixadas para todas as culturas:")
print(f"  Culturas a executar : {CULTURAS}")
print(f"  Tamanho da Imagem   : {IMG_SIZE}x{IMG_SIZE}")
print(f"  Batch Size          : {BATCH_SIZE}")
print(f"  Épocas Cabeça / Fine: {EPOCAS_CABECA} / {EPOCAS_FINETUNE}")
print(f"  Limiar de Incerteza : {LIMIAR:.2f}")
""")

# -------------------------------------------------------------------------
# CÉLULA 6: Dataset, Augmentations e DataLoaders
# -------------------------------------------------------------------------
md("## 5. Módulo de Carregamento de Dados e Augmentation")
code("""
# Augmentation para treino (rotaciona, distorce cores e inverte para robustez a campo)
tf_treino = transforms.Compose([
    transforms.RandomResizedCrop(IMG_SIZE, scale=(0.7, 1.0)),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomRotation(20),
    transforms.ColorJitter(brightness=0.25, contrast=0.25, saturation=0.2),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIO),
])

# Pipeline determinístico para validação e teste (idêntico à inferência na API)
tf_eval = transforms.Compose([
    transforms.Resize((IMG_SIZE, IMG_SIZE)),
    transforms.ToTensor(),
    transforms.Normalize(MEDIA, DESVIO),
])

class FolhasDataset(Dataset):
    def __init__(self, caminhos: List[str], rotulos: List[int], transform):
        self.caminhos = caminhos
        self.rotulos = rotulos
        self.transform = transform

    def __len__(self):
        return len(self.caminhos)

    def __getitem__(self, idx):
        caminho = self.caminhos[idx]
        img = Image.open(caminho).convert("RGB")
        tensor = self.transform(img)
        rotulo = self.rotulos[idx]
        return tensor, rotulo

def preparar_dados_cultura(planta: str, pasta_raiz: str):
    \"\"\"Carrega amostras, faz o split estratificado 70/15/15 e calcula pesos de classe.\"\"\"
    dir_cultura = os.path.join(pasta_raiz, planta)
    if not os.path.isdir(dir_cultura):
        raise FileNotFoundError(f"Pasta da cultura '{planta}' não encontrada em {dir_cultura}")

    # Ordem alfabética rigorosa das subpastas (garante paridade exata com a API)
    classes = sorted([
        d for d in os.listdir(dir_cultura)
        if os.path.isdir(os.path.join(dir_cultura, d)) and not d.startswith(".")
    ])
    classe_para_idx = {c: i for i, c in enumerate(classes)}

    exts = (".jpg", ".jpeg", ".png", ".bmp")
    amostras = []
    for c in classes:
        p_classe = os.path.join(dir_cultura, c)
        for arq in sorted(os.listdir(p_classe)):
            if arq.lower().endswith(exts):
                amostras.append((os.path.join(p_classe, arq), classe_para_idx[c]))

    caminhos = [p for p, _ in amostras]
    rotulos = [y for _, y in amostras]

    # Protocolo Experimental PjBL2: 70% Treino / 15% Validação / 15% Teste estratificado
    X_tr, X_temp, y_tr, y_temp = train_test_split(
        caminhos, rotulos, test_size=0.30, stratify=rotulos, random_state=SEED
    )
    X_va, X_te, y_va, y_te = train_test_split(
        X_temp, y_temp, test_size=0.50, stratify=y_temp, random_state=SEED
    )

    g = torch.Generator()
    g.manual_seed(SEED)
    num_workers = 2 if os.name != "nt" else 0

    dl_tr = DataLoader(FolhasDataset(X_tr, y_tr, tf_treino), batch_size=BATCH_SIZE, shuffle=True, num_workers=num_workers, generator=g)
    dl_va = DataLoader(FolhasDataset(X_va, y_va, tf_eval), batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)
    dl_te = DataLoader(FolhasDataset(X_te, y_te, tf_eval), batch_size=BATCH_SIZE, shuffle=False, num_workers=num_workers)

    # Cálculo dos pesos para contrabalancear classes desiguais
    cont = Counter(y_tr)
    pesos = [len(y_tr) / (len(classes) * cont[i]) for i in range(len(classes))]
    pesos_tensor = torch.tensor(pesos, dtype=torch.float32).to(DEVICE)

    divisao = {"total": len(amostras), "treino": len(X_tr), "validacao": len(X_va), "teste": len(X_te)}
    return classes, amostras, (X_tr, y_tr), (X_va, y_va), (X_te, y_te), (dl_tr, dl_va, dl_te), pesos_tensor, divisao

print("Módulo de dados configurado com sucesso.")
""")

# -------------------------------------------------------------------------
# CÉLULA 7: Rotina de Treinamento com Checkpoint em Disco
# -------------------------------------------------------------------------
md("## 6. Rotina de Treinamento com Checkpoint em Disco")
code("""
def criar_modelo(num_classes: int) -> nn.Module:
    \"\"\"Cria a arquitetura MobileNetV2 com pesos pré-treinados no ImageNet.\"\"\"
    modelo = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    modelo.classifier[1] = nn.Linear(modelo.last_channel, num_classes)
    return modelo.to(DEVICE)

def rodar_epoca(modelo: nn.Module, dl: DataLoader, criterio, otimizador, scaler, usar_amp: bool) -> Tuple[float, float]:
    treinando = otimizador is not None
    modelo.train(treinando)
    perda_total, acertos, total = 0.0, 0, 0

    with torch.set_grad_enabled(treinando):
        for x, y in dl:
            x, y = x.to(DEVICE), y.to(DEVICE)
            with torch.autocast(device_type=DEVICE.type, enabled=usar_amp):
                saida = modelo(x)
                perda = criterio(saida, y)

            if treinando:
                otimizador.zero_grad()
                scaler.scale(perda).backward()
                scaler.step(otimizador)
                scaler.update()

            perda_total += perda.item() * x.size(0)
            acertos += (saida.argmax(dim=1) == y).sum().item()
            total += x.size(0)

    return perda_total / total, acertos / total

def treinar_cultura(planta: str, modelo: nn.Module, dl_tr: DataLoader, dl_va: DataLoader, pesos: torch.Tensor):
    \"\"\"Treina o modelo em 2 fases e grava checkpoints em disco a cada recorde de validação.\"\"\"
    criterio = nn.CrossEntropyLoss(weight=pesos, label_smoothing=0.05)
    usar_amp = DEVICE.type == "cuda"
    scaler = torch.amp.GradScaler("cuda", enabled=usar_amp)

    logs_epocas = []
    melhor_acc = 0.0
    arquivo_checkpoint = f"checkpoint_{planta}_melhor.pt"

    def executar_fase(n_epocas: int, otimizador, nome_fase: str):
        nonlocal melhor_acc
        sem_melhora = 0
        print(f"\\n[{planta.upper()}] Iniciando {nome_fase} ({n_epocas} épocas)...")
        for ep in range(1, n_epocas + 1):
            t0 = time.time()
            l_tr, a_tr = rodar_epoca(modelo, dl_tr, criterio, otimizador, scaler, usar_amp)
            l_va, a_va = rodar_epoca(modelo, dl_va, criterio, None, scaler, usar_amp)
            dt = time.time() - t0

            num_ep = len(logs_epocas) + 1
            logs_epocas.append({
                "fase": nome_fase, "epoca": num_ep,
                "loss_treino": round(l_tr, 5), "acc_treino": round(a_tr, 5),
                "loss_val": round(l_va, 5), "acc_val": round(a_va, 5),
                "segundos": round(dt, 1)
            })

            print(f"  [{nome_fase:8s}] Época {num_ep:02d} | Treino: loss {l_tr:.4f} acc {a_tr:.4f} | Val: loss {l_va:.4f} acc {a_va:.4f} ({dt:.1f}s)")

            if a_va > melhor_acc:
                melhor_acc = a_va
                sem_melhora = 0
                # Salva checkpoint em disco para segurança contra perda de sessão
                torch.save(modelo.state_dict(), arquivo_checkpoint)
                print(f"    -> Checkpoint salvo em disco: {arquivo_checkpoint} (Val Acc: {melhor_acc:.4f})")
            else:
                sem_melhora += 1
                if sem_melhora >= PACIENCIA:
                    print(f"    -> Early stopping acionado na fase {nome_fase}.")
                    break

    # FASE 1: Apenas a cabeça
    for p in modelo.features.parameters():
        p.requires_grad = False
    opt1 = torch.optim.AdamW(modelo.classifier.parameters(), lr=LR_CABECA, weight_decay=1e-4)
    executar_fase(EPOCAS_CABECA, opt1, "cabeca")

    # FASE 2: Fine-tuning de toda a rede
    for p in modelo.parameters():
        p.requires_grad = True
    opt2 = torch.optim.AdamW(modelo.parameters(), lr=LR_FINETUNE, weight_decay=1e-4)
    executar_fase(EPOCAS_FINETUNE, opt2, "finetune")

    # Restaura o melhor modelo a partir do disco
    if os.path.exists(arquivo_checkpoint):
        modelo.load_state_dict(torch.load(arquivo_checkpoint, map_location=DEVICE))
        print(f"[{planta.upper()}] Modelo restaurado do melhor checkpoint. Acurácia de Validação: {melhor_acc:.4f}")

    return logs_epocas, melhor_acc

print("Rotina de treinamento compilada com sucesso.")
""")

# -------------------------------------------------------------------------
# CÉLULA 8: Avaliação, Matriz de Confusão, Limiar e Grad-CAM
# -------------------------------------------------------------------------
md("## 7. Métricas de Teste, Matriz de Confusão e Grad-CAM")
code("""
def grad_cam(modelo: nn.Module, tensor_img: torch.Tensor, classe_alvo: int = None):
    modelo.eval()
    feats, grads = [], []
    def hook(mod, entrada, saida):
        feats.append(saida)
        saida.register_hook(lambda g: grads.append(g))
    h = modelo.features[-1].register_forward_hook(hook)
    saida = modelo(tensor_img)
    h.remove()
    if classe_alvo is None:
        classe_alvo = saida.argmax(dim=1).item()
    modelo.zero_grad()
    saida[0, classe_alvo].backward()
    pesos = grads[0].mean(dim=(2, 3), keepdim=True)
    cam = torch.relu((pesos * feats[0]).sum(dim=1))[0].detach().cpu().numpy()
    cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-8)
    return cam, classe_alvo

def avaliar_e_visualizar(planta: str, modelo: nn.Module, dl_te: DataLoader, X_te: List[str], y_te: List[int], classes: List[str], logs_epocas: list):
    \"\"\"Gera relatório completo de métricas, curvas de perda, matriz de confusão e Grad-CAM.\"\"\"
    modelo.eval()
    todas_probs, todos_y = [], []
    with torch.no_grad():
        for x, y in dl_te:
            probs = torch.softmax(modelo(x.to(DEVICE)).float(), dim=1).cpu().numpy()
            todas_probs.append(probs)
            todos_y.extend(y.numpy().tolist())

    probs_te = np.concatenate(todas_probs, axis=0)
    y_true = np.array(todos_y)
    y_pred = probs_te.argmax(axis=1)

    acc = float(accuracy_score(y_true, y_pred))
    f1_m = float(f1_score(y_true, y_pred, average="macro"))
    prec, rev, f1, sup = precision_recall_fscore_support(y_true, y_pred, labels=range(len(classes)), zero_division=0)
    cm = confusion_matrix(y_true, y_pred, labels=range(len(classes)))

    nomes_curtos = [c.replace("Potato___", "").replace("Corn_(maize)___", "").replace("Tomato___", "") for c in classes]

    print("=" * 65)
    print(f"DESEMPENHO NO TESTE – CULTURA: {planta.upper()}")
    print(f"  Acurácia Global : {acc:.4f} ({acc*100:.2f}%)")
    print(f"  F1-Score Macro  : {f1_m:.4f}")
    print("=" * 65)
    print(classification_report(y_true, y_pred, target_names=nomes_curtos, digits=4))

    # 1. Gráficos das Curvas de Aprendizado
    epocas = [l["epoca"] for l in logs_epocas]
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.8))
    ax1.plot(epocas, [l["loss_treino"] for l in logs_epocas], "o-", label="Treino", color="#2563eb")
    ax1.plot(epocas, [l["loss_val"] for l in logs_epocas], "s-", label="Validação", color="#ea580c")
    ax1.set_title(f"Perda (Cross-Entropy) – {planta.capitalize()}", fontweight="bold")
    ax1.set_xlabel("Época"); ax1.set_ylabel("Loss"); ax1.grid(True, linestyle="--", alpha=0.5); ax1.legend()

    ax2.plot(epocas, [l["acc_treino"] for l in logs_epocas], "o-", label="Treino", color="#2563eb")
    ax2.plot(epocas, [l["acc_val"] for l in logs_epocas], "s-", label="Validação", color="#16a34a")
    ax2.set_title(f"Acurácia – {planta.capitalize()}", fontweight="bold")
    ax2.set_xlabel("Época"); ax2.set_ylabel("Acurácia"); ax2.grid(True, linestyle="--", alpha=0.5); ax2.legend()
    plt.tight_layout()
    plt.show()

    # 2. Matriz de Confusão
    plt.figure(figsize=(max(5.5, len(classes) * 0.8), max(4.5, len(classes) * 0.7)))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Greens", xticklabels=nomes_curtos, yticklabels=nomes_curtos, cbar=False)
    plt.xlabel("Previsto pelo Modelo", fontweight="bold")
    plt.ylabel("Classe Real (Gabarito)", fontweight="bold")
    plt.title(f"Matriz de Confusão (Teste) – {planta.capitalize()}", fontweight="bold", fontsize=11)
    plt.xticks(rotation=35, ha="right"); plt.yticks(rotation=0)
    plt.tight_layout()
    plt.show()

    # 3. Análise do Limiar de Confiança
    confiancas = probs_te.max(axis=1)
    aceitas = confiancas >= LIMIAR
    acertos = (y_pred == y_true)
    print(f"Limiar de Incerteza ({LIMIAR}):")
    print(f"  Fotos aceitas com alta confiança: {aceitas.mean():.1%}")
    print(f"  Acurácia nas fotos aceitas      : {acertos[aceitas].mean():.4f} ({acertos[aceitas].mean()*100:.2f}%)")
    print(f"  Fotos encaminhadas para refazer : {(~aceitas).mean():.1%}")

    # 4. Amostra Grad-CAM
    try:
        indices_demo = [next(i for i, y in enumerate(y_te) if y == c) for c in range(len(classes))]
        fig, eixos = plt.subplots(2, len(indices_demo), figsize=(3.2 * len(indices_demo), 6))
        if len(indices_demo) == 1: eixos = eixos.reshape(2, 1)
        for k, idx in enumerate(indices_demo):
            img_pil = Image.open(X_te[idx]).convert("RGB").resize((IMG_SIZE, IMG_SIZE))
            t = tf_eval(img_pil).unsqueeze(0).to(DEVICE)
            cam, p_idx = grad_cam(modelo, t)
            h_np = np.array(Image.fromarray((cam * 255).astype(np.uint8)).resize((IMG_SIZE, IMG_SIZE), Image.BILINEAR)) / 255.0

            eixos[0, k].imshow(img_pil); eixos[0, k].axis("off")
            eixos[0, k].set_title(nomes_curtos[y_te[idx]], fontsize=9, fontweight="bold")
            eixos[1, k].imshow(img_pil); eixos[1, k].imshow(h_np, cmap="jet", alpha=0.45); eixos[1, k].axis("off")
            cor = "green" if p_idx == y_te[idx] else "red"
            eixos[1, k].set_title(f"Pred: {nomes_curtos[p_idx]}", fontsize=9, color=cor, fontweight="bold")
        plt.suptitle(f"Explicabilidade Grad-CAM – {planta.capitalize()}", fontsize=11, fontweight="bold")
        plt.tight_layout()
        plt.show()
    except Exception as ex:
        print(f"Aviso no Grad-CAM: {ex}")

    metricas = {
        "acuracia": acc,
        "f1_macro": f1_m,
        "precisao_macro": float(prec.mean()),
        "revocacao_macro": float(rev.mean()),
        "por_classe": {
            classes[i]: {"precisao": float(prec[i]), "revocacao": float(rev[i]), "f1": float(f1[i]), "suporte": int(sup[i])}
            for i in range(len(classes))
        }
    }
    return metricas, cm

print("Módulo de métricas e visualizações pronto.")
""")

# -------------------------------------------------------------------------
# CÉLULA 9: Exportação para ONNX e Validação
# -------------------------------------------------------------------------
md("## 8. Exportação para ONNX e Validação Numérica")
code("""
def exportar_modelo_onnx(planta: str, modelo: nn.Module, X_te: List[str], classes: List[str], metricas: dict, cm, logs_epocas: list, divisao: dict):
    \"\"\"Exporta o modelo para ONNX, valida contra o PyTorch e salva metadados.\"\"\"
    arquivo_onnx = f"{planta}.onnx"
    arquivo_classes = f"{planta}.classes.json"
    arquivo_metricas = f"metricas_{planta}.json"

    modelo_cpu = modelo.to("cpu").eval()
    amostra = torch.randn(1, 3, IMG_SIZE, IMG_SIZE)

    params = dict(
        input_names=["input"],
        output_names=["logits"],
        dynamic_axes={"input": {0: "batch"}, "logits": {0: "batch"}},
        opset_version=17
    )

    try:
        torch.onnx.export(modelo_cpu, amostra, arquivo_onnx, dynamo=False, **params)
    except TypeError:
        torch.onnx.export(modelo_cpu, amostra, arquivo_onnx, **params)

    # Verificação de integridade ONNX
    onnx_mod = onnx.load(arquivo_onnx)
    onnx.checker.check_model(onnx_mod)

    # Paridade numérica PyTorch vs ONNX
    sess = ort.InferenceSession(arquivo_onnx, providers=["CPUExecutionProvider"])
    lote = torch.stack([tf_eval(Image.open(p).convert("RGB")) for p in X_te[:16]])
    with torch.no_grad():
        out_pt = modelo_cpu(lote).numpy()
    out_ort = sess.run(None, {sess.get_inputs()[0].name: lote.numpy()})[0]
    dif = float(np.abs(out_pt - out_ort).max())

    print(f"[{planta.upper()}] Erro máximo absoluto PyTorch x ONNX: {dif:.2e}")
    assert dif < 1e-3, f"Inconsistência numérica no ONNX de {planta}!"
    print(f"[{planta.upper()}] Modelo ONNX validado e aprovado!")

    # Metadados
    with open(arquivo_classes, "w", encoding="utf-8") as f:
        json.dump({"planta": planta, "classes": classes}, f, ensure_ascii=False, indent=2)

    with open(arquivo_metricas, "w", encoding="utf-8") as f:
        json.dump({
            "planta": planta,
            "classes": classes,
            "metricas": metricas,
            "matriz_confusao": cm.tolist(),
            "divisao": divisao,
            "logs_epocas": logs_epocas
        }, f, ensure_ascii=False, indent=2)

    tamanho_mb = os.path.getsize(arquivo_onnx) / (1024 * 1024)
    print(f"[{planta.upper()}] Arquivos gerados: {arquivo_onnx} ({tamanho_mb:.2f} MB), {arquivo_classes}, {arquivo_metricas}\\n")
    return arquivo_onnx, arquivo_classes, arquivo_metricas

print("Módulo de exportação ONNX compilado.")
""")

# -------------------------------------------------------------------------
# CÉLULA 10: O GRANDE LOOP FOR (TREINA BATATA, MILHO E TOMATE)
# -------------------------------------------------------------------------
md("""
## 9. Execução do Loop Unificado (Batata, Milho e Tomate)
Esta célula executa o pipeline completo para cada uma das culturas em sequência:
1. Divide a base de dados (70/15/15).
2. Treina o classificador MobileNetV2 com checkpoint salvo em disco.
3. Avalia no conjunto de teste com matriz de confusão e Grad-CAM.
4. Exporta para ONNX e valida a paridade numérica.
5. **Limpa a VRAM da GPU** entre as culturas para garantir que a próxima cultura tenha memória máxima disponível.
""")
code("""
resultados_gerais = {}
arquivos_gerados = []

print("=" * 70)
print(f"INICIANDO PIPELINE UNIFICADO DE TREINAMENTO ({len(CULTURAS)} CULTURAS)")
print("=" * 70)

tempo_inicio_total = time.time()

for idx, planta in enumerate(CULTURAS, start=1):
    print(f"\\n{'#' * 70}")
    print(f"# CULTURA [{idx}/{len(CULTURAS)}]: {planta.upper()}")
    print(f"{'#' * 70}")

    # 1. Dados e Split
    classes, amostras, (X_tr, y_tr), (X_va, y_va), (X_te, y_te), (dl_tr, dl_va, dl_te), pesos, divisao = preparar_dados_cultura(planta, PASTA_ESCOLHA)
    print(f"Classes ({len(classes)}): {classes}")
    print(f"Amostras: Treino={len(X_tr)}, Validação={len(X_va)}, Teste={len(X_te)} (Total={len(amostras)})")

    # 2. Modelo
    modelo = criar_modelo(len(classes))

    # 3. Treino com checkpoint em disco
    logs, melhor_acc = treinar_cultura(planta, modelo, dl_tr, dl_va, pesos)

    # 4. Avaliação e Métricas
    metricas, cm = avaliar_e_visualizar(planta, modelo, dl_te, X_te, y_te, classes, logs)

    # 5. Exportação ONNX
    f_onnx, f_cls, f_met = exportar_modelo_onnx(planta, modelo, X_te, classes, metricas, cm, logs, divisao)
    arquivos_gerados.extend([f_onnx, f_cls, f_met])

    # Guarda resultado para resumo comparativo
    resultados_gerais[planta] = {
        "classes": len(classes),
        "total_fotos": len(amostras),
        "acc_teste": metricas["acuracia"],
        "f1_macro": metricas["f1_macro"]
    }

    # 6. LIMPEZA CRÍTICA DE MEMÓRIA GPU
    del modelo, dl_tr, dl_va, dl_te, X_tr, X_va, X_te, pesos
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    print(f"✓ VRAM liberada com sucesso após treino de {planta.capitalize()}.")

duracao_total = time.time() - tempo_inicio_total
print(f"\\n{'=' * 70}")
print(f"PIPELINE CONCLUÍDO COM SUCESSO EM {duracao_total / 60:.2f} MINUTOS!")
print(f"{'=' * 70}")
""")

# -------------------------------------------------------------------------
# CÉLULA 11: Resumo Comparativo e Empacotamento ZIP
# -------------------------------------------------------------------------
md("""
## 10. Consolidação dos Resultados e Download dos Modelos
Agrupa todos os arquivos gerados em um único arquivo compactado **`modelos_agrosaude.zip`** e inicia o download automático.
""")
code("""
print("=" * 65)
print("TABELA CONSOLIDADA DE RESULTADOS (PjBL2):")
print(f"{'Cultura':10s} | {'Classes':7s} | {'Amostras':8s} | {'Acurácia (Teste)':16s} | {'F1-Macro':8s}")
print("-" * 65)
for cult, res in resultados_gerais.items():
    print(f"{cult.capitalize():10s} | {res['classes']:7d} | {res['total_fotos']:8d} | {res['acc_teste']*100:6.2f}%          | {res['f1_macro']:.4f}")
print("=" * 65)

# Empacota todos os modelos em um único ZIP
ARQUIVO_ZIP = "modelos_agrosaude.zip"
with zipfile.ZipFile(ARQUIVO_ZIP, "w", zipfile.ZIP_DEFLATED) as zf:
    for arq in arquivos_gerados:
        if os.path.exists(arq):
            zf.write(arq, arcname=arq)
            print(f"Incluído no pacote: {arq}")

tamanho_zip_mb = os.path.getsize(ARQUIVO_ZIP) / (1024 * 1024)
print(f"\\nPacote compactado gerado: {ARQUIVO_ZIP} ({tamanho_zip_mb:.2f} MB)")

# Download automático no Colab
try:
    from google.colab import files
    print("Iniciando download automático de 'modelos_agrosaude.zip'...")
    files.download(ARQUIVO_ZIP)
    print("Download disparado no seu navegador!")
except ImportError:
    print(f"Ambiente local: o arquivo compactado está salvo em {os.path.abspath(ARQUIVO_ZIP)}.")
""")

# -------------------------------------------------------------------------
# CÉLULA 12: Instruções de Instalação na API
# -------------------------------------------------------------------------
md("""
## 11. Instruções Finais para o AgroSaúde
Com o arquivo **`modelos_agrosaude.zip`** baixado:
1. Extraia o conteúdo do zip.
2. Mova os arquivos `.onnx` e `.classes.json` para a pasta **`modelos/`** do seu projeto AgroSaúde:
   - `modelos/batata.onnx` e `modelos/batata.classes.json`
   - `modelos/milho.onnx` e `modelos/milho.classes.json`
   - `modelos/tomate.onnx` e `modelos/tomate.classes.json`
3. A interface web do AgroSaúde (`http://localhost:8000`) detectará os 3 modelos automaticamente com as pílulas verdes no topo!
""")

nb = {
    "cells": cells,
    "metadata": {
        "accelerator": "GPU",
        "colab": {"gpuType": "T4", "provenance": []},
        "kernelspec": {"display_name": "Python 3", "name": "python3"},
        "language_info": {"name": "python"},
    },
    "nbformat": 4,
    "nbformat_minor": 0,
}

if __name__ == "__main__":
    destino_dir = os.path.dirname(os.path.abspath(__file__))
    caminho_notebook = os.path.join(destino_dir, "01_treino_culturas.ipynb")
    with open(caminho_notebook, "w", encoding="utf-8") as f:
        json.dump(nb, f, ensure_ascii=False, indent=1)
    print(f"Notebook único gerado com sucesso em: {caminho_notebook} ({len(cells)} células)")
