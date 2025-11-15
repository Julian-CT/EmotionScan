# 🎭 Emotion Scan

> **Análise de Emoções e Sentimentos em Textos de Redes Sociais**  
> Projeto de Conclusão de Curso em Ciência da Computação

---

## 📖 Sobre o Projeto

O **Emotion Scan** é uma aplicação desenvolvida para identificar **emoções** e **sentimentos** em textos em português, especialmente em mensagens de redes sociais (como o Twitter/X).  
O sistema combina técnicas de **Processamento de Linguagem Natural (PLN)** e **Aprendizado de Máquina**, permitindo visualizar e comparar o desempenho de diferentes modelos de classificação.

O projeto foi desenvolvido como parte do **Trabalho de Conclusão de Curso (TCC)**, com foco em pesquisa aplicada em **classificação multilabel de emoções**, explorando tanto abordagens tradicionais quanto redes neurais profundas.

---

## 🧠 Modelos Implementados

| Modelo | Descrição | Tarefa | Principais Resultados |
|:-------|:-----------|:--------|:----------------------|
| **Multinomial Naive Bayes (MNB)** | Modelo clássico de ML baseado em probabilidades condicionais | Emoções / Sentimentos | Acurácia: 20% (emoções), 75% (sentimentos) |
| **BERTimbau Base** | Modelo pré-treinado para o português (baseado em BERT) | Emoções / Sentimentos | Acurácia superior ao MNB em ambas as tarefas |
| **BERTimbau + MLP** | Combina embeddings do BERTimbau com uma camada MLP para classificação multilabel | Emoções | Melhor desempenho geral (F1-score ≈ 0.82) |

---

## ⚙️ Instalação e Execução

### Pré-requisitos
- **Python 3.10+**
- **pip** atualizado (`python -m pip install --upgrade pip`)
- **Virtualenv** opcional, mas recomendado (`python -m venv .venv`)
- **Git LFS** instalado (necessário para baixar os modelos `.pt`)
- Navegador moderno (Chrome/Firefox/Edge)

### 1. Clonar o projeto
```bash
git clone https://github.com/<usuario>/EmotionScan.git
cd EmotionScan
git lfs install    # garante que os arquivos .pt sejam baixados
```

### 2. Configurar ambiente Python
```bash
# (opcional) ativar virtualenv
python -m venv .venv
source .venv/bin/activate        # macOS/Linux
.venv\Scripts\activate          # Windows

# instalar dependências do backend
pip install -r app/api/requirements.txt
```

### 3. Executar a API (Flask)
```bash
cd app/api
python api_server.py
```
A API sobe em `http://localhost:5000`. Os modelos são carregados na inicialização e os endpoints principais são:
- `POST /predict_csv?model=<ID>`
- `GET /metrics?model=<ID>`

### 4. Abrir a interface
1. Volte ao diretório raiz do projeto.
2. Abra o arquivo `app/frontend/index.html` diretamente no navegador (duplo clique ou `open app/frontend/index.html`).
3. A interface consome a API local em `http://localhost:5000`.

### 5. Classificar textos
1. Escolha um arquivo `.csv` ou `.xlsx` com coluna `text`.
2. Selecione o modelo desejado.
3. Clique em **CLASSIFICAR**.
4. Use **Desempenho do Modelo** para visualizar métricas.
5. Após a predição, baixe o CSV com os resultados pelo botão **Baixar Resultados (.csv)**.

### 6. Observações
- Logs de predição são salvos via API em `app/api/emotion_results_<timestamp>.json`.
- Métricas exibidas podem ser ajustadas editando os scripts em `app/api` (por exemplo `analyze_model_results.py`).
- Os pesos dos modelos ficam em `app/models/bertimbau-mlp-*` e são distribuídos via Git LFS; para atualizá-los, substitua os arquivos `.pt` nessas pastas.
