# Projeto-machine-learning

### About do Repositório (GitHub)

**Descrição:**

> Projeto de Machine Learning Clássico para predição contínua da resistência à compressão do betão/concreto em MPa (Regressão) e classificação de risco técnico estrutural em três faixas de conformidade (Classificação) com base no dataset UCI Concrete Compressive Strength.
> 
> 

**Tópicos / Tags:**
`machine-learning`, `python`, `scikit-learn`, `regression`, `classification`, `concrete-strength`, `civil-engineering`, `data-science`, `unisatc`

---
# Aplicação de Machine Learning na Previsão da Resistência e Classificação de Risco Técnico de Misturas de Concreto

![Python](https://img.shields.io/badge/Python-3.10%2B-blue?logo=python)
![scikit-learn](https://img.shields.io/badge/scikit--learn-1.3%2B-orange?logo=scikit-learn)
![Status](https://img.shields.io/badge/Status-Concluído-success)
![License](https://img.shields.io/badge/License-MIT-green)

Este repositório contém a implementação integral, reprodutível e documentada do projeto de Machine Learning aplicado ao controlo tecnológico e avaliação da qualidade de misturas cimentícias na Engenharia Civil[cite: 41]. 

A partir do conjunto de dados experimental *Concrete Compressive Strength* (UCI ID 165)[cite: 44], o pipeline resolve duas tarefas supervisionadas de forma integrada[cite: 42]:
1. **Regressão Contínua:** Estimativa da resistência mecânica à compressão axial ($f_{ck}$) em Megapascals ($\text{MPa}$)[cite: 41, 42].
2. **Classificação Multiclasse:** Estratificação do risco relativo de baixa resistência em três categorias de conformidade técnica ('Risco Técnico', 'Atenção' e 'Baixo Risco')[cite: 42].

---

## 📌 Arquitetura do Pipeline


```

```
              ┌───────────────────────────────┐
              │    Dados Brutos (UCI 165)     │
              │   1.030 linhas × 9 colunas    │
              └──────────────┬────────────────┘
                             │
                             ▼
              ┌───────────────────────────────┐
              │ Higiene Mínima Determinística │
              │  Remoção de 25 duplicadas     │
              │  Checagem de esquema (faixas) │
              └──────────────┬────────────────┘
                             │
                             ▼
              ┌───────────────────────────────┐
              │   Divisão: Teste Sagrado      │
              │ 80% Treino  │ 20% Teste (str.)│
              └──────┬────────────────┬───────┘
                     │                │
      ┌──────────────┘                └──────────────┐
      ▼                                              ▼

```

┌───────────────────────────┐                  ┌───────────────────┐
│     EDA no Treino         │                  │                   │
│ • Matriz de Correlação    │                  │                   │
│ • Dispersão idade × fck   │                  │   Dados de Teste  │
│ • Relação a/c (Abrams)    │                  │     (Intocados)   │
└─────────┬─────────────────┘                  │    201 amostras   │
│                                    │                   │
▼                                    │                   │
┌───────────────────────────┐                  │                   │
│ Pipeline Pré-processamento│                  │                   │
│ • SimpleImputer (mediana) │                  │                   │
│ • StandardScaler (z-score)│                  │                   │
│ • Fit APENAS no Treino    │                  │                   │
└─────────┬─────────────────┘                  │                   │
│                                    │                   │
├─────────────────────────┐          │                   │
▼                         ▼          │                   │
┌───────────────────┐     ┌──────────────────┐ │                   │
│     TAREFA 1:     │     │     TAREFA 2:    │ │                   │
│     REGRESSÃO     │     │   CLASSIFICAÇÃO  │ │                   │
│ • Linear Múltipla │     │ • Reg. Logística │ │                   │
│ • Polinomial Gr.2 │     │ • KNN (k=5)      │ │                   │
└─────────┬─────────┘     └─────────┬────────┘ │                   │
│                         │          │                   │
└────────────┬────────────┘          │                   │
│                       │                   │
▼                       ▼                   │
┌──────────────────────────────────────────────────┐               │
│               Avaliação no Teste                 │◄──────────────┘
│ • Regressão: MAE, RMSE, R², R² ajustado, Resíduos│
│ • Classificação: Accuracy, Precision, Recall, F1 │
│ • Comparação com Baselines (Média e Moda)        │
└──────────────────────────────────────────────────┘

```

---

## 🧱 Conjunto de Dados e Variáveis

O dataset provém de ensaios laboratoriais desenvolvidos por I-Cheng Yeh (1998)[cite: 42, 44]. Após a fase de higiene e validação determinística, a base consolidou-se em **1.005 registos válidos**[cite: 36].

| Variável | Significado Tecnológico | Unidade | Papel no Pipeline |
| :--- | :--- | :---: | :---: |
| `cimento` | Quantidade de cimento Portland por volume unitário | $\text{kg/m}^3$ | Atributo Preditor |
| `escoria` | Escória granulada de alto-forno por volume unitário | $\text{kg/m}^3$ | Atributo Preditor |
| `cinza_volante` | Cinza volante residual por volume unitário | $\text{kg/m}^3$ | Atributo Preditor |
| `agua` | Água livre de mistura e amassamento | $\text{kg/m}^3$ | Atributo Preditor |
| `superplastificante` | Aditivo químico redutor de água | $\text{kg/m}^3$ | Atributo Preditor |
| `agregado_graudo` | Quantidade de brita / agregado graúdo | $\text{kg/m}^3$ | Atributo Preditor |
| `agregado_miudo` | Quantidade de areia / agregado fino | $\text{kg/m}^3$ | Atributo Preditor |
| `idade` | Tempo de cura até à rotura mecânica | dias | Atributo Preditor |
| `relacao_ac` | Relação em massa água/cimento (Lei de Abrams) | adimensional | *Feature Engineering* |
| `resistencia` | Resistência mecânica à compressão | $\text{MPa}$ | **Alvo (Regressão)** |
| `classe_risco` | Faixa de risco técnico relativo | Categórico | **Alvo (Classificação)** |

### Definição das Classes de Risco Estrutural
As amostras foram estratificadas em três faixas de conformidade[cite: 42]:
* **Risco Técnico:** Resistência $< 25,0\ \text{MPa}$ ($28,1\%$ das amostras)[cite: 42].
* **Atenção:** Resistência no intervalo $[25,0;\ 35,0[\ \text{MPa}$ ($26,0\%$ das amostras)[cite: 42].
* **Baixo Risco:** Resistência $\ge 35,0\ \text{MPa}$ ($45,9\%$ das amostras)[cite: 42].

---

## 🔬 Metodologia e Boas Práticas

* **Higiene Determinística:** Eliminação de 25 duplicados exatos e verificação de regras de validação de domínio (ausência de valores negativos e campos nulos)[cite: 36].
* **Teste Sagrado (80/20):** Separação estratificada dos dados (`stratify=y_clf`, `random_state=42`), mantendo 201 observações estritamente isoladas até à avaliação final[cite: 36, 38].
* **EDA Restrito ao Treino:** Diagnóstico de dispersão, assimetrias e correlação linear de Pearson calculados exclusivamente sobre as 804 instâncias de treino, prevenindo vazamento de dados (*data leakage*)[cite: 36, 37].
* **Padronização (`StandardScaler`):** Conversão de escala por $Z\text{-score}$ ($z = \frac{x - \mu}{\sigma}$), com os parâmetros aprendidos no treino e aplicados de forma linear ao conjunto de teste[cite: 36, 38].

---

## 📊 Avaliação e Resultados (Conjunto de Teste)

A avaliação foi conduzida sobre as **201 amostras do conjunto de teste sagrado**[cite: 36, 38].

### 1. Tarefa de Regressão Contínua (Alvo: Resistência em MPa)

| Modelo | MAE (MPa) | RMSE (MPa) | $R^2$ | $R^2$ Ajustado | Diagnóstico dos Resíduos |
| :--- | :---: | :---: | :---: | :---: | :--- |
| **Baseline (Média do Treino)**[cite: 37, 40] | $13,42$ | $16,48$ | $0,000$ | $0,000$ | Erro sistemático severo[cite: 40] |
| **Linear Simples (Cimento)**[cite: 37] | $11,58$ | $14,41$ | $0,237$ | $0,233$ | Erro típico elevado[cite: 40] |
| **Linear Múltipla**[cite: 37] | $7,78$ | $9,86$ | $0,642$ | $0,627$ | Leve curvatura sistemática[cite: 40] |
| **Polinomial (Grau 2)**[cite: 36, 37] | **$5,31$** | **$7,08$** | **$0,816$** | **$0,764$** | **Nuvem aleatória centrada em zero**[cite: 40] |

### 2. Tarefa de Classificação (Alvo: Faixa de Risco Técnico)

| Classificador | Acurácia | Precisão (*Weighted*) | Recall (*Weighted*) | F1-Score (*Weighted*) | Recall ('Risco Técnico') |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Referência (Dummy - Moda)**[cite: 38] | $45,77\%$ | $20,95\%$ | $45,77\%$ | $28,74\%$ | $0,00\%$ |
| **Regressão Logística**[cite: 38] | $77,61\%$ | $77,15\%$ | $77,61\%$ | $77,34\%$ | $80,36\%$ |
| **KNN ($K=5$, Manhattan)**[cite: 49] | **$85,07\%$** | **$85,12\%$** | **$85,07\%$** | **$85,04\%$** | **$89,29\%$** |

> **Critério de Decisão de Engenharia:** Na triagem de segurança do betão, um **Falso Negativo (FN)** — aprovar uma mistura deficiente ($< 25\ \text{MPa}$) como se fosse segura — acarreta risco iminente de colapso estrutural[cite: 40, 42]. O modelo **KNN** destacou-se ao atingir **$89,29\%$ de Recall na classe crítica de risco técnico**, minimizando o envio de lotes inadequados para a obra[cite: 40, 49].

---

## 📁 Estrutura de Pastas do Repositório

```text
├── data/
│   ├── raw/                      # Ficheiros originais obtidos do repositório UCI
│   └── processed/                # Dados tratados após limpeza determinística
├── notebooks/
│   └── pipeline_concreto.ipynb   # Execução passo a passo interativa com gráficos
├── src/
│   ├── data_pipeline.py          # Script de higiene, cálculo da relação a/c e split
│   ├── regression.py             # Ajuste e validação dos modelos de regressão
│   └── classification.py         # Ajuste e validação dos modelos de classificação
├── requirements.txt              # Ficheiro de dependências do ambiente
├── LICENSE                       # Licença de utilização (MIT)
└── README.md                     # Documentação técnica do projeto

```

---

## 🚀 Como Executar o Projeto

### 1. Clonar o repositório

```bash
git clone [https://github.com/seu-usuario/concrete-strength-ml.git](https://github.com/seu-usuario/concrete-strength-ml.git)
cd concrete-strength-ml

```

### 2. Configurar o ambiente virtual

```bash
# Linux / macOS
python3 -m venv venv
source venv/bin/activate

# Windows
python -m venv venv
venv\Scripts\activate

```

### 3. Instalar as dependências

```bash
pip install -r requirements.txt

```

### 4. Executar os scripts ou abrir o notebook

Para correr a execução em lote via terminal:

```bash
python src/data_pipeline.py

```

Para executar o ambiente interativo com visualização gráfica dos resíduos e matrizes de confusão:

```bash
jupyter notebook notebooks/pipeline_concreto.ipynb

```

---

## 🛠️ Especificações Técnicas de Reprodutibilidade

* **Linguagem:** Python >= 3.10, < 3.13
* **Dependências Principais (`requirements.txt`):**
```text
ucimlrepo>=0.0.6
numpy>=1.24.0
pandas>=2.0.0
scikit-learn>=1.3.0
matplotlib>=3.7.0
seaborn>=0.12.0

```


* **Controlo de Aleatoriedade:** Semente pseudoaleatória fixada em `random_state=42` no particionamento e nos estimadores.


* **Requisitos Mínimos de Hardware:** Processador dual-core (x86-64 ou ARM64), 4 GB de memória RAM e 500 MB de armazenamento livre em disco.

```

```
