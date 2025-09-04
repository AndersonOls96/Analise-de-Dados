# Aula Prática: Análise de Dados com VS Code e Jupyter (Continuação)

## Entendendo o Streamlit: Transformando Análises em Aplicações Web Interativas

Após dominar o NumPy para computação numérica e o Pandas para manipulação de dados, o próximo passo natural em sua jornada de análise de dados é aprender a compartilhar e comunicar seus insights de forma efetiva. O Streamlit é uma biblioteca Python revolucionária que permite transformar scripts de análise de dados em aplicações web interativas e atrativas, sem necessidade de conhecimento em desenvolvimento web tradicional.

### O que é o Streamlit?

O Streamlit é um framework open-source desenvolvido especificamente para cientistas de dados e analistas que desejam criar aplicações web rapidamente. Diferentemente de frameworks web tradicionais que requerem conhecimento em HTML, CSS e JavaScript, o Streamlit permite criar interfaces web completas usando apenas Python. Sua filosofia é simples: "Se você pode escrever um script Python, você pode criar uma aplicação web".

O Streamlit é construído com base no conceito de "data apps" - aplicações focadas na apresentação e interação com dados. Ele se integra perfeitamente com as bibliotecas que você já conhece (NumPy, Pandas, Matplotlib, Plotly) e transforma automaticamente seu código Python em elementos web interativos.

### Para que serve o Streamlit na Análise de Dados?

O Streamlit preenche uma lacuna importante no workflow de análise de dados: a comunicação efetiva dos resultados. Suas principais aplicações incluem:

- **Democratização de Insights**: Permite que pessoas sem conhecimento técnico interajam com suas análises através de interfaces intuitivas, expandindo o alcance de seu trabalho.
- **Prototipagem Rápida**: Facilita a criação de protótipos para testar hipóteses e validar ideias com stakeholders de forma visual e interativa.
- **Dashboards Dinâmicos**: Cria painéis de controle que se atualizam em tempo real, permitindo monitoramento contínuo de métricas e KPIs.
- **Apresentações Interativas**: Transforma análises estáticas em experiências dinâmicas onde o público pode explorar diferentes cenários e parâmetros.
- **Portfólio Profissional**: Constrói um portfólio online de projetos de análise de dados que demonstra suas habilidades de forma tangível.
- **Colaboração em Equipe**: Cria ferramentas internas que permitem que diferentes membros da equipe interajam com os dados sem precisar executar código.

### Vantagens do Streamlit para Analistas de Dados

A integração do Streamlit no workflow de análise de dados oferece benefícios únicos:

- **Simplicidade Extrema**: Widgets complexos são criados com uma única linha de código. Por exemplo, `st.slider()` cria um controle deslizante funcional.
- **Reatividade Automática**: A aplicação se atualiza automaticamente quando parâmetros mudam, criando uma experiência interativa fluida.
- **Integração Nativa**: Funciona perfeitamente com NumPy arrays, DataFrames do Pandas, gráficos Matplotlib/Plotly e modelos de machine learning.
- **Caching Inteligente**: Sistema de cache integrado (`@st.cache_data`) que otimiza performance ao evitar recálculos desnecessários.
- **Deploy Simplificado**: Aplicações podem ser facilmente compartilhadas através da Streamlit Community Cloud ou outras plataformas.
- **Customização Flexível**: Permite personalização de layout, temas e componentes para criar experiências únicas.

Com o Streamlit, você pode transformar um notebook Jupyter cheio de análises em uma aplicação web profissional em questão de minutos, mantendo o foco no que realmente importa: seus dados e insights.

## Configuração do Streamlit no Ambiente de Desenvolvimento

Agora que você já tem o ambiente base configurado com VS Code, Python e as bibliotecas essenciais, vamos adicionar o Streamlit ao seu toolkit. O processo é simples e se integra perfeitamente com a configuração existente.

### 1. Instalação do Streamlit

Com seu ambiente virtual ativado (lembre-se do comando `source .venv/bin/activate`), instale o Streamlit junto com algumas bibliotecas complementares úteis:

```bash
pip install streamlit requests Pillow
```

Após a instalação, você pode verificar se tudo funcionou corretamente executando:

```bash
streamlit --version
```

### 2. Extensões Adicionais do VS Code para Streamlit

Embora não sejam obrigatórias, estas extensões melhoram significativamente a experiência de desenvolvimento com Streamlit:

1. **Python Docstring Generator**: Ajuda na documentação de funções, importante para manter código limpo
2. **autoDocstring**: Gera docstrings automaticamente
3. **Python Type Hint**: Melhora o IntelliSense para type hints

Para instalar, vá em Extensions (`Ctrl+Shift+X`) no VS Code e pesquise pelos nomes acima.

### 3. Estrutura de Projeto Recomendada para Streamlit

Organize seus projetos de análise com Streamlit da seguinte forma:

```
projeto_analise_streamlit/
├── .venv/                 # Ambiente virtual
├── data/                  # Dados do projeto
│   ├── raw/              # Dados brutos
│   ├── processed/        # Dados processados
│   └── external/         # Dados externos
├── notebooks/            # Jupyter notebooks para exploração
├── src/                  # Código fonte
│   ├── data_processing.py    # Funções de processamento
│   ├── visualizations.py    # Funções de visualização
│   └── utils.py             # Utilitários gerais
├── streamlit_app.py      # Aplicação Streamlit principal
├── requirements.txt      # Dependências
├── config.py            # Configurações
└── README.md           # Documentação
```

### 4. Primeira Aplicação Streamlit

Vamos criar sua primeira aplicação para testar a configuração. Crie um arquivo chamado `hello_streamlit.py`:

```python
import streamlit as st
import numpy as np
import pandas as pd

# Configuração da página (deve ser a primeira chamada Streamlit)
st.set_page_config(
    page_title="Minha Primeira App Streamlit",
    page_icon="📊",
    layout="wide"
)

# Título da aplicação
st.title("📊 Olá, Streamlit!")
st.write("Esta é minha primeira aplicação de análise de dados interativa!")

# Verificação das bibliotecas
st.subheader("🔧 Verificação do Ambiente")
st.write(f"Streamlit versão: {st.__version__}")
st.write(f"NumPy versão: {np.__version__}")
st.write(f"Pandas versão: {pd.__version__}")

# Teste simples com dados
st.subheader("📈 Teste com Dados")
data = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100)
})

st.write("Dados aleatórios gerados:")
st.dataframe(data.head())

st.write("Gráfico de dispersão:")
st.scatter_chart(data)

st.success("✅ Ambiente configurado com sucesso!")
```

Para executar a aplicação, use o terminal (com o ambiente virtual ativado):

```bash
streamlit run hello_streamlit.py
```

Se tudo estiver configurado corretamente, uma nova aba do navegador abrirá automaticamente com sua aplicação rodando, geralmente em `http://localhost:8501`.

### 5. Configuração do VS Code para Desenvolvimento Streamlit

Para otimizar o desenvolvimento com Streamlit no VS Code:

1. **Configure tarefas personalizadas**: Crie um arquivo `.vscode/tasks.json` no seu projeto:

```json
{
    "version": "2.0.0",
    "tasks": [
        {
            "label": "Run Streamlit",
            "type": "shell",
            "command": "streamlit",
            "args": ["run", "streamlit_app.py"],
            "group": "build",
            "presentation": {
                "echo": true,
                "reveal": "always",
                "focus": false,
                "panel": "new"
            }
        }
    ]
}
```

2. **Configure snippets para Streamlit**: Crie um arquivo `.vscode/python.json` com snippets úteis:

```json
{
    "Streamlit App Template": {
        "prefix": "st-app",
        "body": [
            "import streamlit as st",
            "import pandas as pd",
            "import numpy as np",
            "",
            "st.set_page_config(",
            "    page_title=\"${1:App Title}\",",
            "    page_icon=\"${2:📊}\",",
            "    layout=\"wide\"",
            ")",
            "",
            "st.title(\"${3:App Title}\")",
            "st.write(\"${4:App description}\")",
            "",
            "${0}"
        ],
        "description": "Template básico para aplicação Streamlit"
    }
}
```

Agora você tem um ambiente completo para desenvolvimento com Streamlit! Na próxima seção, vamos explorar os componentes fundamentais desta ferramenta poderosa.

## Componentes Fundamentais do Streamlit

O Streamlit oferece uma ampla gama de componentes (widgets) que permitem criar interfaces ricas e interativas. Vamos explorar os principais elementos que você utilizará em suas análises de dados.

### 1. Elementos de Texto e Formatação

O Streamlit oferece diversas maneiras de apresentar texto e informações:

```python
import streamlit as st

# Títulos e cabeçalhos
st.title("Título Principal")
st.header("Cabeçalho")
st.subheader("Subcabeçalho")

# Texto comum
st.write("Texto simples usando st.write()")
st.text("Texto sem formatação usando st.text()")

# Markdown
st.markdown("**Texto em negrito** e *itálico*")
st.markdown("## Título Markdown com lista:")
st.markdown("""
- Item 1
- Item 2
- Item 3
""")

# Código
st.code("print('Hello, Streamlit!')", language='python')

# LaTeX
st.latex(r"\sum_{i=1}^{n} x_i^2")
```

### 2. Widgets de Entrada (Input)

Estes componentes permitem que usuários interajam com sua aplicação:

```python
import streamlit as st
import datetime

# Slider numérico
idade = st.slider("Selecione sua idade", 0, 100, 25)

# Input de texto
nome = st.text_input("Digite seu nome")

# Área de texto
comentario = st.text_area("Comentários")

# Selectbox (dropdown)
cidade = st.selectbox("Escolha sua cidade", 
                      ["São Paulo", "Rio de Janeiro", "Belo Horizonte"])

# Multiselect
hobbies = st.multiselect("Selecione seus hobbies",
                         ["Leitura", "Esportes", "Música", "Cinema"])

# Radio buttons
genero = st.radio("Gênero", ["Masculino", "Feminino", "Outro"])

# Checkbox
aceito = st.checkbox("Aceito os termos")

# Date input
data_nascimento = st.date_input("Data de nascimento",
                                datetime.date(1990, 1, 1))

# Number input
salario = st.number_input("Salário", min_value=0.0, step=100.0)

# File uploader
arquivo = st.file_uploader("Carregue um arquivo CSV", type=['csv'])
```

### 3. Exibição de Dados

O Streamlit oferece várias maneiras de exibir dados de forma clara e interativa:

```python
import streamlit as st
import pandas as pd
import numpy as np

# Criando dados de exemplo
df = pd.DataFrame({
    'nome': ['Ana', 'Bruno', 'Carlos', 'Daniela'],
    'idade': [25, 30, 35, 28],
    'salario': [5000, 7000, 8500, 6200],
    'cidade': ['SP', 'RJ', 'BH', 'SP']
})

# Dataframe interativo
st.dataframe(df)

# Tabela estática
st.table(df)

# Métricas
col1, col2, col3 = st.columns(3)
with col1:
    st.metric("Idade Média", f"{df['idade'].mean():.1f}")
with col2:
    st.metric("Salário Médio", f"R$ {df['salario'].mean():,.2f}")
with col3:
    st.metric("Total Funcionários", len(df))

# JSON
st.json({
    'total_funcionarios': len(df),
    'idade_media': df['idade'].mean(),
    'cidades': df['cidade'].unique().tolist()
})
```

### 4. Visualizações

O Streamlit tem suporte nativo para diversos tipos de gráficos:

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px

# Dados de exemplo
df = pd.DataFrame({
    'x': np.random.randn(100),
    'y': np.random.randn(100),
    'categoria': np.random.choice(['A', 'B', 'C'], 100)
})

# Gráficos nativos do Streamlit
st.line_chart(df[['x', 'y']])
st.area_chart(df[['x', 'y']])
st.bar_chart(df.groupby('categoria').size())
st.scatter_chart(df, x='x', y='y')

# Matplotlib
fig, ax = plt.subplots()
ax.hist(df['x'], bins=20)
ax.set_title('Histograma')
st.pyplot(fig)

# Plotly (mais interativo)
fig_plotly = px.scatter(df, x='x', y='y', color='categoria',
                        title='Gráfico de Dispersão Interativo')
st.plotly_chart(fig_plotly, use_container_width=True)

# Mapa (se você tiver dados de latitude/longitude)
map_data = pd.DataFrame({
    'lat': np.random.uniform(-23.6, -23.5, 20),
    'lon': np.random.uniform(-46.7, -46.6, 20)
})
st.map(map_data)
```

### 5. Layout e Organização

Organize sua aplicação com diferentes layouts:

```python
import streamlit as st

# Sidebar
st.sidebar.title("Controles")
filtro = st.sidebar.selectbox("Filtro", ["Todos", "Categoria A", "Categoria B"])

# Colunas
col1, col2, col3 = st.columns(3)
with col1:
    st.write("Coluna 1")
with col2:
    st.write("Coluna 2")
with col3:
    st.write("Coluna 3")

# Colunas com proporções diferentes
col1, col2 = st.columns([2, 1])  # col1 é 2x maior que col2
with col1:
    st.write("Coluna principal")
with col2:
    st.write("Sidebar secundária")

# Expander (seção colapsável)
with st.expander("Clique para expandir"):
    st.write("Conteúdo oculto que aparece quando expandido")

# Container
container = st.container()
container.write("Este é um container")

# Placeholder (para conteúdo dinâmico)
placeholder = st.empty()
placeholder.text("Texto que pode ser substituído")
```

### 6. Elementos de Feedback

Forneça feedback visual para os usuários:

```python
import streamlit as st
import time

# Mensagens de status
st.success("Operação concluída com sucesso!")
st.info("Informação importante")
st.warning("Atenção: verifique os dados")
st.error("Erro: algo deu errado")

# Exception
try:
    result = 1 / 0
except Exception as e:
    st.exception(e)

# Progress bar
progress_bar = st.progress(0)
for i in range(100):
    time.sleep(0.01)
    progress_bar.progress(i + 1)

# Spinner
with st.spinner("Carregando..."):
    time.sleep(2)
st.success("Carregamento concluído!")

# Balloons (celebração)
if st.button("Clique para comemorar!"):
    st.balloons()
```

Estes são os componentes fundamentais que você usará para construir aplicações de análise de dados interativas. Na próxima seção, vamos ver exemplos práticos de como combiná-los em aplicações reais.

## Exemplos Básicos de Uso do Streamlit

Agora que você conhece os componentes fundamentais, vamos explorar exemplos práticos que demonstram como usar o Streamlit para criar aplicações de análise de dados. Cada exemplo constrói sobre o anterior, aumentando gradualmente a complexidade.

### 1. Calculadora de Estatísticas Básicas

Este exemplo mostra como criar uma calculadora interativa para estatísticas descritivas:

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Configuração da página
st.set_page_config(
    page_title="Calculadora de Estatísticas",
    page_icon="🧮",
    layout="wide"
)

st.title("🧮 Calculadora de Estatísticas Descritivas")
st.write("Insira números separados por vírgula ou carregue um arquivo CSV")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Opção de entrada de dados
opcao_entrada = st.sidebar.radio(
    "Como você quer inserir os dados?",
    ["Digitar números", "Carregar arquivo CSV"]
)

# Inicializar dados
dados = None

if opcao_entrada == "Digitar números":
    # Input manual de números
    numeros_texto = st.text_area(
        "Digite os números separados por vírgula:",
        value="1, 2, 3, 4, 5, 6, 7, 8, 9, 10",
        height=100
    )
    
    try:
        # Converter texto em lista de números
        numeros = [float(x.strip()) for x in numeros_texto.split(',') if x.strip()]
        dados = pd.Series(numeros, name='valores')
        st.success(f"✅ {len(numeros)} números carregados com sucesso!")
    except ValueError:
        st.error("❌ Erro: Certifique-se de inserir apenas números separados por vírgula")

else:
    # Upload de arquivo
    arquivo = st.file_uploader(
        "Carregue um arquivo CSV",
        type=['csv'],
        help="O arquivo deve ter uma coluna numérica"
    )
    
    if arquivo is not None:
        try:
            df = pd.read_csv(arquivo)
            st.write("📄 Dados carregados:")
            st.dataframe(df.head())
            
            # Seleção da coluna
            coluna_numerica = st.selectbox(
                "Selecione a coluna para análise:",
                df.select_dtypes(include=[np.number]).columns
            )
            
            dados = df[coluna_numerica].dropna()
            st.success(f"✅ {len(dados)} valores válidos encontrados na coluna '{coluna_numerica}'")
            
        except Exception as e:
            st.error(f"❌ Erro ao carregar arquivo: {str(e)}")

# Análise dos dados
if dados is not None and len(dados) > 0:
    st.header("📊 Resultados da Análise")
    
    # Estatísticas em colunas
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric("📏 Média", f"{dados.mean():.2f}")
        st.metric("📊 Mediana", f"{dados.median():.2f}")
    
    with col2:
        st.metric("📈 Máximo", f"{dados.max():.2f}")
        st.metric("📉 Mínimo", f"{dados.min():.2f}")
    
    with col3:
        st.metric("📐 Desvio Padrão", f"{dados.std():.2f}")
        st.metric("🎯 Variância", f"{dados.var():.2f}")
    
    with col4:
        st.metric("📏 Amplitude", f"{dados.max() - dados.min():.2f}")
        st.metric("📊 Tamanho", f"{len(dados)}")
    
    # Tabela de estatísticas completa
    st.subheader("📋 Estatísticas Detalhadas")
    stats_df = pd.DataFrame({
        'Estatística': ['Contagem', 'Média', 'Desvio Padrão', 'Mínimo', 
                       '25% (Q1)', '50% (Mediana)', '75% (Q3)', 'Máximo'],
        'Valor': [dados.count(), dados.mean(), dados.std(), dados.min(),
                 dados.quantile(0.25), dados.median(), dados.quantile(0.75), dados.max()]
    })
    stats_df['Valor'] = stats_df['Valor'].round(3)
    st.table(stats_df)
    
    # Visualizações
    st.subheader("📈 Visualizações")
    
    # Escolha do tipo de gráfico
    tipo_grafico = st.selectbox(
        "Selecione o tipo de visualização:",
        ["Histograma", "Box Plot", "Gráfico de Linha", "Gráfico de Barras"]
    )
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    if tipo_grafico == "Histograma":
        bins = st.slider("Número de bins:", 5, 50, 20)
        ax.hist(dados, bins=bins, edgecolor='black', alpha=0.7)
        ax.set_title("Histograma dos Dados")
        ax.set_xlabel("Valores")
        ax.set_ylabel("Frequência")
        
    elif tipo_grafico == "Box Plot":
        ax.boxplot(dados)
        ax.set_title("Box Plot dos Dados")
        ax.set_ylabel("Valores")
        
    elif tipo_grafico == "Gráfico de Linha":
        ax.plot(dados.values, marker='o')
        ax.set_title("Gráfico de Linha dos Dados")
        ax.set_xlabel("Índice")
        ax.set_ylabel("Valores")
        
    else:  # Gráfico de Barras
        if len(dados) <= 50:  # Só mostrar barras se não há muitos dados
            ax.bar(range(len(dados)), dados.values)
            ax.set_title("Gráfico de Barras dos Dados")
            ax.set_xlabel("Índice")
            ax.set_ylabel("Valores")
        else:
            st.warning("⚠️ Muitos dados para gráfico de barras. Escolha outro tipo.")
            ax.hist(dados, bins=20, edgecolor='black', alpha=0.7)
            ax.set_title("Histograma dos Dados (alternativo)")
    
    st.pyplot(fig)
    
    # Informações adicionais
    with st.expander("ℹ️ Informações Adicionais"):
        st.write("**Interpretação das Estatísticas:**")
        st.write("- **Média**: Valor médio dos dados")
        st.write("- **Mediana**: Valor central quando os dados estão ordenados")
        st.write("- **Desvio Padrão**: Medida de dispersão dos dados")
        st.write("- **Quartis**: Dividem os dados em quatro partes iguais")
        
        if dados.std() > dados.mean():
            st.info("📊 Os dados apresentam alta variabilidade (desvio padrão > média)")
        else:
            st.info("📊 Os dados apresentam baixa variabilidade (desvio padrão ≤ média)")

else:
    st.info("👆 Insira alguns dados acima para começar a análise!")

# Footer
st.markdown("---")
st.markdown("📚 **Desenvolvido com Streamlit** | 🐍 Python para Análise de Dados")
```

### 2. Explorador de Dataset Interativo

Este exemplo cria uma aplicação para explorar datasets carregados pelo usuário:

```python
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px

# Configuração da página
st.set_page_config(
    page_title="Explorador de Dataset",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Explorador de Dataset Interativo")
st.write("Carregue um arquivo CSV e explore seus dados de forma interativa!")

# Cache para carregar dados
@st.cache_data
def load_data(file):
    return pd.read_csv(file)

# Upload do arquivo
uploaded_file = st.file_uploader(
    "Escolha um arquivo CSV",
    type=['csv'],
    help="Carregue um arquivo CSV para começar a exploração"
)

if uploaded_file is not None:
    # Carregar dados
    df = load_data(uploaded_file)
    
    # Informações básicas sobre o dataset
    st.header("📊 Visão Geral do Dataset")
    
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📏 Linhas", df.shape[0])
    with col2:
        st.metric("📊 Colunas", df.shape[1])
    with col3:
        st.metric("💾 Tamanho", f"{df.memory_usage(deep=True).sum() / 1024:.1f} KB")
    with col4:
        st.metric("❓ Valores Faltantes", df.isnull().sum().sum())
    
    # Sidebar para filtros
    st.sidebar.header("🎛️ Filtros e Configurações")
    
    # Seleção de colunas para exibir
    colunas_selecionadas = st.sidebar.multiselect(
        "Selecione as colunas para exibir:",
        df.columns.tolist(),
        default=df.columns.tolist()[:5]  # Primeiras 5 colunas por padrão
    )
    
    # Número de linhas para exibir
    num_linhas = st.sidebar.slider(
        "Número de linhas para exibir:",
        5, min(100, len(df)), 10
    )
    
    # Mostrar dados filtrados
    if colunas_selecionadas:
        st.subheader("🗂️ Dados do Dataset")
        st.dataframe(df[colunas_selecionadas].head(num_linhas))
        
        # Informações sobre os tipos de dados
        st.subheader("🏷️ Tipos de Dados")
        tipos_df = pd.DataFrame({
            'Coluna': df[colunas_selecionadas].columns,
            'Tipo': df[colunas_selecionadas].dtypes.astype(str),
            'Valores Únicos': df[colunas_selecionadas].nunique(),
            'Valores Faltantes': df[colunas_selecionadas].isnull().sum(),
            '% Faltantes': (df[colunas_selecionadas].isnull().sum() / len(df) * 100).round(2)
        })
        st.dataframe(tipos_df)
        
        # Análise por tipo de dados
        colunas_numericas = df[colunas_selecionadas].select_dtypes(include=[np.number]).columns.tolist()
        colunas_categoricas = df[colunas_selecionadas].select_dtypes(include=['object']).columns.tolist()
        
        if colunas_numericas:
            st.subheader("📈 Análise de Variáveis Numéricas")
            
            # Seleção da variável numérica
            var_numerica = st.selectbox("Selecione uma variável numérica:", colunas_numericas)
            
            if var_numerica:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Estatísticas descritivas
                    st.write("**Estatísticas Descritivas:**")
                    stats = df[var_numerica].describe()
                    st.dataframe(stats.to_frame())
                    
                with col2:
                    # Histograma
                    fig, ax = plt.subplots()
                    ax.hist(df[var_numerica].dropna(), bins=30, edgecolor='black', alpha=0.7)
                    ax.set_title(f'Histograma de {var_numerica}')
                    ax.set_xlabel(var_numerica)
                    ax.set_ylabel('Frequência')
                    st.pyplot(fig)
        
        if colunas_categoricas:
            st.subheader("🏷️ Análise de Variáveis Categóricas")
            
            # Seleção da variável categórica
            var_categorica = st.selectbox("Selecione uma variável categórica:", colunas_categoricas)
            
            if var_categorica:
                col1, col2 = st.columns(2)
                
                with col1:
                    # Contagem de valores
                    contagem = df[var_categorica].value_counts()
                    st.write("**Contagem de Valores:**")
                    st.dataframe(contagem.to_frame())
                
                with col2:
                    # Gráfico de barras
                    fig, ax = plt.subplots()
                    contagem.plot(kind='bar', ax=ax)
                    ax.set_title(f'Distribuição de {var_categorica}')
                    ax.set_xlabel(var_categorica)
                    ax.set_ylabel('Contagem')
                    plt.xticks(rotation=45)
                    st.pyplot(fig)
        
        # Análise de correlação (apenas para variáveis numéricas)
        if len(colunas_numericas) > 1:
            st.subheader("🔗 Análise de Correlação")
            
            # Matriz de correlação
            corr_matrix = df[colunas_numericas].corr()
            
            # Heatmap
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', center=0, ax=ax)
            ax.set_title('Matriz de Correlação')
            st.pyplot(fig)
            
            # Scatter plot interativo
            if len(colunas_numericas) >= 2:
                st.subheader("📊 Gráfico de Dispersão Interativo")
                
                col1, col2 = st.columns(2)
                with col1:
                    x_var = st.selectbox("Variável X:", colunas_numericas, key='x_var')
                with col2:
                    y_var = st.selectbox("Variável Y:", 
                                       [col for col in colunas_numericas if col != x_var], 
                                       key='y_var')
                
                if x_var and y_var:
                    # Opção de colorir por variável categórica
                    color_var = None
                    if colunas_categoricas:
                        usar_cor = st.checkbox("Colorir por variável categórica")
                        if usar_cor:
                            color_var = st.selectbox("Variável para cor:", colunas_categoricas)
                    
                    # Criar gráfico com Plotly
                    fig = px.scatter(
                        df, 
                        x=x_var, 
                        y=y_var, 
                        color=color_var,
                        title=f'{y_var} vs {x_var}',
                        opacity=0.7
                    )
                    st.plotly_chart(fig, use_container_width=True)
        
        # Resumo executivo
        with st.expander("📋 Resumo Executivo"):
            st.write("**Características do Dataset:**")
            st.write(f"- **Tamanho**: {df.shape[0]:,} linhas e {df.shape[1]} colunas")
            st.write(f"- **Variáveis Numéricas**: {len(colunas_numericas)}")
            st.write(f"- **Variáveis Categóricas**: {len(colunas_categoricas)}")
            st.write(f"- **Valores Faltantes**: {df.isnull().sum().sum():,} ({df.isnull().sum().sum()/df.size*100:.1f}%)")
            
            if colunas_numericas:
                st.write("**Insights das Variáveis Numéricas:**")
                for col in colunas_numericas[:3]:  # Primeiras 3 colunas
                    media = df[col].mean()
                    std = df[col].std()
                    st.write(f"- {col}: Média = {media:.2f}, Desvio Padrão = {std:.2f}")
    
    else:
        st.warning("⚠️ Selecione pelo menos uma coluna para exibir os dados.")

else:
    st.info("📁 Carregue um arquivo CSV para começar a exploração!")
    
    # Exemplo de dados para demonstração
    st.subheader("💡 Exemplo com Dados de Demonstração")
    if st.button("Carregar Dados de Exemplo"):
        # Criar dados de exemplo
        np.random.seed(42)
        exemplo_df = pd.DataFrame({
            'idade': np.random.randint(18, 80, 100),
            'salario': np.random.normal(50000, 15000, 100),
            'experiencia': np.random.randint(0, 40, 100),
            'departamento': np.random.choice(['TI', 'Vendas', 'Marketing', 'RH'], 100),
            'genero': np.random.choice(['M', 'F'], 100)
        })
        
        st.write("Dados de exemplo carregados:")
        st.dataframe(exemplo_df.head())
        st.info("💡 Use o uploader acima para carregar seus próprios dados!")

# Footer
st.markdown("---")
st.markdown("🔧 **Ferramenta desenvolvida com Streamlit** | 📊 Análise de Dados Interativa")
```

### 3. Comparador de Distribuições

Este exemplo avançado permite comparar distribuições estatísticas:

```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import plotly.graph_objects as go
from plotly.subplots import make_subplots

# Configuração da página
st.set_page_config(
    page_title="Comparador de Distribuições",
    page_icon="📊",
    layout="wide"
)

st.title("📊 Comparador de Distribuições Estatísticas")
st.write("Compare diferentes distribuições estatísticas e seus parâmetros interativamente!")

# Sidebar para configurações
st.sidebar.header("⚙️ Configurações")

# Número de amostras
n_samples = st.sidebar.slider("Número de amostras:", 100, 10000, 1000, step=100)

# Configuração das distribuições
st.sidebar.subheader("📈 Distribuição 1")
dist1_type = st.sidebar.selectbox(
    "Tipo de distribuição 1:",
    ["Normal", "Exponencial", "Uniforme", "Binomial", "Poisson"]
)

# Parâmetros para distribuição 1
if dist1_type == "Normal":
    mu1 = st.sidebar.slider("Média (μ):", -10.0, 10.0, 0.0, step=0.1, key="mu1")
    sigma1 = st.sidebar.slider("Desvio padrão (σ):", 0.1, 5.0, 1.0, step=0.1, key="sigma1")
    dist1_data = np.random.normal(mu1, sigma1, n_samples)
    dist1_params = f"μ={mu1}, σ={sigma1}"

elif dist1_type == "Exponencial":
    lambda1 = st.sidebar.slider("Lambda (λ):", 0.1, 5.0, 1.0, step=0.1, key="lambda1")
    dist1_data = np.random.exponential(1/lambda1, n_samples)
    dist1_params = f"λ={lambda1}"

elif dist1_type == "Uniforme":
    a1 = st.sidebar.slider("Limite inferior (a):", -10.0, 10.0, 0.0, key="a1")
    b1 = st.sidebar.slider("Limite superior (b):", a1, 20.0, 10.0, key="b1")
    dist1_data = np.random.uniform(a1, b1, n_samples)
    dist1_params = f"a={a1}, b={b1}"

elif dist1_type == "Binomial":
    n1 = st.sidebar.slider("Número de tentativas (n):", 1, 100, 10, key="n1")
    p1 = st.sidebar.slider("Probabilidade (p):", 0.01, 1.0, 0.5, step=0.01, key="p1")
    dist1_data = np.random.binomial(n1, p1, n_samples)
    dist1_params = f"n={n1}, p={p1}"

else:  # Poisson
    lam1 = st.sidebar.slider("Lambda (λ):", 0.1, 20.0, 3.0, step=0.1, key="lam1")
    dist1_data = np.random.poisson(lam1, n_samples)
    dist1_params = f"λ={lam1}"

# Configuração da segunda distribuição
st.sidebar.subheader("📉 Distribuição 2")
comparar = st.sidebar.checkbox("Comparar com segunda distribuição")

if comparar:
    dist2_type = st.sidebar.selectbox(
        "Tipo de distribuição 2:",
        ["Normal", "Exponencial", "Uniforme", "Binomial", "Poisson"]
    )
    
    # Parâmetros para distribuição 2
    if dist2_type == "Normal":
        mu2 = st.sidebar.slider("Média (μ):", -10.0, 10.0, 2.0, step=0.1, key="mu2")
        sigma2 = st.sidebar.slider("Desvio padrão (σ):", 0.1, 5.0, 1.5, step=0.1, key="sigma2")
        dist2_data = np.random.normal(mu2, sigma2, n_samples)
        dist2_params = f"μ={mu2}, σ={sigma2}"

    elif dist2_type == "Exponencial":
        lambda2 = st.sidebar.slider("Lambda (λ):", 0.1, 5.0, 2.0, step=0.1, key="lambda2")
        dist2_data = np.random.exponential(1/lambda2, n_samples)
        dist2_params = f"λ={lambda2}"

    elif dist2_type == "Uniforme":
        a2 = st.sidebar.slider("Limite inferior (a):", -10.0, 10.0, -5.0, key="a2")
        b2 = st.sidebar.slider("Limite superior (b):", a2, 20.0, 5.0, key="b2")
        dist2_data = np.random.uniform(a2, b2, n_samples)
        dist2_params = f"a={a2}, b={b2}"

    elif dist2_type == "Binomial":
        n2 = st.sidebar.slider("Número de tentativas (n):", 1, 100, 20, key="n2")
        p2 = st.sidebar.slider("Probabilidade (p):", 0.01, 1.0, 0.3, step=0.01, key="p2")
        dist2_data = np.random.binomial(n2, p2, n_samples)
        dist2_params = f"n={n2}, p={p2}"

    else:  # Poisson
        lam2 = st.sidebar.slider("Lambda (λ):", 0.1, 20.0, 5.0, step=0.1, key="lam2")
        dist2_data = np.random.poisson(lam2, n_samples)
        dist2_params = f"λ={lam2}"

# Conteúdo principal
if not comparar:
    # Análise de uma distribuição
    st.header(f"📊 Análise da Distribuição {dist1_type}")
    
    # Estatísticas descritivas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("📏 Média", f"{np.mean(dist1_data):.3f}")
    with col2:
        st.metric("📊 Mediana", f"{np.median(dist1_data):.3f}")
    with col3:
        st.metric("📐 Desvio Padrão", f"{np.std(dist1_data):.3f}")
    with col4:
        st.metric("📈 Assimetria", f"{stats.skew(dist1_data):.3f}")
    
    # Visualizações
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Histograma', 'Box Plot', 'Q-Q Plot', 'Função de Densidade'],
        specs=[[{"secondary_y": False}, {"secondary_y": False}],
               [{"secondary_y": False}, {"secondary_y": False}]]
    )
    
    # Histograma
    fig.add_trace(
        go.Histogram(x=dist1_data, nbinsx=50, name="Histograma", 
                    histnorm='probability density'),
        row=1, col=1
    )
    
    # Box Plot
    fig.add_trace(
        go.Box(y=dist1_data, name="Box Plot"),
        row=1, col=2
    )
    
    # Q-Q Plot
    theoretical_quantiles = stats.norm.ppf(np.linspace(0.01, 0.99, len(dist1_data)))
    sample_quantiles = np.sort(dist1_data)
    fig.add_trace(
        go.Scatter(x=theoretical_quantiles, y=sample_quantiles, 
                  mode='markers', name="Q-Q Plot"),
        row=2, col=1
    )
    
    # Linha de referência para Q-Q plot
    min_val = min(theoretical_quantiles.min(), sample_quantiles.min())
    max_val = max(theoretical_quantiles.max(), sample_quantiles.max())
    fig.add_trace(
        go.Scatter(x=[min_val, max_val], y=[min_val, max_val], 
                  mode='lines', name="Linha de Referência", line=dict(dash='dash')),
        row=2, col=1
    )
    
    # Função de densidade (KDE)
    from scipy.stats import gaussian_kde
    kde = gaussian_kde(dist1_data)
    x_range = np.linspace(dist1_data.min(), dist1_data.max(), 100)
    density = kde(x_range)
    
    fig.add_trace(
        go.Scatter(x=x_range, y=density, mode='lines', name="KDE"),
        row=2, col=2
    )
    
    fig.update_layout(height=700, showlegend=False, 
                     title_text=f"Análise da Distribuição {dist1_type} ({dist1_params})")
    st.plotly_chart(fig, use_container_width=True)

else:
    # Comparação de duas distribuições
    st.header(f"📊 Comparação: {dist1_type} vs {dist2_type}")
    
    # Estatísticas comparativas
    col1, col2 = st.columns(2)
    
    with col1:
        st.subheader(f"📈 {dist1_type} ({dist1_params})")
        stats1_df = pd.DataFrame({
            'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Variância', 'Assimetria', 'Curtose'],
            'Valor': [
                np.mean(dist1_data),
                np.median(dist1_data),
                np.std(dist1_data),
                np.var(dist1_data),
                stats.skew(dist1_data),
                stats.kurtosis(dist1_data)
            ]
        })
        stats1_df['Valor'] = stats1_df['Valor'].round(4)
        st.dataframe(stats1_df)
    
    with col2:
        st.subheader(f"📉 {dist2_type} ({dist2_params})")
        stats2_df = pd.DataFrame({
            'Estatística': ['Média', 'Mediana', 'Desvio Padrão', 'Variância', 'Assimetria', 'Curtose'],
            'Valor': [
                np.mean(dist2_data),
                np.median(dist2_data),
                np.std(dist2_data),
                np.var(dist2_data),
                stats.skew(dist2_data),
                stats.kurtosis(dist2_data)
            ]
        })
        stats2_df['Valor'] = stats2_df['Valor'].round(4)
        st.dataframe(stats2_df)
    
    # Testes estatísticos
    st.subheader("🧪 Testes Estatísticos")
    
    # Teste de normalidade
    _, p_norm1 = stats.normaltest(dist1_data)
    _, p_norm2 = stats.normaltest(dist2_data)
    
    # Teste t de Student (se ambas forem aproximadamente normais)
    if p_norm1 > 0.05 and p_norm2 > 0.05:
        t_stat, p_ttest = stats.ttest_ind(dist1_data, dist2_data)
        st.write(f"**Teste t de Student**: t = {t_stat:.4f}, p-value = {p_ttest:.4f}")
        if p_ttest < 0.05:
            st.success("✅ Há diferença significativa entre as médias (p < 0.05)")
        else:
            st.info("📊 Não há diferença significativa entre as médias (p ≥ 0.05)")
    
    # Teste de Mann-Whitney (não-paramétrico)
    u_stat, p_mann = stats.mannwhitneyu(dist1_data, dist2_data, alternative='two-sided')
    st.write(f"**Teste de Mann-Whitney**: U = {u_stat:.4f}, p-value = {p_mann:.4f}")
    if p_mann < 0.05:
        st.success("✅ Há diferença significativa entre as distribuições (p < 0.05)")
    else:
        st.info("📊 Não há diferença significativa entre as distribuições (p ≥ 0.05)")
    
    # Teste de Kolmogorov-Smirnov
    ks_stat, p_ks = stats.ks_2samp(dist1_data, dist2_data)
    st.write(f"**Teste de Kolmogorov-Smirnov**: D = {ks_stat:.4f}, p-value = {p_ks:.4f}")
    
    # Visualizações comparativas
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=['Histogramas Sobrepostos', 'Box Plots Comparativos', 
                       'Funções de Densidade', 'Funções de Distribuição Cumulativa'],
    )
    
    # Histogramas sobrepostos
    fig.add_trace(
        go.Histogram(x=dist1_data, nbinsx=50, name=f"{dist1_type}", 
                    opacity=0.7, histnorm='probability density'),
        row=1, col=1
    )
    fig.add_trace(
        go.Histogram(x=dist2_data, nbinsx=50, name=f"{dist2_type}", 
                    opacity=0.7, histnorm='probability density'),
        row=1, col=1
    )
    
    # Box plots
    fig.add_trace(
        go.Box(y=dist1_data, name=f"{dist1_type}", x0=f"{dist1_type}"),
        row=1, col=2
    )
    fig.add_trace(
        go.Box(y=dist2_data, name=f"{dist2_type}", x0=f"{dist2_type}"),
        row=1, col=2
    )
    
    # Funções de densidade
    kde1 = gaussian_kde(dist1_data)
    kde2 = gaussian_kde(dist2_data)
    x_min = min(dist1_data.min(), dist2_data.min())
    x_max = max(dist1_data.max(), dist2_data.max())
    x_range = np.linspace(x_min, x_max, 200)
    
    fig.add_trace(
        go.Scatter(x=x_range, y=kde1(x_range), mode='lines', name=f"KDE {dist1_type}"),
        row=2, col=1
    )
    fig.add_trace(
        go.Scatter(x=x_range, y=kde2(x_range), mode='lines', name=f"KDE {dist2_type}"),
        row=2, col=1
    )
    
    # Funções de distribuição cumulativa
    x1_sorted = np.sort(dist1_data)
    y1_cdf = np.arange(1, len(x1_sorted) + 1) / len(x1_sorted)
    x2_sorted = np.sort(dist2_data)
    y2_cdf = np.arange(1, len(x2_sorted) + 1) / len(x2_sorted)
    
    fig.add_trace(
        go.Scatter(x=x1_sorted, y=y1_cdf, mode='lines', name=f"CDF {dist1_type}"),
        row=2, col=2
    )
    fig.add_trace(
        go.Scatter(x=x2_sorted, y=y2_cdf, mode='lines', name=f"CDF {dist2_type}"),
        row=2, col=2
    )
    
    fig.update_layout(height=800, title_text="Comparação de Distribuições")
    st.plotly_chart(fig, use_container_width=True)

# Seção educativa
with st.expander("📚 Sobre as Distribuições"):
    st.write("""
    **Normal**: Distribuição simétrica em forma de sino, muito comum na natureza.
    
    **Exponencial**: Modela tempo entre eventos em processos de Poisson.
    
    **Uniforme**: Todos os valores no intervalo têm a mesma probabilidade.
    
    **Binomial**: Número de sucessos em n tentativas independentes.
    
    **Poisson**: Número de eventos em um intervalo fixo de tempo ou espaço.
    """)

with st.expander("🧪 Sobre os Testes Estatísticos"):
    st.write("""
    **Teste t de Student**: Compara médias de duas amostras (assume normalidade).
    
    **Teste de Mann-Whitney**: Versão não-paramétrica do teste t.
    
    **Teste de Kolmogorov-Smirnov**: Compara as distribuições completas.
    
    **p-value < 0.05**: Evidência estatística de diferença significativa.
    """)

# Footer
st.markdown("---")
st.markdown("📊 **Desenvolvido com Streamlit** | 🔬 Análise Estatística Interativa")
```

Estes exemplos demonstram o poder do Streamlit para criar aplicações de análise de dados sofisticadas e interativas. Na próxima seção, vamos criar exercícios práticos para consolidar o aprendizado.
