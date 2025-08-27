## Visualização de Dados com Matplotlib e Seaborn

A visualização de dados é uma etapa crucial na análise de dados, pois permite explorar padrões, identificar anomalias e comunicar insights de forma eficaz. Python oferece bibliotecas poderosas para essa finalidade, sendo as mais populares Matplotlib e Seaborn.

### Matplotlib: A Base da Visualização em Python

Matplotlib é uma biblioteca de plotagem 2D para Python que produz figuras de qualidade de publicação em uma variedade de formatos de impressão e ambientes interativos. É a base para muitas outras bibliotecas de visualização em Python, incluindo o Seaborn. Com Matplotlib, você tem controle granular sobre cada elemento do seu gráfico, desde os rótulos dos eixos até as cores e estilos das linhas.

**Principais tipos de gráficos com Matplotlib:**

-   **Gráficos de Linha (Line Plots)**: Ideais para mostrar tendências ao longo do tempo ou de uma variável contínua.
-   **Gráficos de Dispersão (Scatter Plots)**: Usados para visualizar a relação entre duas variáveis numéricas.
-   **Histogramas (Histograms)**: Mostram a distribuição de uma única variável numérica.
-   **Gráficos de Barras (Bar Plots)**: Comparar quantidades entre diferentes categorias.
-   **Gráficos de Pizza (Pie Charts)**: Representar proporções de um todo (usar com cautela, pois podem ser difíceis de interpretar para muitas categorias).

### Seaborn: Visualizações Estatísticas Aprimoradas

Seaborn é uma biblioteca de visualização de dados Python baseada em Matplotlib que fornece uma interface de alto nível para desenhar gráficos estatísticos atraentes e informativos. Ele se integra muito bem com DataFrames do Pandas e é projetado para trabalhar com dados estruturados, facilitando a criação de visualizações complexas com menos código. Seaborn é particularmente útil para explorar relações entre múltiplas variáveis e para visualizar distribuições estatísticas.

**Vantagens do Seaborn:**

-   **Estética Aprimorada**: Gráficos mais bonitos e profissionais por padrão.
-   **Facilidade de Uso**: Sintaxe mais simples para gráficos estatísticos complexos.
-   **Integração com Pandas**: Funciona nativamente com DataFrames, permitindo plotar diretamente a partir de colunas.
-   **Gráficos Estatísticos Avançados**: Oferece gráficos especializados para análise estatística, como mapas de calor, gráficos de violino, gráficos de pares, etc.

Ao combinar Matplotlib e Seaborn, você pode criar uma ampla gama de visualizações, desde gráficos simples e personalizados até visualizações estatísticas complexas e esteticamente agradáveis. A escolha entre um e outro (ou a combinação de ambos) depende da complexidade da visualização e do nível de controle que você precisa sobre os detalhes do gráfico.

### Exemplos Práticos de Visualização de Dados

Vamos ver alguns exemplos de como criar gráficos usando Matplotlib e Seaborn.

#### Exemplo 1: Gráfico de Linha (Matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

# Dados de exemplo
x = np.linspace(0, 10, 100)
y = np.sin(x)

# Criando o gráfico de linha
plt.figure(figsize=(8, 4))
plt.plot(x, y, label=\'Função Seno\', color=\'blue\')
plt.title(\'Gráfico de Seno\')
plt.xlabel(\'Eixo X\')
plt.ylabel(\'Eixo Y\')
plt.grid(True)
plt.legend()
plt.show()
```

#### Exemplo 2: Gráfico de Barras (Matplotlib)

```python
import matplotlib.pyplot as plt

# Dados de exemplo
categorias = [\'A\', \'B\', \'C\', \'D\']
valores = [23, 45, 56, 12]

# Criando o gráfico de barras
plt.figure(figsize=(7, 5))
plt.bar(categorias, valores, color=\'skyblue\')
plt.title(\'Vendas por Categoria\')
plt.xlabel(\'Categoria\')
plt.ylabel(\'Vendas\')
plt.show()
```

#### Exemplo 3: Histograma (Matplotlib)

```python
import matplotlib.pyplot as plt
import numpy as np

# Dados de exemplo (distribuição normal)
dados = np.random.randn(1000)

# Criando o histograma
plt.figure(figsize=(7, 5))
plt.hist(dados, bins=30, color=\'lightgreen\', edgecolor=\'black\')
plt.title(\'Distribuição de Dados Aleatórios\')
plt.xlabel(\'Valor\')
plt.ylabel(\'Frequência\')
plt.show()
```

#### Exemplo 4: Gráfico de Dispersão (Seaborn)

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Dados de exemplo
df_scatter = pd.DataFrame({
    \'Tamanho_Casa\': [100, 120, 150, 130, 180, 200, 90, 110, 140, 160],
    \'Preco\': [250000, 280000, 350000, 300000, 400000, 450000, 220000, 260000, 330000, 380000]
})

# Criando o gráfico de dispersão com Seaborn
plt.figure(figsize=(8, 6))
sns.scatterplot(x=\'Tamanho_Casa\', y=\'Preco\', data=df_scatter, hue=\'Tamanho_Casa\', size=\'Preco\', sizes=(50, 200), palette=\'viridis\')
plt.title(\'Preço da Casa vs. Tamanho da Casa\')
plt.xlabel(\'Tamanho da Casa (m²)\' )
plt.ylabel(\'Preço (R$)\' )
plt.grid(True, linestyle=\'--\', alpha=0.7)
plt.show()
```

#### Exemplo 5: Box Plot (Seaborn)

```python
import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

# Dados de exemplo
df_boxplot = pd.DataFrame({
    \'Grupo\': [\'A\', \'A\', \'A\', \'B\', \'B\', \'B\', \'C\', \'C\', \'C\'],
    \'Valores\': [10, 12, 11, 15, 18, 16, 20, 22, 21]
})

# Criando o box plot com Seaborn
plt.figure(figsize=(7, 5))
sns.boxplot(x=\'Grupo\', y=\'Valores\', data=df_boxplot, palette=\'pastel\')
plt.title(\'Distribuição de Valores por Grupo\')
plt.xlabel(\'Grupo\')
plt.ylabel(\'Valores\')
plt.show()
```

Esses exemplos demonstram a facilidade e o poder das bibliotecas Matplotlib e Seaborn para criar visualizações de dados informativas e atraentes. A escolha do tipo de gráfico dependerá do tipo de dados e da mensagem que você deseja transmitir. Experimente diferentes tipos de gráficos e personalize-os para atender às suas necessidades de análise.

## Exercícios Sugeridos: Visualização de Dados

Para praticar a visualização de dados, tente criar os seguintes gráficos usando Matplotlib e/ou Seaborn.

### Exercício 7: Gráficos Básicos com Matplotlib

Considere os seguintes dados de temperatura média diária em uma semana:

```python
dias = [\"Seg\", \"Ter\", \"Qua\", \"Qui\", \"Sex\", \"Sáb\", \"Dom\"]
temperaturas = [22, 24, 23, 25, 27, 26, 24]
```

1.  Crie um gráfico de linha mostrando a variação da temperatura ao longo da semana.
2.  Crie um gráfico de barras comparando a temperatura de cada dia.

### Exercício 8: Gráficos com Seaborn

Considere o seguinte DataFrame de dados de estudantes:

```python
import pandas as pd

dados_estudantes = pd.DataFrame({
    \"Materia\": [\"Matemática\", \"Matemática\", \"Ciências\", \"Ciências\", \"Matemática\", \"Ciências\"],
    \"Nota\": [75, 88, 92, 70, 80, 85],
    \"Genero\": [\"F\", \"M\", \"F\", \"M\", \"F\", \"M\"]
})
```

1.  Crie um box plot para visualizar a distribuição das notas por matéria.
2.  Crie um gráfico de barras que mostre a média das notas por gênero.

### Exercício 9: Combinando Matplotlib e Seaborn

Utilize o DataFrame `df_vendas` do Exercício 5 (Leitura e Análise de Dados CSV).

1.  Crie um gráfico de linha mostrando as vendas ao longo dos meses.
2.  Crie um gráfico de barras que mostre o total de vendas por região, usando Seaborn para a estética e Matplotlib para personalizar títulos e rótulos.

Estes exercícios ajudarão a solidificar sua compreensão sobre as capacidades de visualização de dados e como eles podem ser usados para comunicar insights de forma eficaz.

## Referências (Visualização de Dados e Manipulação Avançada)

-   [11] Data Visualisation in Python using Matplotlib and Seaborn. Disponível em: <https://www.geeksforgeeks.org/data-visualization/data-visualisation-in-python-using-matplotlib-and-seaborn/>
-   [12] Python Seaborn Tutorial For Beginners: Start Visualizing Data. Disponível em: <https://www.datacamp.com/tutorial/seaborn-python-tutorial>
-   [13] Working with missing data — pandas documentation. Disponível em: <https://pandas.pydata.org/docs/user_guide/missing_data.html>
-   [14] Merge, join, concatenate and compare — pandas documentation. Disponível em: <https://pandas.pydata.org/docs/user_guide/merging.html>

