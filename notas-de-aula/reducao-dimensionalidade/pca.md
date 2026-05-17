# Inteligência Artificial — Aula: Redução de Dimensionalidade e PCA (Versão Avançada)

> **Pré-requisitos:** Álgebra linear avançada (matrizes, vetores, projeções ortogonais, determinante e o conceito de autovalores/autovetores), estatística básica (média, variância e covariância) e Python/NumPy.
>
> **Duração estimada:** 3h
>
> **Disciplina:** Inteligência Artificial / Aprendizado de Máquina

---

## Objetivos de Aprendizagem

Ao final desta aula, o aluno será capaz de:

1. Situar a redução de dimensionalidade no paradigma do **aprendizado não-supervisionado**.
2. Explicar os impactos geométricos e estatísticos da **maldição da dimensionalidade**.
3. Deduzir e aplicar rigorosamente o cálculo da **Matriz de Covariância** e da **Equação Característica** para encontrar autovalores e autovetores.
4. Calcular e interpretar a **Variância Explicada Acumulada** e o gráfico do cotovelo (*scree plot*) para seleção de hiperparâmetros.
5. Identificar as limitações da PCA e discernir quando utilizar métodos não-lineares (**t-SNE** e **UMAP**).

---

## 1. O Cenário: Aprendizado Não-Supervisionado

No aprendizado não-supervisionado, trabalhamos **apenas com os dados brutos $\{x_i\}$, sem rótulos ou alvos preditivos ($y$)**. O objetivo central do paradigma é mapear a estrutura intrínseca dos dados.

A **Redução de Dimensionalidade** busca uma transformação matemática capaz de mapear dados originais de um espaço de alta dimensão $X \in \mathbb{R}^{n \times d}$ para um subespaço compacto $Z \in \mathbb{R}^{n \times k}$ (onde $k \ll d$), garantindo a retenção da informação estatística mais relevante e a eliminação de ruídos ou redundâncias.

> 💡 **Intuição Pedagógica:** No aprendizado supervisionado, há um professor corrigindo sua prova. No não-supervisionado, você é colocado em uma biblioteca gigante totalmente bagunçada e sem catálogo. Seu papel é organizar os livros por conta própria, descobrindo quais características latentes (gênero, densidade de vocabulário, época) melhor estruturam o acervo.

---

## 2. Por que reduzir? A Maldição da Dimensionalidade

À medida que o número de atributos ($d$) cresce, as propriedades geométricas e estatísticas do espaço Euclidiano degradam-se bruscamente:

* **Espaço Ralo (Sparsity):** O volume do espaço cresce exponencialmente em relação a $d$. Para cobrir uniformemente um hipercubo de 10 dimensões com apenas 10 divisões por eixo, seriam necessários $10^{10}$ (10 bilhões) de pontos. Na prática, datasets reais de alta dimensão tornam-se extremamente esparsos, impedindo que modelos generalizem sem sofrer *overfitting*.
* **Equalização de Distâncias:** Em alta dimensão, a distância geométrica (Minkowski/Euclidiana) entre o ponto mais próximo e o ponto mais distante de uma query aproxima-se de uma constante. Como todas as distâncias tendem a se equalizar, o conceito de "vizinhança" perde o sentido semântico, colapsando o desempenho de algoritmos como K-NN, K-Means e SVMs com kernel RBF.

### Motivações Práticas para Redução:
1. **Visualização:** Restrição cognitiva humana limitando a interpretação a plots em 2D ou 3D.
2. **Eficiência Computacional:** Redução de complexidade de tempo e espaço ($O$) ao alimentar modelos downstream com matrizes menores.
3. **Regularização Inerente:** Descarte de componentes de baixa variância funciona frequentemente como remoção de ruído e colinearidade.

---

## 3. PCA (Principal Component Analysis)

A PCA é o algoritmo linear canônico para redução de dimensionalidade. Sua meta é encontrar direções ortogonais no espaço que maximizem a variância dos dados projetados.

### 3.1 A Intuição Geométrica da Lanterna 🔦
Imagine que sua nuvem de dados 3D tem o formato de um balão de festa elíptico e alongado flutuando em um quarto escuro. Fazer a PCA significa rotacionar uma lanterna ao redor desse balão procurando o ângulo exato em que a **sombra projetada na parede seja a maior e mais longa possível**.
* A **Primeira Componente Principal (PC1)** representa o eixo de maior alongamento da nuvem, ou seja, a direção de **máxima variância**.
* A **Segunda Componente Principal (PC2)** é uma nova direção, obrigatoriamente **perpendicular** à primeira, que captura o máximo da variância restante.

---

## 4. O Rigor Matemático da PCA

### 4.1 Cálculo da Matriz de Covariância ($\Sigma$)
A matriz de covariância descreve a variabilidade conjunta e a dependência linear entre todas as features do dataset. Dado uma matriz de dados centralizada $X_c \in \mathbb{R}^{n \times d}$ (onde a média de cada coluna foi subtraída), a matriz de covariância empírica $\Sigma \in \mathbb{R}^{d \times d}$ é formalmente definida por:

$$\Sigma = \frac{1}{n-1} X_c^\top X_c$$

Cada elemento individual $\sigma_{jk}$ (linha $j$, coluna $k$) que compõe a matriz representa a covariância entre o atributo $j$ e o atributo $k$, calculada via:

$$\sigma_{jk} = \frac{1}{n-1} \sum_{i=1}^{n} (x_{ij} - \bar{x}_j)(x_{ik} - \bar{x}_k)$$

* Se $\sigma_{jk} > 0$: As variáveis mudam juntas na mesma direção.
* Se $\sigma_{jk} < 0$: As variáveis mudam em direções opostas.
* Se $\sigma_{jk} \approx 0$: As variáveis são linearmente independentes.
* A diagonal principal ($\sigma_{jj}$) contém a variância isolada de cada atributo.

### 4.2 O Problema de Otimização e a Equação Característica
Buscamos uma direção de projeção definida pelo vetor unitário $w$ que maximize a variância dos dados projetados ($w^\top \Sigma w$). Pelo Teorema Espectral, os pontos críticos desse sistema sob a restrição $\|w\| = 1$ coincidem com a autodecomposição da matriz de covariância:

$$\Sigma w = \lambda w$$

Para extrair os **autovalores ($\lambda$)**, que quantificam a magnitude da variância em cada eixe, resolvemos a **Equação Característica** impondo que o operador linear seja singular:

$$\det(\Sigma - \lambda I) = 0$$

Onde $I \in \mathbb{R}^{d \times d}$ é a matriz identidade. A expansão deste determinante resulta em um polinômio característico de grau $d$. As suas $d$ raízes reais e não-negativas são os autovalores ($\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_d \ge 0$).

Após isolar cada $\lambda_i$, encontramos o seu **autovetor ($v_i$)** associado substituindo-o de volta no sistema linear homogêneo:

$$(\Sigma - \lambda_i I)v_i = 0$$

Cada autovetor obtido é normalizado de modo que $\|v_i\| = 1$, fornecendo a direção geométrica exata da Componente Principal.

---

## 5. Seleção Estatística de Componentes

### 5.1 Variância Explicada Acumulada
A variância total contida no sistema equivale à soma das variâncias de todas as features originais, o que matematicamente corresponde ao traço da matriz de covariância ($\text{tr}(\Sigma)$) ou à soma de todos os seus autovalores. 

A proporção de variância explicada ($\text{VE}$) por uma única componente $i$ é dada pela razão:

$$\text{VE}_i = \frac{\lambda_i}{\sum_{j=1}^d \lambda_j}$$

A **Variância Explicada Acumulada ($\text{VA}$)** avalia o percentual total de informação retido ao truncar o espaço nas primeiras $k$ componentes:

$$\text{VA}_k = \sum_{i=1}^k \text{VE}_i = \frac{\sum_{i=1}^k \lambda_i}{\sum_{j=1}^d \lambda_j}$$

A heurística padrão de mercado adota o menor valor de $k$ tal que $\text{VA}_k \ge 0.90 \text{ ou } 0.95$ (retendo 90% a 95% da informação original).

### 5.2 O Método e Cálculo do Cotovelo (*Scree Plot*)
O *Scree Plot* ordena de forma decrescente os índices das componentes no eixo X e a magnitude de seus respectivos autovalores $\lambda_i$ (ou % de VE) no eixo Y. 

Matematicamente, buscamos encontrar o ponto de **máxima curvatura** da função discreta (ponto de inflexão). Esse ponto representa graficamente um "cotovelo". A partir desse limite, a taxa de decréscimo dos autovalores estabiliza-se em um patamar linear baixo (cauda achatada), sinalizando que as componentes subsequentes não carregam sinal estrutural, apenas ruído estocástico. Mantêm-se no modelo as componentes localizadas estritamente antes da inflexão.

---

## 6. O Algoritmo Passo a Passo Formal

Dado $X \in \mathbb{R}^{n \times d}$ e o número alvo de dimensões $k$:

1. **Centralização:** Calcular o vetor coluna de médias $\bar{x} \in \mathbb{R}^d$ e computar $X_c = X - \bar{x}$.
2. **Padronização:** Caso as features estejam em escalas distintas, computar o desvio padrão de cada coluna e dividir: $x_{ij} = x_{ij} / s_j$.
3. **Matriz de Covariância:** Construir $\Sigma = \frac{1}{n-1} X_c^\top X_c$.
4. **Decomposição:** Resolver $\det(\Sigma - \lambda I) = 0$ para obter os autovalores e seus respectivos autovetores normados $(\Sigma - \lambda I)v = 0$.
5. **Ordenação e Truncamento:** Ordenar os pares por ordem decrescente de $\lambda$ e empilhar os $k$ primeiros autovetores como colunas para construir a matriz de pesos/projeção $W_k \in \mathbb{R}^{d \times k}$.
6. **Projeção Espacial:** Calcular a nova matriz reduzida através do produto matricial: $Z = X_c W_k \in \mathbb{R}^{n \times k}$.

> 💡 **Nota Computacional de Engenharia:** Construir explicitamente $X_c^\top X_c$ pode induzir a erros de arredondamento e instabilidade numérica se $d$ for muito grande. Frameworks como o *Scikit-Learn* executam a PCA aplicando a **SVD (Decomposição em Valores Singulares)** diretamente sobre a matriz $X_c$ ($X_c = U S V^\top$). As colunas de $V$ equivalem exatamente aos autovetores de $\Sigma$, contornando o cálculo explícito da matriz de covariância de forma estável e performática.

---

## 7. Limitações e Alternativas Não-Lineares

Por ser uma transformação projetiva linear, a PCA assume que o dataset habita um hiperplano plano. Diante de estruturas com topologia curva (ex: o dataset *Swiss Roll* ou distribuições em espiral), a PCA achata a geometria, colapsando pontos distantes e destruindo a vizinhança original.

Para tarefas focadas puramente em **visualização de dados exploratória**, empregamos alternativas não-lineares baseadas em grafos de proximidade:

* **t-SNE:** Constrói distribuições de probabilidade gaussianas sobre as distâncias dos pares no espaço original e distribuições $t$-Student no espaço reduzido, minimizando a Divergência KL entre ambas via gradiente descendente. É excelente para isolar clusters complexos em 2D, mas distorce distâncias globais.
* **UMAP:** Fundamentado em geometria riemanniana e topologia algébrica, modela o espaço como conjuntos difusos. É significativamente mais rápido que o t-SNE, escala de forma eficiente para milhões de amostras e preserva de forma superior o balanço entre vizinhança local e estrutura macro/global do dataset.

---

## Exemplo Detalhado Resolvido

**Enunciado:** Considere as seguintes 3 amostras já centralizadas no espaço $\mathbb{R}^2$: $x_1 = (-1, -1)$, $x_2 = (0, 0)$ e $x_3 = (1, 1)$. Encontre a primeira componente principal e sua variância explicada através do cálculo completo.

**Solução:**

**1. Construção da Matriz de Covariância $\Sigma$:**
A nossa matriz de dados centralizados é:
$$X_c = \begin{pmatrix} -1 & -1 \\ 0 & 0 \\ 1 & 1 \end{pmatrix}$$

Multiplicando $X_c^\top X_c$:
$$X_c^\top X_c = \begin{pmatrix} -1 & 0 & 1 \\ -1 & 0 & 1 \end{pmatrix} \begin{pmatrix} -1 & -1 \\ 0 & 0 \\ 1 & 1 \end{pmatrix} = \begin{pmatrix} (-1)^2+1^2 & (-1)(-1)+(1)(1) \\ (-1)(-1)+(1)(1) & (-1)^2+1^2 \end{pmatrix} = \begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix}$$

Dividindo por $n-1 = 3-1 = 2$:
$$\Sigma = \frac{1}{2} \begin{pmatrix} 2 & 2 \\ 2 & 2 \end{pmatrix} = \begin{pmatrix} 1 & 1 \\ 1 & 1 \end{pmatrix}$$

**2. Resolução da Equação Característica para Autovalores ($\lambda$):**
$$\det(\Sigma - \lambda I) = \det \begin{pmatrix} 1 - \lambda & 1 \\ 1 & 1 - \lambda \end{pmatrix} = 0$$
$$(1 - \lambda)(1 - \lambda) - (1)(1) = 0 \implies 1 - 2\lambda + \lambda^2 - 1 = 0$$
$$\lambda^2 - 2\lambda = 0 \implies \lambda(\lambda - 2) = 0$$

As duas raízes do polinômio são: $\lambda_1 = 2$ e $\lambda_2 = 0$.

**3. Cálculo da Variância Explicada:**
$$\text{VE}_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{2}{2 + 0} = 1.0 \implies 100\%$$
A primeira componente captura toda a informação contida nas variáveis originais.

**4. Extração do Autovetor associado a $\lambda_1 = 2$:**
$$(\Sigma - \lambda_1 I)v_1 = 0 \implies \begin{pmatrix} 1 - 2 & 1 \\ 1 & 1 - 2 \end{pmatrix} \begin{pmatrix} v_{11} \\ v_{12} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix}$$
$$\begin{pmatrix} -1 & 1 \\ 1 & -1 \end{pmatrix} \begin{pmatrix} v_{11} \\ v_{12} \end{pmatrix} = \begin{pmatrix} 0 \\ 0 \end{pmatrix} \implies -v_{11} + v_{12} = 0 \implies v_{11} = v_{12}$$

Sabendo que o vetor deve ser unitário ($v_{11}^2 + v_{12}^2 = 1$):
$$2v_{11}^2 = 1 \implies v_{11} = \frac{1}{\sqrt{2}}$$

O autovetor final (PC1) é:
$$v_1 = \begin{pmatrix} \frac{1}{\sqrt{2}} \\ \frac{1}{\sqrt{2}} \end{pmatrix} \approx \begin{pmatrix} 0.707 \\ 0.707 \end{pmatrix}$$

**Conclusão:** Como os pontos originais estavam perfeitamente alinhados na bissetriz dos quadrantes ímpares ($y = x$), a PCA detectou essa colinearidade e indicou que podemos reduzir o problema para 1D sem perder absolutamente nenhum dado.

---

## Exercícios de Fixação

1. Prove formalmente porque a soma de todos os autovalores de uma matriz de covariância sempre será igual à variância total (soma das variâncias das features originais).
2. Explique detalhadamente o risco analítico de aplicar o método do cotovelo (*Scree Plot*) em um dataset cujos atributos originais possuem escalas radicalmente distintas e não passaram por um processo de padronização prévia.

---
*Fim das notas de aula avançadas.*