# Exercício Resolvido — PCA Passo a Passo

> **Objetivo:** aplicar manualmente o algoritmo PCA por **autodecomposição** sobre um dataset $X \in \mathbb{R}^{12 \times 2}$, percorrendo cada um dos seis passos do procedimento canônico apresentado em `pca.md`. Ao final, o aluno deverá conseguir reproduzir todos os números com NumPy puro (sem `sklearn`).

---

## Enunciado

Um time de engenharia coletou **12 amostras** geradas por dois sensores fortemente correlacionados (por exemplo: tensão $x_1$ e corrente $x_2$ medidas em um mesmo barramento). Suspeita-se que a informação útil possa ser comprimida em uma **única dimensão latente**.

Aplique a PCA por autodecomposição e responda:

1. Qual a direção da **primeira componente principal** $v_1$?
2. Qual a **variância explicada acumulada** ao reter apenas $k=1$?
3. Quais os valores das projeções $z_i = x_i^c \cdot v_1$ dos 12 pontos sobre $v_1$?

---

## Dataset

A matriz de dados original $X \in \mathbb{R}^{12 \times 2}$ é:

| ID ($i$) | $x_1$ | $x_2$ |
|:---:|:---:|:---:|
| 1  | 1.0  | 2.0  |
| 2  | 2.0  | 1.0  |
| 3  | 3.0  | 4.0  |
| 4  | 4.0  | 3.0  |
| 5  | 5.0  | 5.0  |
| 6  | 5.0  | 6.0  |
| 7  | 6.0  | 7.0  |
| 8  | 7.0  | 6.0  |
| 9  | 8.0  | 9.0  |
| 10 | 9.0  | 9.0  |
| 11 | 10.0 | 10.0 |
| 12 | 12.0 | 10.0 |

> Plotando rapidamente, vê-se que a nuvem se alonga aproximadamente sobre a reta $y = x$ — exatamente o cenário ideal para a PCA brilhar.

---

## Passo 1 — Centralização

A PCA exige que a origem do espaço esteja no **centro de massa** dos dados; caso contrário, a "direção de maior variância" seria contaminada pelo deslocamento da nuvem em relação à origem.

### 1.1 Cálculo do vetor de médias $\bar{x}$

$$\bar{x}_1 = \frac{1}{n} \sum_{i=1}^{n} x_{i1} = \frac{1+2+3+4+5+5+6+7+8+9+10+12}{12} = \frac{72}{12} = 6.0$$

$$\bar{x}_2 = \frac{1}{n} \sum_{i=1}^{n} x_{i2} = \frac{2+1+4+3+5+6+7+6+9+9+10+10}{12} = \frac{72}{12} = 6.0$$

Portanto $\bar{x} = (6.0, \; 6.0)^\top$.

### 1.2 Construção de $X_c = X - \bar{x}$

Subtraímos $\bar{x}$ de cada linha de $X$:

| ID | $x_1^c = x_1 - 6$ | $x_2^c = x_2 - 6$ |
|:---:|:---:|:---:|
| 1  | -5 | -4 |
| 2  | -4 | -5 |
| 3  | -3 | -2 |
| 4  | -2 | -3 |
| 5  | -1 | -1 |
| 6  | -1 |  0 |
| 7  |  0 |  1 |
| 8  |  1 |  0 |
| 9  |  2 |  3 |
| 10 |  3 |  3 |
| 11 |  4 |  4 |
| 12 |  6 |  4 |

**Sanity check:** as colunas de $X_c$ devem somar zero.
$\sum x_1^c = -5-4-3-2-1-1+0+1+2+3+4+6 = 0$ ✓
$\sum x_2^c = -4-5-2-3-1+0+1+0+3+3+4+4 = 0$ ✓

---

## Passo 2 — Padronização (análise contextual)

A padronização é **opcional** e só se torna obrigatória quando as features estão em escalas radicalmente diferentes (ex.: salário em R\$ vs. idade em anos). Vamos verificar a necessidade comparando as **variâncias amostrais** das duas colunas.

### 2.1 Cálculo das variâncias

Lembrando que $\text{Var}(x_j) = \sigma_{jj} = \frac{1}{n-1} \sum_{i=1}^n (x_{ij} - \bar{x}_j)^2$:

$$\sum_{i=1}^{12} (x_{i1}^c)^2 = 25+16+9+4+1+1+0+1+4+9+16+36 = 122$$

$$\sum_{i=1}^{12} (x_{i2}^c)^2 = 16+25+4+9+1+0+1+0+9+9+16+16 = 106$$

Com $n-1 = 11$:

$$\text{Var}(x_1) = \sigma_{11} = \frac{122}{11} \approx 11.0909$$

$$\text{Var}(x_2) = \sigma_{22} = \frac{106}{11} \approx 9.6364$$

### 2.2 Decisão

Como $\sigma_{11} \approx \sigma_{22}$ (razão $\sigma_{11}/\sigma_{22} \approx 1.15$, ambas na faixa de $\sim 9.6$ a $\sim 11.1$), as duas features estão **na mesma ordem de grandeza de dispersão**. **Não aplicamos padronização** neste exercício — usaremos diretamente $X_c$.

> 💡 A padronização adequada divide cada coluna pela raiz da sua variância. Após esse passo, a matriz de covariância coincide com a **matriz de correlação**. Os autovetores resultantes seriam diferentes — e geralmente é isso que se quer fazer quando as variâncias diferem em **ordens de magnitude** (p. ex. $\sigma_{11}/\sigma_{22} > 100$).

---

## Passo 3 — Matriz de Covariância $\Sigma$

A covariância empírica é definida por $\Sigma = \frac{1}{n-1} X_c^\top X_c \in \mathbb{R}^{d \times d}$. Com $d = 2$, teremos uma matriz $2 \times 2$.

### 3.1 Cálculo dos somatórios

Já temos $\sum (x_1^c)^2 = 122$ e $\sum (x_2^c)^2 = 106$. Falta o termo cruzado:

$$\sum_{i=1}^{12} x_{i1}^c \cdot x_{i2}^c = (-5)(-4) + (-4)(-5) + (-3)(-2) + (-2)(-3) + (-1)(-1) + (-1)(0)$$
$$+ (0)(1) + (1)(0) + (2)(3) + (3)(3) + (4)(4) + (6)(4)$$
$$= 20 + 20 + 6 + 6 + 1 + 0 + 0 + 0 + 6 + 9 + 16 + 24 = 108$$

### 3.2 Montagem de $\Sigma$

$$X_c^\top X_c = \begin{pmatrix} 122 & 108 \\ 108 & 106 \end{pmatrix}$$

Dividindo por $n - 1 = 11$:

$$\Sigma = \frac{1}{11} \begin{pmatrix} 122 & 108 \\ 108 & 106 \end{pmatrix} \approx \begin{pmatrix} 11.0909 & 9.8182 \\ 9.8182 & 9.6364 \end{pmatrix}$$

> **Leitura da matriz:** a diagonal contém as variâncias individuais; o termo $\sigma_{12} \approx 9.82$ é **grande e positivo**, indicando que $x_1$ e $x_2$ crescem juntas — i.e., são fortemente correlacionadas. É exatamente esse sinal redundante que a PCA irá comprimir.

---

## Passo 4 — Autodecomposição

Queremos $\lambda, v$ tais que $\Sigma v = \lambda v$. Resolvemos primeiro a **equação característica** $\det(\Sigma - \lambda I) = 0$.

### 4.1 Polinômio característico

$$\det \begin{pmatrix} 11.0909 - \lambda & 9.8182 \\ 9.8182 & 9.6364 - \lambda \end{pmatrix} = 0$$

$$(11.0909 - \lambda)(9.6364 - \lambda) - (9.8182)^2 = 0$$

Para evitar acúmulo de erro de arredondamento, multiplicamos a equação por $11^2 = 121$ e fazemos a substituição $u = 11\lambda$:

$$(122 - u)(106 - u) - 108^2 = 0$$

$$u^2 - 228\,u + 12932 - 11664 = 0$$

$$\boxed{u^2 - 228\,u + 1268 = 0}$$

### 4.2 Raízes do polinômio

Pela fórmula de Bhaskara:

$$u = \frac{228 \pm \sqrt{228^2 - 4 \cdot 1268}}{2} = \frac{228 \pm \sqrt{51984 - 5072}}{2} = \frac{228 \pm \sqrt{46912}}{2}$$

$$\sqrt{46912} \approx 216.5918$$

$$u_1 = \frac{228 + 216.5918}{2} \approx 222.296 \qquad u_2 = \frac{228 - 216.5918}{2} \approx 5.704$$

Voltando para $\lambda = u/11$:

$$\boxed{\lambda_1 \approx 20.2087 \qquad \lambda_2 \approx 0.5186}$$

**Sanity check (traço):** $\lambda_1 + \lambda_2 \approx 20.7273 = \sigma_{11} + \sigma_{22} = (122 + 106)/11$ ✓

### 4.3 Variância explicada

$$\text{VE}_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{20.2087}{20.7273} \approx 0.9750 \;\;(97{,}50\%)$$

$$\text{VE}_2 = \frac{\lambda_2}{\lambda_1 + \lambda_2} \approx 0.0250 \;\;(2{,}50\%)$$

Um único componente já carrega **97,5% da informação**.

### 4.4 Autovetor associado a $\lambda_1$

Resolvemos $(\Sigma - \lambda_1 I)\,v_1 = 0$. Pegando a primeira linha:

$$(11.0909 - 20.2087)\,v_{11} + 9.8182\,v_{12} = 0$$
$$-9.1178\,v_{11} + 9.8182\,v_{12} = 0$$
$$v_{12} = \frac{9.1178}{9.8182}\,v_{11} \approx 0.9287\,v_{11}$$

Aplicando a restrição $\|v_1\| = 1$:

$$v_{11}^2 + v_{12}^2 = 1 \;\Longrightarrow\; v_{11}^2 \,(1 + 0.9287^2) = 1$$
$$v_{11}^2 \cdot 1.8624 = 1 \;\Longrightarrow\; v_{11} \approx 0.7328$$
$$v_{12} \approx 0.9287 \cdot 0.7328 \approx 0.6805$$

$$\boxed{v_1 \approx \begin{pmatrix} 0.7328 \\ 0.6805 \end{pmatrix}}$$

> Geometricamente, $v_1$ aponta para o primeiro quadrante formando um ângulo $\arctan(0.6805 / 0.7328) \approx 42.9^\circ$ com o eixo $x_1$ — bem próximo da bissetriz $y = x$, como prevíamos pelo gráfico de dispersão.

### 4.5 Autovetor associado a $\lambda_2$

Como $\Sigma$ é simétrica, $v_2 \perp v_1$. Em $\mathbb{R}^2$ basta rotacionar $v_1$ em $90^\circ$:

$$\boxed{v_2 \approx \begin{pmatrix} -0.6805 \\ 0.7328 \end{pmatrix}}$$

(Pode-se confirmar resolvendo $(\Sigma - \lambda_2 I)\,v_2 = 0$.)

---

## Passo 5 — Ordenação e Truncamento

### 5.1 Ordenação decrescente

Já temos $\lambda_1 > \lambda_2$, então a ordem está correta. Em código (`numpy.linalg.eig`), a ordem **não** é garantida — sempre reordene com `np.argsort(eigvals)[::-1]`.

### 5.2 Escolha de $k$

Pelo critério de variância acumulada $\geq 95\%$, escolhemos:

$$k = 1 \quad \text{pois}\quad \text{VA}_1 = 97{,}5\% \geq 95\%$$

### 5.3 Construção da matriz de projeção $W_k$

Empilhamos os $k$ primeiros autovetores como **colunas**:

$$W_1 = \begin{pmatrix} 0.7328 \\ 0.6805 \end{pmatrix} \in \mathbb{R}^{2 \times 1}$$

> Se quiséssemos manter $k=2$ (rotação completa, sem compressão), teríamos $W_2 = [v_1 \;|\; v_2]$, que é uma matriz ortogonal $2\times 2$.

---

## Passo 6 — Projeção Espacial

Calculamos $Z = X_c \cdot W_1 \in \mathbb{R}^{12 \times 1}$. Cada linha de $Z$ é o produto interno do ponto centralizado com $v_1$:

$$z_i = 0.7328 \cdot x_{i1}^c + 0.6805 \cdot x_{i2}^c$$

Aplicando ponto a ponto:

| ID | $x_1^c$ | $x_2^c$ | $z_i = X_c \cdot v_1$ |
|:---:|:---:|:---:|:---:|
| 1  | -5 | -4 | $-3.664 - 2.722 = \mathbf{-6.386}$ |
| 2  | -4 | -5 | $-2.931 - 3.402 = \mathbf{-6.334}$ |
| 3  | -3 | -2 | $-2.198 - 1.361 = \mathbf{-3.559}$ |
| 4  | -2 | -3 | $-1.466 - 2.041 = \mathbf{-3.507}$ |
| 5  | -1 | -1 | $-0.733 - 0.680 = \mathbf{-1.413}$ |
| 6  | -1 |  0 | $-0.733 + 0.000 = \mathbf{-0.733}$ |
| 7  |  0 |  1 | $0.000 + 0.680 = \mathbf{0.680}$ |
| 8  |  1 |  0 | $0.733 + 0.000 = \mathbf{0.733}$ |
| 9  |  2 |  3 | $1.466 + 2.041 = \mathbf{3.507}$ |
| 10 |  3 |  3 | $2.198 + 2.041 = \mathbf{4.240}$ |
| 11 |  4 |  4 | $2.931 + 2.722 = \mathbf{5.653}$ |
| 12 |  6 |  4 | $4.397 + 2.722 = \mathbf{7.119}$ |

---

## Verificação Final

Três sanity-checks que **devem sempre ser executados** ao final de uma PCA manual:

**1. Média de $Z$ deve ser zero (a projeção preserva o centro de massa):**
$$\bar{z} = \frac{1}{12} \sum_{i=1}^{12} z_i \approx 0.000 \;\checkmark$$

**2. Variância de $Z$ deve coincidir com $\lambda_1$:**
$$\text{Var}(Z) = \frac{1}{n-1} \sum_{i=1}^{12} z_i^2 = \frac{222.31}{11} \approx 20.21 \approx \lambda_1 \;\checkmark$$

**3. Variância total preservada (traço da matriz de covariância):**
$$\text{tr}(\Sigma) = \lambda_1 + \lambda_2 = 20.7273 \;\checkmark$$

---

## Conclusão

* O dataset bidimensional original foi reduzido a **1 dimensão** preservando **97,5%** da variância.
* A direção da primeira componente principal é $v_1 \approx (0.7328, \; 0.6805)$, o que confirma a intuição visual de que a nuvem segue a bissetriz $y = x$.
* Os 12 escalares $z_i$ na tabela acima substituem os 24 valores originais sem perda perceptual de informação — uma compressão de **50%** com fidelidade de **97,5%**.

> 💡 **Próximo passo do aluno:** reproduzir esta resolução em Python sem `sklearn`, comparando a matriz $W_1$ obtida via `np.linalg.eig` com a obtida via `np.linalg.svd` aplicada diretamente em $X_c$. Ambas devem coincidir até sinal.
