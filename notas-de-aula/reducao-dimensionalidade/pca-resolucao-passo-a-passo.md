# PCA Passo a Passo — Resolução Completa

> **Exemplo trabalhado:** redução de dimensionalidade de 12 pares $(x_1, x_2)$ usando Análise de Componentes Principais. Cobre todos os seis passos do algoritmo, da tabela bruta até as coordenadas no novo eixo, com verificações de sanidade em cada etapa.
>
> **Pré-requisitos:** álgebra linear básica (autovalores, autovetores, determinantes 2×2), estatística (média, variância, covariância), Bhaskara.

---

## Dataset

| ID ($i$) | $x_1$ | $x_2$ |
|:---:|:---:|:---:|
| 1  | 1  | 3  |
| 2  | 2  | 1  |
| 3  | 3  | 5  |
| 4  | 4  | 2  |
| 5  | 5  | 4  |
| 6  | 5  | 7  |
| 7  | 6  | 8  |
| 8  | 7  | 5  |
| 9  | 8  | 10 |
| 10 | 9  | 8  |
| 11 | 10 | 11 |
| 12 | 12 | 8  |

São duas variáveis com correlação positiva forte mas não perfeita — o cenário típico em que a PCA agrega valor real (em vez de ser trivial ou ineficaz).

---

## Passo 1 — Médias

A média de cada variável é a soma de seus valores dividida por $n = 12$.

**Soma de $x_1$:**
$$
\sum_{i=1}^{12} x_{1,i} = 1 + 2 + 3 + 4 + 5 + 5 + 6 + 7 + 8 + 9 + 10 + 12 = 72.
$$

**Soma de $x_2$:**
$$
\sum_{i=1}^{12} x_{2,i} = 3 + 1 + 5 + 2 + 4 + 7 + 8 + 5 + 10 + 8 + 11 + 8 = 72.
$$

**Médias:**
$$
\bar{x}_1 = \frac{72}{12} = 6, \qquad \bar{x}_2 = \frac{72}{12} = 6.
$$

O **centróide da nuvem** é o ponto $(\bar{x}_1, \bar{x}_2) = (6, 6)$ — é em torno dele que vamos centralizar os dados no próximo passo.

> 💡 **Sobre a escolha de números redondos.** Médias inteiras não são exigência da PCA — o método funciona com quaisquer dados. Mas para uma resolução à mão, escolher dados que produzem médias inteiras evita carregar frações por todos os passos seguintes e reduz o risco de erros aritméticos.

---

## Passo 2 — Centralização

Subtraindo o centróide $(6, 6)$ de cada ponto, e já calculando as três colunas auxiliares que vão alimentar a matriz de covariância:

| ID | $x_1$ | $x_2$ | $x_1 - \bar{x}_1$ | $x_2 - \bar{x}_2$ | $(x_1 - \bar{x}_1)^2$ | $(x_2 - \bar{x}_2)^2$ | $(x_1 - \bar{x}_1)(x_2 - \bar{x}_2)$ |
|:---:|:---:|:---:|:---:|:---:|:---:|:---:|:---:|
| 1  | 1  | 3  | $-5$ | $-3$ | 25  | 9   | $+15$ |
| 2  | 2  | 1  | $-4$ | $-5$ | 16  | 25  | $+20$ |
| 3  | 3  | 5  | $-3$ | $-1$ | 9   | 1   | $+3$  |
| 4  | 4  | 2  | $-2$ | $-4$ | 4   | 16  | $+8$  |
| 5  | 5  | 4  | $-1$ | $-2$ | 1   | 4   | $+2$  |
| 6  | 5  | 7  | $-1$ | $+1$ | 1   | 1   | $-1$  |
| 7  | 6  | 8  | $0$  | $+2$ | 0   | 4   | $0$  |
| 8  | 7  | 5  | $+1$ | $-1$ | 1   | 1   | $-1$  |
| 9  | 8  | 10 | $+2$ | $+4$ | 4   | 16  | $+8$  |
| 10 | 9  | 8  | $+3$ | $+2$ | 9   | 4   | $+6$  |
| 11 | 10 | 11 | $+4$ | $+5$ | 16  | 25  | $+20$ |
| 12 | 12 | 8  | $+6$ | $+2$ | 36  | 4   | $+12$ |
| **Σ** | — | — | **0** | **0** | **122** | **110** | **+92** |

### Verificações

**Soma dos desvios deve ser zero.** Conferindo a coluna $(x_1 - \bar{x}_1)$:
$$
(-5) + (-4) + (-3) + (-2) + (-1) + (-1) + 0 + 1 + 2 + 3 + 4 + 6 = 0. \;\checkmark
$$
Idem para $(x_2 - \bar{x}_2)$. Se algum desses dois somatórios não der zero, há erro de aritmética em algum lugar.

### O que significam as três últimas colunas

Os totais do rodapé são exatamente as **somas de produtos** que alimentam a matriz de covariância:

- $S_{xx} = \sum (x_1 - \bar{x}_1)^2 = 122$ → vira $\Sigma_{11}$ (variância de $x_1$) ao dividir por $n$.
- $S_{yy} = \sum (x_2 - \bar{x}_2)^2 = 110$ → vira $\Sigma_{22}$ (variância de $x_2$) ao dividir por $n$.
- $S_{xy} = \sum (x_1 - \bar{x}_1)(x_2 - \bar{x}_2) = +92$ → vira $\Sigma_{12}$ (covariância) ao dividir por $n$.

> 💡 **Sinal de $S_{xy}$ vale ouro.** O sinal da soma dos produtos centralizados antecipa o tipo de relação:
> - $S_{xy} > 0$ → variáveis variam juntas (correlação positiva). É o nosso caso.
> - $S_{xy} < 0$ → variam em direções opostas.
> - $S_{xy} \approx 0$ → variáveis quase independentes. PCA tem pouco a oferecer.

---

## Passo 3 — Matriz de covariância

Dividindo cada soma do passo anterior por $n = 12$:

$$
\Sigma = \frac{1}{n} \begin{pmatrix} S_{xx} & S_{xy} \\ S_{xy} & S_{yy} \end{pmatrix} = \frac{1}{12} \begin{pmatrix} 122 & 92 \\ 92 & 110 \end{pmatrix} \approx \begin{pmatrix} 10{,}17 & 7{,}67 \\ 7{,}67 & 9{,}17 \end{pmatrix}.
$$

### Lendo a matriz

| Entrada | Valor | Significado |
|---|---|---|
| $\Sigma_{11} = 10{,}17$ | variância de $x_1$ | Espalhamento de $x_1$ em torno de $\bar{x}_1 = 6$. |
| $\Sigma_{22} = 9{,}17$ | variância de $x_2$ | Espalhamento de $x_2$ em torno de $\bar{x}_2 = 6$. |
| $\Sigma_{12} = \Sigma_{21} = 7{,}67$ | covariância | $x_1$ e $x_2$ tendem a andar juntos. |

**Simetria.** $\Sigma_{12} = \Sigma_{21}$ sempre. A matriz de covariância é simétrica por construção, e isso garante (teoricamente) que os autovalores serão reais e os autovetores ortogonais — exatamente o que precisamos para a PCA.

**Correlação adimensional.** Dividindo a covariância pelo produto dos desvios-padrão:
$$
r_{12} = \frac{\Sigma_{12}}{\sqrt{\Sigma_{11} \cdot \Sigma_{22}}} = \frac{7{,}67}{\sqrt{10{,}17 \cdot 9{,}17}} = \frac{7{,}67}{\sqrt{93{,}26}} \approx 0{,}79.
$$
Correlação de 0,79 confirma: relação positiva forte, mas com folga.

**Anisotropia.** $\Sigma_{11} \neq \Sigma_{22}$ (10,17 contra 9,17). A nuvem é ligeiramente mais espalhada horizontalmente do que verticalmente — o que significa que $w_1$ não será exatamente a bissetriz, mas algo um pouco inclinado em favor de $x_1$.

---

## Passo 4 — Polinômio característico e autovalores

Vamos montar $\Sigma - \lambda I$ e impor $\det(\Sigma - \lambda I) = 0$.

### Calculando traço e determinante (em frações exatas)

Trabalhar com decimais carrega ruído de arredondamento. Vou usar frações.

**Traço de $\Sigma$:**
$$
\text{tr}(\Sigma) = \Sigma_{11} + \Sigma_{22} = \frac{122}{12} + \frac{110}{12} = \frac{232}{12} = \frac{58}{3}.
$$

**Determinante de $\Sigma$:**
$$
\det(\Sigma) = \Sigma_{11} \Sigma_{22} - \Sigma_{12}^2 = \frac{122 \cdot 110 - 92^2}{144} = \frac{13\,420 - 8\,464}{144} = \frac{4\,956}{144} = \frac{413}{12}.
$$

### Polinômio característico

$$
p(\lambda) = \lambda^2 - \text{tr}(\Sigma)\,\lambda + \det(\Sigma) = \lambda^2 - \frac{58}{3}\lambda + \frac{413}{12} = 0.
$$

Multiplicando toda a equação por $12$ para limpar denominadores:

$$
\boxed{\;12\lambda^2 - 232\lambda + 413 = 0.\;}
$$

### Resolvendo via Bhaskara

Coeficientes: $a = 12$, $b = -232$, $c = 413$.

**Discriminante:**
$$
\Delta = b^2 - 4ac = 53\,824 - 19\,824 = 34\,000.
$$

**Raiz quadrada:**
$$
\sqrt{\Delta} = \sqrt{34\,000} \approx 184{,}39.
$$

**Raízes:**
$$
\lambda = \frac{232 \pm 184{,}39}{24},
$$

$$
\lambda_1 = \frac{416{,}39}{24} \approx 17{,}35, \qquad \lambda_2 = \frac{47{,}61}{24} \approx 1{,}98.
$$

### Verificações de sanidade

**1. Soma = traço.** $\lambda_1 + \lambda_2 = 17{,}35 + 1{,}98 = 19{,}33 = \tfrac{58}{3}$. ✓

**2. Produto = determinante.** $\lambda_1 \cdot \lambda_2 \approx 17{,}35 \cdot 1{,}98 \approx 34{,}35 \approx \tfrac{413}{12} \approx 34{,}42$. ✓

(Pequena diferença é arredondamento de $\sqrt{34\,000}$.)

### Variância explicada

$$
\rho_1 = \frac{\lambda_1}{\lambda_1 + \lambda_2} = \frac{17{,}35}{19{,}33} \approx 89{,}74\%, \qquad \rho_2 \approx 10{,}26\%.
$$

Reduzindo de $\mathbb{R}^2$ para $\mathbb{R}^1$, retemos ~90% da variabilidade — bom trade-off.

> 💡 **Dispersão vs. covariância.** Se tivéssemos usado $X_c^\top X_c$ direto (sem dividir por 12), obteríamos $\lambda_1 \approx 208{,}20$ e $\lambda_2 \approx 23{,}80$. Note que $208{,}20/12 \approx 17{,}35$. O fator $1/n$ é só uma constante: escala os autovalores mas **não muda os autovetores nem as proporções** $\rho_i$.

---

## Passo 5 — Autovetores

Para cada autovalor, resolvemos $(\Sigma - \lambda_i I) w = 0$ e normalizamos.

### Autovetor $w_1$ (associado a $\lambda_1 \approx 17{,}35$)

$$
(\Sigma - \lambda_1 I) = \begin{pmatrix} 10{,}17 - 17{,}35 & 7{,}67 \\ 7{,}67 & 9{,}17 - 17{,}35 \end{pmatrix} = \begin{pmatrix} -7{,}18 & 7{,}67 \\ 7{,}67 & -8{,}18 \end{pmatrix}.
$$

**Por que esse sistema tem infinitas soluções.** Por construção, $\det(\Sigma - \lambda_1 I) = 0$ — foi exatamente assim que escolhemos $\lambda_1$. Determinante zero significa que as duas linhas são linearmente dependentes — sobra uma reta inteira de soluções.

**Usando a primeira linha** (chamando o autovetor de $(a, b)$):
$$
-7{,}18 \, a + 7{,}67 \, b = 0 \;\;\Longrightarrow\;\; b = \frac{7{,}18}{7{,}67} \, a \approx 0{,}9362 \, a.
$$

**Tomando $a = 1$:**
$$
w_1^{\text{(não-normalizado)}} = (1,\ 0{,}9362).
$$

**Normalizando para $\|w_1\| = 1$:**
$$
\|w_1\| = \sqrt{1^2 + 0{,}9362^2} = \sqrt{1{,}8765} \approx 1{,}3699.
$$

$$
\boxed{\;w_1 = \left(\frac{1}{1{,}3699},\ \frac{0{,}9362}{1{,}3699}\right) \approx (0{,}7300,\ 0{,}6834).\;}
$$

**Verificação na segunda linha:**
$$
7{,}67 \cdot 0{,}7300 + (-8{,}18) \cdot 0{,}6834 = 5{,}5991 - 5{,}5902 \approx 0. \;\checkmark
$$

**Interpretação geométrica.** $w_1$ está inclinado a $\arctan(0{,}6834/0{,}7300) \approx 43{,}1°$ acima do eixo $x_1$. Próximo da bissetriz (45°), mas ligeiramente puxado para $x_1$ — coerente com $\Sigma_{11} > \Sigma_{22}$.

### Autovetor $w_2$ (associado a $\lambda_2 \approx 1{,}98$)

$$
(\Sigma - \lambda_2 I) = \begin{pmatrix} 8{,}19 & 7{,}67 \\ 7{,}67 & 7{,}19 \end{pmatrix}.
$$

**Primeira linha:**
$$
8{,}19 \, a + 7{,}67 \, b = 0 \;\;\Longrightarrow\;\; b = -\frac{8{,}19}{7{,}67} \, a \approx -1{,}0678 \, a.
$$

**Tomando $a = 1$ e normalizando:**
$$
\|w_2\| = \sqrt{1 + 1{,}0678^2} \approx 1{,}4629.
$$

$$
\boxed{\;w_2 \approx (0{,}6836,\ -0{,}7299).\;}
$$

### Verificações finais

**1. Ortogonalidade** (autovetores de matriz simétrica são perpendiculares):
$$
w_1 \cdot w_2 = 0{,}7300 \cdot 0{,}6836 + 0{,}6834 \cdot (-0{,}7299) = 0{,}4990 - 0{,}4988 \approx 0. \;\checkmark
$$

**2. Normalização:**
$$
\|w_1\|^2 = 0{,}7300^2 + 0{,}6834^2 = 0{,}5329 + 0{,}4670 \approx 1. \;\checkmark
$$

**3. Geometria.** $w_1$ a $\approx 43°$, $w_2$ a $\approx -47°$ — perpendiculares como esperado, formando um sistema de eixos rotacionado em relação aos canônicos.

> 💡 **Sobre o sinal de $w_2$.** Poderíamos ter escolhido $w_2 = (-0{,}68,\,+0{,}73)$ (vetor oposto) sem mudar nada da PCA — só inverteria o sinal das projeções em PC2. Bibliotecas adotam convenções diferentes; para conta à mão, qualquer escolha é válida.

---

## Passo 6 — Projeção dos pontos na nova base

Cada ponto centralizado $(x_1^c, x_2^c)$ vira coordenadas $(z_1, z_2)$ no sistema $w_1, w_2$:

$$
z_1 = x_1^c \cdot 0{,}7300 + x_2^c \cdot 0{,}6834,
$$

$$
z_2 = x_1^c \cdot 0{,}6836 + x_2^c \cdot (-0{,}7299).
$$

### Tabela de projeções

| ID | $(x_1^c, x_2^c)$ | $z_1$ (PC1) | $z_2$ (PC2) |
|:---:|:---:|---:|---:|
| 1  | $(-5, -3)$ | $-5{,}70$ | $-1{,}23$ |
| 2  | $(-4, -5)$ | $-6{,}34$ | $+0{,}92$ |
| 3  | $(-3, -1)$ | $-2{,}87$ | $-1{,}32$ |
| 4  | $(-2, -4)$ | $-4{,}19$ | $+1{,}55$ |
| 5  | $(-1, -2)$ | $-2{,}10$ | $+0{,}78$ |
| 6  | $(-1, +1)$ | $-0{,}05$ | $-1{,}41$ |
| 7  | $(0, +2)$ | $+1{,}37$ | $-1{,}46$ |
| 8  | $(+1, -1)$ | $+0{,}05$ | $+1{,}41$ |
| 9  | $(+2, +4)$ | $+4{,}19$ | $-1{,}55$ |
| 10 | $(+3, +2)$ | $+3{,}56$ | $+0{,}59$ |
| 11 | $(+4, +5)$ | $+6{,}34$ | $-0{,}92$ |
| 12 | $(+6, +2)$ | $+5{,}75$ | $+2{,}64$ |

**Exemplo de cálculo (linha 1).** Ponto $(1, 3)$ centralizado é $(-5, -3)$:

$$
z_1 = (-5)(0{,}7300) + (-3)(0{,}6834) = -3{,}6500 - 2{,}0502 = -5{,}7002 \approx -5{,}70.
$$

$$
z_2 = (-5)(0{,}6836) + (-3)(-0{,}7299) = -3{,}4180 + 2{,}1897 = -1{,}2283 \approx -1{,}23.
$$

### Três verificações importantes

**1. Médias das projeções são zero.** Como os dados foram centralizados antes da projeção e a média é linear, $\bar{z}_1 = \bar{z}_2 = 0$. Se sua soma não der zero, há erro em algum passo anterior.

**2. Variâncias das projeções recuperam os autovalores.**
$$
\text{Var}(z_1) = \frac{1}{12}\sum_i (z_1^{(i)})^2 \approx 17{,}35 = \lambda_1, \quad \text{Var}(z_2) \approx 1{,}98 = \lambda_2.
$$
**Por isso $\lambda_i$ "é" a variância na direção $w_i$** — é exatamente a propriedade que derivamos no Passo 4 via Lagrange.

**3. As projeções são descorrelacionadas:** $\text{Cov}(z_1, z_2) \approx 0$. Esse é o ponto-chave da PCA — as novas coordenadas são estatisticamente independentes (na medida em que a estrutura linear permite), o que é muito útil para modelos downstream que assumem features descorrelacionadas.

### Redução para $\mathbb{R}^1$

Descartando PC2 (perda de ~10% da variância), o dataset reduzido fica:

$$
\mathbf{z} = (-5{,}70,\ -6{,}34,\ -2{,}87,\ -4{,}19,\ -2{,}10,\ -0{,}05,\ +1{,}37,\ +0{,}05,\ +4{,}19,\ +3{,}56,\ +6{,}34,\ +5{,}75).
$$

Doze pontos em $\mathbb{R}^2$ → doze números em $\mathbb{R}^1$.

### Reconstrução: voltando para $\mathbb{R}^2$

Para reconstruir um ponto a partir apenas de $z_1$:
$$
\hat{x}^{(i)} = z_1^{(i)} \cdot w_1 + \bar{x} = z_1^{(i)} \cdot (0{,}7300,\, 0{,}6834) + (6, 6).
$$

**Exemplo: ponto 1**, $z_1 = -5{,}70$:
$$
\hat{x}^{(1)} = -5{,}70 \cdot (0{,}7300,\, 0{,}6834) + (6, 6) = (-4{,}16,\, -3{,}90) + (6, 6) = (1{,}84,\, 2{,}10).
$$

Original: $(1, 3)$. Reconstrução: $(1{,}84,\, 2{,}10)$. **Erro:** $\sqrt{(1-1{,}84)^2 + (3-2{,}10)^2} \approx 1{,}23 = |z_2^{(1)}|$. Os 10% de variância em PC2 são exatamente o que se perde.

---

## Encerramento — fluxo completo em uma tabela

| Passo | Resultado |
|---|---|
| 1. Médias | $\bar{x}_1 = 6$, $\bar{x}_2 = 6$ |
| 2. Centralização | tabela centralizada + somas $S_{xx}=122$, $S_{yy}=110$, $S_{xy}=92$ |
| 3. Covariância | $\Sigma = \frac{1}{12}\begin{pmatrix} 122 & 92 \\ 92 & 110 \end{pmatrix} \approx \begin{pmatrix} 10{,}17 & 7{,}67 \\ 7{,}67 & 9{,}17 \end{pmatrix}$ |
| 4. Autovalores | $\lambda_1 \approx 17{,}35$, $\lambda_2 \approx 1{,}98$ ($\rho_1 \approx 89{,}74\%$) |
| 5. Autovetores | $w_1 \approx (0{,}73,\,0{,}68)$, $w_2 \approx (0{,}68,\,-0{,}73)$ |
| 6. Projeção | 12 valores $z_1 \in \mathbb{R}$ retendo ~90% da variância |

Em produção, tudo isso é uma linha de código:

```python
from sklearn.decomposition import PCA
import numpy as np

X = np.array([[1,3],[2,1],[3,5],[4,2],[5,4],[5,7],
              [6,8],[7,5],[8,10],[9,8],[10,11],[12,8]])
Z = PCA(n_components=1).fit_transform(X)
```

Mas ter feito à mão uma vez torna concreto o que cada parte da chamada está fazendo — qual é o tamanho de $\Sigma$, por que $\lambda_1$ é a variância em PC1, por que $w_1$ e $w_2$ são ortogonais, qual é o significado de $\rho_1$. Em qualquer dataset real você não vai calcular tudo isso à mão (nem deve), mas vai precisar interpretar os outputs — e a interpretação só é possível se a mecânica estiver internalizada.

---

*Resolução completa em 6 passos. Para o material teórico que justifica cada passo (derivação via Lagrange, $\Sigma$ como transformação linear, conexão com SVD, escolha de $k$), consultar as notas de aula `aula-reducao-dimensionalidade.md`.*
