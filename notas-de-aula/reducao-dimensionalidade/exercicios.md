# Exercícios de Fixação — PCA

> **Antes de começar:** revise as 6 etapas do algoritmo em [pca.md](pca.md) (seção 6) e o exemplo passo-a-passo em [exercicio-resolvido.md](exercicio-resolvido.md). Ao final de cada exercício, valide seus números com NumPy puro (`np.linalg.eigh`, sem `sklearn`).

---

## Exercício 1 — Fácil (2D) ⭐

### Contexto

Um professor coletou as notas finais de **6 alunos** em duas disciplinas correlacionadas: **Cálculo** ($x_1$) e **Álgebra Linear** ($x_2$):

| Aluno | Cálculo ($x_1$) | Álgebra Linear ($x_2$) |
|:---:|:---:|:---:|
| 1 | 5 | 6  |
| 2 | 6 | 7  |
| 3 | 7 | 8  |
| 4 | 8 | 9  |
| 5 | 9 | 8  |
| 6 | 7 | 10 |

O professor deseja construir um **índice composto de desempenho matemático** que sintetize as duas notas em um único escalar por aluno.

### Pede-se

1. Calcule o vetor de médias $\bar{x}$ e construa $X_c = X - \bar{x}$.
2. Compute os desvios-padrão $s_1$ e $s_2$. **É necessário padronizar?** Justifique.
3. Construa a matriz de covariância $\Sigma \in \mathbb{R}^{2 \times 2}$.
4. Resolva $\det(\Sigma - \lambda I) = 0$ e obtenha os dois autovalores $\lambda_1 > \lambda_2$.
5. Calcule a **variância explicada** por PC1 e PC2.
6. Determine os autovetores normalizados $v_1$ e $v_2$ e verifique que são ortogonais ($v_1 \cdot v_2 = 0$).
7. Construa $W_1 = [v_1]$ e calcule a projeção $Z = X_c W_1 \in \mathbb{R}^{6 \times 1}$.
8. **Interprete:** o que significa um aluno com $z_i > 0$? E com $z_i < 0$? Ordene os alunos pelo índice composto.

### Sanity checks (não pule!)

* As colunas de $X_c$ devem somar zero.
* $\lambda_1 + \lambda_2 = \text{tr}(\Sigma) = \sigma_{11} + \sigma_{22}$.
* $\lambda_1 \cdot \lambda_2 = \det(\Sigma)$.

---

## Exercício 2 — Desafiador (3D) ⭐⭐⭐

### Contexto

Uma equipe de marketing digital coletou três métricas para **5 campanhas online** distintas:

* $x_1$: **investimento diário** (em centenas de reais)
* $x_2$: **tempo médio de leitura** do anúncio (em segundos)
* $x_3$: **número de cliques** no CTA (call-to-action)

| Campanha | $x_1$ | $x_2$ | $x_3$ |
|:---:|:---:|:---:|:---:|
| 1 | 0 | 1 | 1 |
| 2 | 1 | 2 | 3 |
| 3 | 3 | 3 | 4 |
| 4 | 5 | 4 | 6 |
| 5 | 6 | 5 | 6 |

A meta é encontrar um único **"índice de engajamento"** que sintetize as três métricas em um escalar por campanha.

---

### 📐 Como calcular a matriz de covariância $\Sigma \in \mathbb{R}^{3 \times 3}$

Em três dimensões, $\Sigma$ tem **9 entradas**, mas, por ser simétrica, basta calcular **6 valores únicos**: as **3 variâncias** da diagonal e as **3 covariâncias** acima da diagonal.

Após centralizar os dados ($X_c$), monte $\Sigma$ entrada por entrada:

$$
\Sigma = \frac{1}{n-1} \begin{pmatrix}
\sum (x^c_{i1})^2 & \sum x^c_{i1} x^c_{i2} & \sum x^c_{i1} x^c_{i3} \\
\sum x^c_{i1} x^c_{i2} & \sum (x^c_{i2})^2 & \sum x^c_{i2} x^c_{i3} \\
\sum x^c_{i1} x^c_{i3} & \sum x^c_{i2} x^c_{i3} & \sum (x^c_{i3})^2
\end{pmatrix}
$$

> 💡 **Atalho operacional:** monte primeiro a matriz inteira $S = X_c^\top X_c$ e divida tudo por $n-1$ **só no final**. Isso evita arrastar frações pelo cálculo intermediário.

---

### 📐 Como calcular $\det(\Sigma - \lambda I)$ para matriz $3 \times 3$

A equação característica produz agora um **polinômio cúbico** em $\lambda$, cuja forma compacta é:

$$\det(\Sigma - \lambda I) = -\lambda^3 + \text{tr}(\Sigma)\,\lambda^2 - M\,\lambda + \det(\Sigma) = 0$$

Multiplicando por $-1$ para deixar mônica:

$$\boxed{\;\lambda^3 - \text{tr}(\Sigma)\,\lambda^2 + M\,\lambda - \det(\Sigma) = 0\;}$$

Onde cada coeficiente é calculado da seguinte maneira:

**(a) Traço $\text{tr}(\Sigma)$** — soma da diagonal:

$$\text{tr}(\Sigma) = \sigma_{11} + \sigma_{22} + \sigma_{33}$$

**(b) Soma $M$ dos menores principais $2 \times 2$** — para cada $i$, remova a $i$-ésima linha e coluna e calcule o determinante $2\times 2$ resultante; depois some os três:

$$
M = \det\begin{pmatrix}\sigma_{22} & \sigma_{23} \\ \sigma_{23} & \sigma_{33}\end{pmatrix}
  + \det\begin{pmatrix}\sigma_{11} & \sigma_{13} \\ \sigma_{13} & \sigma_{33}\end{pmatrix}
  + \det\begin{pmatrix}\sigma_{11} & \sigma_{12} \\ \sigma_{12} & \sigma_{22}\end{pmatrix}
$$

**(c) Determinante $\det(\Sigma)$** — duas técnicas equivalentes:

**🔹 Regra de Sarrus** (exclusiva para $3 \times 3$):

$$
\det(\Sigma) = \sigma_{11}\sigma_{22}\sigma_{33} \;+\; \sigma_{12}\sigma_{23}\sigma_{31} \;+\; \sigma_{13}\sigma_{21}\sigma_{32}
\;-\; \sigma_{13}\sigma_{22}\sigma_{31} \;-\; \sigma_{11}\sigma_{23}\sigma_{32} \;-\; \sigma_{12}\sigma_{21}\sigma_{33}
$$

Visualmente: copia-se as duas primeiras colunas à direita da matriz, soma-se os produtos das **3 diagonais descendo** e subtraem-se os produtos das **3 diagonais subindo**.

**🔹 Expansão por cofatores** (generaliza para qualquer $n \times n$):

$$
\det(\Sigma) = \sigma_{11}\,\det\begin{pmatrix}\sigma_{22} & \sigma_{23} \\ \sigma_{32} & \sigma_{33}\end{pmatrix}
- \sigma_{12}\,\det\begin{pmatrix}\sigma_{21} & \sigma_{23} \\ \sigma_{31} & \sigma_{33}\end{pmatrix}
+ \sigma_{13}\,\det\begin{pmatrix}\sigma_{21} & \sigma_{22} \\ \sigma_{31} & \sigma_{32}\end{pmatrix}
$$

> ⚠️ **Cuidado com os sinais!** A expansão por cofatores alterna $(+, -, +)$ ao longo da primeira linha. Errar isso é o erro #1 do exercício.

---

### 📐 Como resolver a cúbica $\lambda^3 - \text{tr}(\Sigma)\lambda^2 + M\lambda - \det(\Sigma) = 0$

Diferente do caso $2\times 2$ (onde Bhaskara resolve direto), uma cúbica exige estratégia:

1. **Teste o Teorema das Raízes Racionais primeiro.** Se $\lambda^* = p/q$ é raiz racional, então $p$ divide $\det(\Sigma)$ e $q$ divide o coeficiente líder ($1$, aqui). Logo, basta **testar os divisores inteiros de $\det(\Sigma)$**.

2. **Se encontrar uma raiz $\lambda_1$,** fatore $(\lambda - \lambda_1)$ usando **divisão sintética (Briot-Ruffini)**, reduzindo a um polinômio quadrático que se resolve com Bhaskara.

3. **Se nenhuma raiz racional existir** (o caso comum em dados reais), use **métodos numéricos**: bisseção, Newton-Raphson, ou diretamente `np.linalg.eigvalsh(Sigma)` (que internamente aplica QR-iteration sobre $\Sigma$, evitando o polinômio).

> 💡 **Insight pedagógico:** em 2D, sempre teremos Bhaskara. Em 3D, a fórmula fechada existe (Cardano, século XVI) mas é desconfortável. Por isso, **3D é o limite prático da PCA manual** — a partir de $d \geq 4$ é NumPy.

---

### Pede-se

1. Calcule o vetor de médias $\bar{x}$ e construa $X_c \in \mathbb{R}^{5 \times 3}$.
2. Calcule a matriz $S = X_c^\top X_c$ e, em seguida, $\Sigma = S/(n-1)$.
3. **Calcule $\det(\Sigma)$ por dois métodos: Sarrus E expansão por cofatores.** Confirme que os dois resultados coincidem.
4. Calcule o traço $\text{tr}(\Sigma)$ e a soma $M$ dos três menores principais $2 \times 2$.
5. Monte o polinômio característico cúbico e tente o **Teorema das Raízes Racionais**.
6. Resolva numericamente os três autovalores e ordene $\lambda_1 \geq \lambda_2 \geq \lambda_3$.
7. Calcule $\text{VE}_1, \text{VE}_2, \text{VE}_3$. **Quantas componentes você manteria pelo critério $\geq 95\%$?**
8. Para PC1, resolva $(\Sigma - \lambda_1 I)\,v_1 = 0$ obtendo o autovetor normalizado.
9. Projete as 5 campanhas sobre $v_1$ — **qual a campanha com maior índice de engajamento?**

### Sanity checks

| Verificação | Por quê |
|---|---|
| Cada coluna de $X_c$ soma zero | Centralização correta |
| $\text{tr}(\Sigma) = \lambda_1 + \lambda_2 + \lambda_3$ | Conservação da variância total |
| $\det(\Sigma) = \lambda_1 \cdot \lambda_2 \cdot \lambda_3$ | Detecta erros de Sarrus/cofatores |
| $v_1^\top v_2 = v_1^\top v_3 = v_2^\top v_3 = 0$ | $\Sigma$ simétrica $\Rightarrow$ autovetores ortogonais |
| $\text{Var}(Z_{\cdot,j}) = \lambda_j$ | Projeção preserva a variância de cada PC |

### Desafio extra (opcional)

Reaplique a PCA **padronizando** os dados antes (dividindo cada coluna por $s_j$). Os autovalores mudam? Por quê? Em qual cenário do mundo real (escalas heterogêneas, p. ex. R\$ vs. segundos vs. quantidade) você preferiria a versão padronizada?

---

*Bom estudo! Lembre-se: o objetivo da PCA não é só comprimir — é **revelar a estrutura latente** dos seus dados.*
