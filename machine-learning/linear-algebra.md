---
# Page settings
layout: default
keywords:
comments: false

# Hero section
title: Linear Algebra
description:

# Micro navigation
micro_nav: true

# Page navigation
page_nav:
    prev:
        content:
        url: '#'
    next:
        content:
        url: '#'
---

## Matrix

### Symmetric Matrices

A square matrix $$A \in R^{n \times n}$$ is **symmetric** if $$A = A^T$$. It is **anti-symmetric** if $$A = -A^T$$. It is easy to show that for any matrix $$A \in R^{n \times n}$$, the matrix $$A + A^T$$ is symmetric and the matrix $$A - A^T$$ is anti-symmetric. From this it follows that any square matrix $$A \in R^{n \times n}$$ can be represented as a sum of a symmetric matrix and an anti-symmetric matrix, since

$$ A = \frac{1}{2} (A + A^T) + \frac{1}{2} (A - A^T) $$

and the first matrix on the right is symmetric, while the second is anti-symmetric. It is common to denote the set of all symmetric matrices of size $$n$$ as $$S^n$$, so that $$A \in S^n$$ means that $$A$$ is a symmetric $$n \times n$$ matrix.

### The Trace

The **trace** of a square matrix $$A \in R^{n \times n}$$, denoted $$tr(A)$$, is the sum of diagonal elements in the matrix:

$$ trA = \sum_{i=1}^{n} A_{ii} $$

The trace has the following properties :

* For $$A \in R^{n \times n}$$, $$tr(A) = tr(A^T)$$.
* For $$A, B \in R^{n \times n}$$, $$tr(A + B) = trA + trB$$.
* For $$A \in R^{n \times n}$$, $$t \in R, tr(tA) = t \text{ } trA$$.
* For $$A, B$$ such that $$AB$$ is square, $$trAB = trBA$$.
* For $$A, B, C$$ such that $$ABC$$ is square, $$trABC = trBCA = trCAB$$, and so on for the product of more matrices.

### Norms

A **norm** of a vector $$\Vert x \Vert$$ is informally measure of the "length" of the vector. For example, we have the commonly used Euclidean or $$\ell_2$$ norm,

$$ {\Vert x \Vert}_2 = \bigg( \sum_{i=1}^{n} x_{i}^2 \bigg) ^ \frac{1}{2} $$

Note that $${\Vert x \Vert}_2^2 = x^T x$$

More formally, a norm is any function $$f : R^n \to R$$ that satisfies 4 properties:

* non-negativity : For all $$x \in R^n, f(x) \ge 0$$.
* definiteness : $$f(x) = 0$$ if and only if $$x = 0$$.
* homogeneity : For all $$x \in R^n, t \in R, f(tx) = \vert t \vert \text{ } f(x)$$.
* triangle inequality : For all $$x, y \in R^n, f(x + y) \le f(x) + f(y)$$.

Other examples of norms are the $$\ell_1$$ norm,

$$ {\Vert x \Vert}_1 = \sum_{i=1}^{n} \vert x_{i} \vert $$

and the $$\ell_\infty$$ norm,

$$ {\Vert x \Vert}_\infty = max_i \vert x_i \vert $$

In fact, all three norms presented so far are examples of the family of $$\ell_p$$ norms, which are parameterized by a real number $$p \ge 1$$, and defined as

$$ {\Vert x \Vert}_p = \bigg( \sum_{i=1}^{n} {\vert x_{i} \vert}^p \bigg) ^ \frac{1}{p} $$

Norms can also be defined for matrices, such as the **Frobenius norm**,

$$ {\Vert A \Vert}_F = \bigg( \sum_{i=1}^{m} \sum_{j=1}^{n} A_{ij}^2 \bigg) ^ \frac{1}{2} = \sqrt{tr(A^T A)} $$

### Linear Independence and Rank

A set of vectors $$[x_1, x_2, \dots, x_n]$$ is said to be **linearly independent** if no vector can be represented as a linear combination of the remaining vectors. Conversely, a vector which can be represented as a linear combination of the remaining vectors is said to be **linearly dependent**. For example, if

$$ x_n = \sum_{i=1}^{n-1} \alpha_i x_i $$

for some $$[\alpha_1, \dots, \alpha_{n-1}]$$ then $$x_n$$ is dependent on $$[x_1, \dots, x_{n-1}]$$. otherwise, it is independent of $$[x_1, \dots, x_{n-1}]$$.

The **column rank** of a matrix $$A$$ is the largest number of columns of $$A$$ that constitute linearly independent set. In the same way, the **row rank** is the largest number of rows of $$A$$ that constitute a linearly independent set.

It is a basic fact of linear algebra, that for any matrix $$A, columnrank(A) = rowrank(A)$$, and so this quantity is simply refereed to as the **rank** of $$A$$, denoted as $$rank(A)$$. The following are some basic properties of the rank:

* For $$A \in R^{m \times n}, rank(A) \le min(m,n)$$. If $$rank(A) = min(m, n)$$, then $$A$$ is said to be **full rank**.
* For $$A \in R^{m \times n}, rank(A) = rank(A^T)$$.
* For $$A \in R^{m \times n} , B \in R^{n \times p} , rank(AB) \le min(rank(A), rank(B))$$.
* For $$A, B \in R^{m \times n}, rank(A + B) \le rank(A) + rank(B)$$.

### The Inverse

The **inverse** of a square matrix $$A \in R^{n \times n}$$ is denoted $$A^{-1}$$, and is the unique matrix such that

$$ A^{-1} A = I = A A^{-1} $$

It turns out that $$A^{-1}$$ may not exist for some matrices $$A$$. we say $$A$$ is **invertible** or **non-singular** if $$A^{-1}$$ exists and **non-invertible** or **singular** otherwise. One condition for invertibility we already know is that $$A^{-1}$$ exists if and only if $$A$$ is full rank.

The following are properties of the inverse; all assume that $$A, B \in R^{n \times n}$$ are non-singular:

* $$(A^{-1})^{-1} = A$$
* If $$Ax = b$$, we can multiply by $$A^{-1}$$ on both sides to obtain $$x = A^{-1} b$$.
* $$(AB)^{-1} = B^{-1} A^{-1}$$
* $$(A^{-1})^T = (A^T)^{-1}$$. For this reason this matrix is often denoted $$A^{-T}$$.

### Orthonormal Matrices

Two vectors $$x, y \in R^n$$ are **orthogonal** if $$x^T y = 0$$. A vector $$x \in R^n$$ is **normalized** if $${\Vert x \Vert}_2 = 1$$. A square matrix $$U \in R^{n \times n}$$ is **orthonormal** if all its columns are orthogonal to each other and are normalized.

It follows immediately from the definition of orthogonality and normality that

$$ U^T U = I = U U^T $$

In other words, the inverse of an orthonormal matrix is its transpose.

Another nice property of orthonormal matrices is that operating on a vector with an orthogonal matrix will not change its Euclidean norm, i.e.,

$$ {\Vert Ux \Vert}_2 = {\Vert x \Vert}_2 $$

for any $$x \in R^n, U \in R^{n \times n}$$ orthonormal.

### Columnspace and Nullspace of a Matrix

The **span** of a set of vectors $$[x_1, x_2, \dots, x_n]$$ is the set of all vectors that can be expressed as a linear combination of $$[x_1, x_2, \dots, x_n]$$. That is,

$$ span(x_1, x_2, \dots, x_n) = [ v : v = \sum_{i=1}^{n} \alpha_i x_i, \alpha_i \in R ] $$

It can be shown that if $$[x_1, x_2, \dots, x_n]$$ is a set of $$n$$ linearly independent vectors, where each $$x_i \in R^n$$, then $$span(x_1, x_2, \dots, x_n) = R^n$$. In other words, any vector $$v \in R^n$$ can be written as a linear combination of $$x_1$$ through $$x_n$$.

The **projection** of a vector $$y \in R^m$$ onto the span of $$[x_1, x_2, \dots, x_n]$$ (here we assume $$x_i \in R^m$$) is the vector $$v \in span(x_1, \dots, x_n)$$, such that $$v$$ as close as possible to $$y$$, as measured by the Euclidean norm $${\Vert v - y \Vert}_2$$. We denote the projection as $$Proj(y; x_1, \dots, x_n)$$ and can define it formally as

$$ Proj(y; x_1, \dots, x_n) = argmin_{v \in span(x_1, \dots, x_n)} {\Vert v - y \Vert}_2 $$

The **Columnspace** of a matrix $$A \in R^{m \times n}$$, denoted $$R(A)$$, is the the span of the columns of $$A$$. In other words,

$$ R(A) = [v \in R^m : v = Ax, x \in R^n] $$

Making a few technical assumptions (namely that $$A$$ is full rank and that $$n < m$$), the projection of a vector $$y \in R^m$$ onto the columnspace of $$A$$ is given by,

$$ Proj(y; A) = argmin_{v \in R(A)} {\Vert v - y \Vert}_2 = A (A^T A)^{-1} A^T y $$

When $$A$$ contains only a single column, $$a \in R^m$$, this gives the special case for a projection of a vector on to a line:

$$ Proj(y; a) = \frac{a a^T}{a^T a} y $$

The **nullspace** of a matrix $$A \in R^{m \times n}$$, denoted $$N(A)$$ is the set of all vectors that equal $$0$$ when multiplied by $$A$$, i.e.,

$$ N(A) = [x \in R^n : Ax = 0] $$

Note that vectors in $$R(A)$$ are of size $$m$$, while vectors in the $$N(A)$$ are of size $$n$$, so vectors in $$R(A^T)$$ and $$N(A)$$ are both in $$R^n$$. In fact, we can say much more. It turns out that

$$ [w : w = u + v, u \in R(A^T), v \in N(A)] = R^n \text{ and } R(A^T) \cap N(A) = \emptyset $$

In other words, $$R(A^T)$$ and $$N(A)$$ are disjoint subsets that together span the entire space of $$R^n$$. Sets of this type are called **orthogonal complements**, and we denote this $$R(A^T) = N(A)^\perp$$

### The Determinant

The **determinant** of a square matrix $$A \in R^{n \times n}$$, is a function $$det : R^{n \times n} \to R$$, and is denoted $$\vert A \vert$$ or $$detA$$.

properties of the determinant:

* The determinant of the identity is 1, $$\vert I \vert = 1$$
* Given a matrix $$A \in R^{n \times n}$$, if we multiply a single row in $$A$$ by a scalar $$t \in R$$, then the determinant of the new matrix is $$t \text{ } \vert A \vert$$
* If we exchange any two rows $$a_i^T$$ and $$a_j^T$$ of $$A$$, then the determinant of the new matrix is $$- \vert A \vert$$
* For $$A \in R^{n \times n}, \vert A \vert = \vert A^T \vert$$.
* For $$A, B \in R^{n \times n}, \vert AB \vert = \vert A \vert \vert B \vert$$.
* For $$A \in R^{n \times n}, \vert A \vert = 0$$ if and only if $$A$$ is singular (i.e., non-invertible).
* For $$A \in R^{n \times n}$$ and $$A$$ non-singular, $$\vert A^{-1} \vert = \frac{1}{\vert A \vert}$$.

Before given the general definition for the determinant, we define, for $$A \in R^{n \times n}, A_{ij} \in R^{(n-1) \times (n-1)}$$ to be the matrix that results from deleting the $$i$$ th row and $$j$$ th column from $$A$$. The general recursive formula for the determinant is

$$
\vert A \vert = \sum_{i=1}^{n} (-1)^{i+j} a_{ij} \vert A_{ij} \vert, \text{ for any } j \in (1, \dots, n) \\
\vert A \vert = \sum_{j=1}^{n} (-1)^{i+j} a_{ij} \vert A_{ij} \vert, \text{ for any } i \in (1, \dots, n) \\
$$

The **classical adjoint** of a matrix $$A \in R^{n \times n}$$, is denoted $$adj(A)$$, and defined as

$$ adj(A) \in R^{n \times n}, {(adj(A))}_{ij} = (-1)^{i+j} \vert A_{ji} \vert $$

(note the switch in the indices $$A_{ji}$$). It can be shown that for any nonsingular $$A \in R^{n \times n}$$,

$$ A^{-1} = \frac{1}{\vert A \vert} adj(A) $$

### Quadratic Forms and Positive Semidefinite Matrices

Given a matrix square $$A \in R^{n \times n}$$ and a vector $$x \in R^n$$, the scalar value $$x^T A x$$ is called a **quadratic form**. Written explicitly, we see that

$$ x^T A x = \sum_{i=1}^{n} \sum_{j=1}^{n} x_i A_{ij} x_j $$

Note that,

$$ x^T A x = (x^T A x)^T = x^T A^T x = x^T \bigg( \frac{1}{2} A + \frac{1}{2} A^T \bigg) x $$

i.e., only the symmetric part of $$A$$ contributes to the quadratic form. For this reason, we often implicitly assume that the matrices appearing in a quadratic form are symmetric.

We give the following definitions:

* A symmetric matrix $$A \in S^n$$ is **positive definite** (PD) if for all non-zero vectors $$x \in R^n, x^T A x > 0$$. This is usually denoted $$A \succ 0$$ (or just $$A > 0$$), and often times the set of all positive definite matrices is denoted $$S_{+ +}^n$$.

* A symmetric matrix $$A \in S^n$$ is **positive semidefinite** (PSD) if for all vectors $$x^T A x \ge 0$$. This is writen $$A \succeq 0$$ (or just $$A \ge 0$$), and the set of all positive semidefinite matrices is often denoted $$S_{+}^n$$.

* Likewise, a symmetric matrix $$A \in S^n$$ is **negative definite** (ND), denoted $$A \prec 0$$ (or just $$A < 0$$) if for all non-zero $$x \in R^n , x^T A x < 0$$.

* Similarly, a symmetric matrix $$A \in S^n$$ is **negative semidefinite** (NSD), denoted $$A \preceq 0$$ (or just $$A \le 0$$) if for all $$x \in R^n , x^T A x \le 0$$.

* Finally, a symmetric matrix $$A \in S^n$$ is **indefinite**, if it is neither positive semidefinite nor negative semidefinite - i.e., if there exists $$x_1, x_2 \in R^n$$ such that $$x_1^T A x_1 > 0$$ and $$x_2^T A x_2 < 0$$.

It should be obvious that if $$A$$ is positive definite, then $$-A$$ is negative definite and vice versa. Likewise, if $$A$$ is positive semidefinite then $$-A$$ is negative semidefinite and vice versa. If $$A$$ is indefinite, then so is $$-A$$. It can also be shown that positive definite and negative definite matrices are always invertible.

Finally, there is one type of positive definite matrix that comes up frequently, and so deserves some special mention. Given any matrix $$A \in R^{m \times n}$$ (not necessarily symmetric or even square), the matrix $$G = A^T A$$ (sometimes called a **Gram matrix**) is always positive semidefinite. Further, if $$m \ge n$$ (and we assume for convenience that $$A$$ is full rank), then $$G = A^T A$$ is positive definite.

### Eigenvalues and Eigenvectors

Given a square matrix $$A \in R^{n \times n}$$, we say that $$\lambda \in C$$ is an **eigenvalue** of $$A$$ and $$x \in C^n$$ is the corresponding **eigenvector** if

$$ A x = \lambda x, x \neq 0 $$

Intuitively, this definition means that multiplying $$A$$ by the vector $$x$$ results in a new vector that points in the same direction as $$x$$, but scaled by a factor $$\lambda$$. Also note that for any eigenvector $$x \in C^n$$, and scalar $$t \in C, A(cx) = cAx = c \lambda x = \lambda (cx)$$, so $$cx$$ is also an eigenvector. For this reason when we talk about the eigenvector associated with $$\lambda$$, we usually assume that the eigenvector is normalized to have length $$1$$.

We can rewrite the equation above to state that $$(\lambda,x)$$ is an eigenvalue-eigenvector pair of $$A$$ if,

$$ (\lambda I - A)x = 0, x \neq 0 $$

But $$(\lambda I - A)x = 0$$ has a non-zero solution to $$x$$ if and only if $$(\lambda I - A)$$ has a non-empty nullspace, which is only the case if $$(\lambda I - A)$$ is singular, i.e.,

$$ \vert (\lambda I - A) \vert = 0 $$

We can now use the definition of the determinant to expand this expression into a very large polynomial in $$\lambda$$, where $$\lambda$$ will have maximum degree $$n$$. We then find the $$n$$ (possibly complex) roots of this polynomial to find the $$n$$ eigenvalues $$\lambda_1, \dots, \lambda_n$$. To find the eigenvector corresponding to the eigenvalue $$\lambda_i$$, we simply solve the linear equation $$(\lambda_i I - A)x = 0$$.

The following are properties of eigenvalues and eigenvectors (in all cases assume $$A \in R^{n \times n}$$ has eigenvalues $$\lambda_i, \dots, \lambda_n$$ and associated eigenvectors $$x_1, \dots, x_n$$):

* The trace of a $$A$$ is equal to the sum of its eigenvalues,

$$ trA = \sum_{i=1}^{n} \lambda_i $$

* The determinant of $$A$$ is equal to the product of its eigenvalues,

$$ \vert A \vert = \prod_{i=1}^{n} \lambda_i $$

* The rank of $$A$$ is equal to the number of non-zero eigenvalues of $$A$$.

* If $$A$$ is non-singular then $$1 / \lambda_i$$ is an eigenvalue of $$A^{-1}$$ with associated eigenvector $$x_i$$, i.e., $$A^{-1} x_i = (1 / \lambda_i) x_i$$.

* The eigenvalues of a diagonal matrix $$D = diag(d_1, \dots, d_n)$$ are just the diagonal entries $$d_1, \dots, d_n$$.

We can write all the eigenvector equations simultaneously as

$$ A X = X \Lambda $$

where the columns of $$X \in R^{n \times n}$$ are the eigenvectors of $$A$$ and $$\Lambda$$ is a diagonal matrix whose entries are the eigenvalues of $$A$$, i.e.,

$$
X \in R^{n \times n} =
\left(
\begin{array}{c}
  \vert & \vert &       & \vert \\
  x_1   &  x_2  & \dots & x_n   \\
  \vert & \vert &       & \vert \\
\end{array}
\right)
$$

$$ \Lambda = diag(\lambda_1, \dots, \lambda_n) $$

If the eigenvectors of $$A$$ are linearly independent, then the matrix $$X$$ will be invertible, so $$A = X \Lambda X^{-1}$$. Matrix that can be written in this form is called **diagonalizable**.

### Eigenvalues and Eigenvectors of Symmetric Matrices

Two remarkable properties come about when we look at the eigenvalues and eigenvectors of a symmetric matrix $$A \in S^n$$. First, it can be shown that all the eigenvalues of $$A$$ are real. Secondly, the eigenvectors of $$A$$ are orthonormal, i.e., the matrix $$X$$ defined above is an orthonormal matrix (for this reason, we denote the matrix of eigenvectors as $$U$$ in this case). We can therefore represent $$A$$ as $$A = U \Lambda U^T$$, remembering from above that the inverse of an orthonormal matrix is just its transpose.

Using this, we can show that the definiteness of a matrix depends entirely on the sign of its eigenvalues. Suppose $$A \in S^n = U \Lambda U^T$$. Then

$$ x^T A x = x^T U \Lambda U^T x = y^T \Lambda y = \sum_{i=1}^{n} λ_i y_i^2 $$

where $$y = U^T x$$ (and since $$U$$ is full rank, any vector $$y \in R^n$$ can be represented in this form). Because $$y_i^2$$ is always positive, the sign of this expression depends entirely on the $$\lambda_i$$’s. If all $$\lambda_i > 0$$, then the matrix is positive definite; if all $$\lambda_i \ge 0$$, it is positive semidefinite. Likewise, if all $$\lambda_i < 0$$ or $$\lambda_i \le 0$$, then $$A$$ is negative definite or negative semidefinite respectively. Finally, if $$A$$ has both positive and negative eigenvalues, it is indefinite.

An application where eigenvalues and eigenvectors come up frequently is in maximizing some function of a matrix. In particular, for a matrix $$A \in S^n$$, consider the following maximization problem.

$$ max_{x \in R^n} \text{ } x^T A x \text{ subject to } \Vert x \Vert_2^2 = 1 $$

i.e., we want to find the vector (of norm $$1$$) which maximizes the quadratic form. Assuming the eigenvalues are ordered as $$\lambda_1 \ge \lambda_2 \ge \dots \ge \lambda_n$$, the optimal $$X$$ for this optimization problem is $$x_1$$, the eigenvector corresponding to $$\lambda_1$$. In this case the maximal value of the quadratic form is $$\lambda_1$$. Similarly, the optimal solution to the minimization problem,

$$ min_{x \in R^n} \text{ } x^T A x \text{ subject to } \Vert x \Vert_2^2 = 1 $$

is $$x_n$$, the eigenvector corresponding to $$\lambda_n$$, and the minimal value is $$\lambda_n$$.

## Matrix Calculus

### The Gradient

Suppose that $$f : R^{m \times n} \to R$$ is a function that takes as input a matrix $$A$$ of size $$m \times n$$ and returns a real value. Then the **gradient** of $$f$$ (with respect to $$A \in R^{m \times n}$$) is the matrix of partial derivatives, defined as:

$$
\nabla_A f(A) \in R^{m \times n} =
\left(
\begin{array}{c}
  \frac{\partial f(A)}{\partial A_{11}} & \frac{\partial f(A)}{\partial A_{12}} & \cdots & \frac{\partial f(A)}{\partial A_{1n}} \\
  \frac{\partial f(A)}{\partial A_{21}} & \frac{\partial f(A)}{\partial A_{22}} & \cdots & \frac{\partial f(A)}{\partial A_{2n}} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  \frac{\partial f(A)}{\partial A_{m1}} & \frac{\partial f(A)}{\partial A_{m2}} & \cdots & \frac{\partial f(A)}{\partial A_{mn}} \\
\end{array}
\right)
$$

i.e., an $$m \times n$$ matrix with

$$ {(\nabla_A f(A))}_{ij} = \frac{\partial f(A)}{\partial A_{ij}} $$

Note that the size of $$\nabla_A f(A)$$ is always the same as the size of $$A$$. So if, in particular, $$A$$ is just a vector $$x \in R^n$$,

$$
\nabla_x f(x) =
\left(
\begin{array}{c}
  \frac{\partial f(x)}{\partial x_1} \\
  \frac{\partial f(x)}{\partial x_2} \\
  \vdots \\
  \frac{\partial f(x)}{\partial x_n} \\
\end{array}
\right)
$$

It is very important to remember that the gradient of a function is only defined if the function is real-valued. We can not, for example, take the gradient of $$Ax, A \in R^{n \times n}$$ with respect to $$x$$, since this quantity is vector-valued.

It follows directly from the equivalent properties of partial derivatives that:

* $$\nabla_x (f(x) + g(x)) = \nabla_x f(x) + \nabla_x g(x)$$

* $$\text{For } t \in R, \nabla_x (t f(x)) = t \text{ } \nabla_x f(x)$$

### The Hessian

Suppose that $$f : R^n \to R$$ is a function that takes a vector in $$R^n$$ and returns a real number. Then the **Hessian** matrix with respect to $$x$$, written $$\nabla_x^2 f(x)$$ or simply as $$H$$ is the $$n \times n$$ matrix of partial derivatives,

$$
\nabla_x^2 f(x) \in R^{n \times n} =
\left(
\begin{array}{c}
  \frac{\partial^2 f(x)}{\partial x_1^2} & \frac{\partial^2 f(x)}{\partial x_1 \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_1 \partial x_n} \\
  \frac{\partial^2 f(x)}{\partial x_2 \partial x_1} & \frac{\partial^2 f(x)}{\partial x_2^2} & \cdots & \frac{\partial^2 f(x)}{\partial x_2 \partial x_n} \\
  \vdots  & \vdots  & \ddots & \vdots  \\
  \frac{\partial^2 f(x)}{\partial x_n \partial x_1} & \frac{\partial^2 f(x)}{\partial x_n \partial x_2} & \cdots & \frac{\partial^2 f(x)}{\partial x_n^2} \\
\end{array}
\right)
$$

In other words, $$\nabla_x^2 f(x) \in R^{n \times n}$$, with

$$ {(\nabla_x^2 f(x))}_{ij} = \frac{\partial^2 f(x)}{\partial x_i \partial x_j} $$

Note that the Hessian is always symmetric, since

$$ \frac{\partial^2 f(x)}{\partial x_i \partial x_j} = \frac{\partial^2 f(x)}{\partial x_j \partial x_i} $$

Similar to the gradient, the Hessian is defined only when $$f(x)$$ is real-valued.

It is natural to think of the gradient as the analogue of the first derivative for functions of vectors, and the Hessian as the analogue of the second derivative (and the symbols we use also suggest this relation). This intuition is generally correct, but there a few caveats to keep in mind.

First, for real-valued functions of one variable $$f : R \to R$$, it is a basic definition that the second derivative is the derivative of the first derivative, i.e.,

$$ \frac{\partial^2 f(x)}{\partial x^2} = \frac{\partial}{\partial x} \frac{\partial}{\partial x} f(x) $$

However, for functions of a vector, the gradient of the function is a vector, and we cannot take the gradient of a vector - i.e.,

$$
\nabla_x \nabla_x f(x) = \nabla_x
\left(
\begin{array}{c}
  \frac{\partial f(x)}{\partial x_1} \\
  \frac{\partial f(x)}{\partial x_2} \\
  \vdots                             \\
  \frac{\partial f(x)}{\partial x_n} \\
\end{array}
\right)
$$

and this expression is not defined. Therefore, it is not the case that the Hessian is the gradient of the gradient. However, this is almost true, in the following sense: If we look at the $$i$$th entry of the gradient $${(\nabla_x f(x))}_i = \partial f(x) / \partial x_i$$, and take the gradient with respect to $$X$$ we get

$$
\nabla_x \frac{\partial f(x)}{\partial x_i} =
\left(
\begin{array}{c}
  \frac{\partial^2 f(x)}{\partial x_i \partial x_1} \\
  \frac{\partial^2 f(x)}{\partial x_i \partial x_2} \\
  \vdots                                            \\
  \frac{\partial^2 f(x)}{\partial x_i \partial x_n} \\
\end{array}
\right)
$$

which is the $$i$$ th column (or row) of the Hessian. Therefore,

$$
\nabla_x^2 f(x) =
\left(
\begin{array}{c}
  \nabla_x {(\nabla_x f(x))}_1, \nabla_x {(\nabla_x f(x))}_2, \dots, \nabla_x {(\nabla_x f(x))}_n
\end{array}
\right)
$$

If we don’t mind being a little bit sloppy we can say that (essentially) $$\nabla_x^2 f(x) = \nabla_x (\nabla_x f(x))^T$$, so long as we understand that this really means taking the gradient of each entry of $$(\nabla_x f(x))^T$$, not the gradient of the whole vector.

### Gradients and Hessians of Quadratic and Linear Functions

Now let’s try to determine the gradient and Hessian matrices for a few simple functions.

For $$x \in R^n$$, let $$f(x) = b^T x$$ for some known vector $$b \in R^n$$. then

$$ f(x) = \sum_{i=1}^{n} b_i x_i $$

so

$$ \frac{\partial f(x)}{\partial x_k} = \frac{\partial}{\partial x_k} \sum_{i=1}^{n} b_i x_i = b_k $$

From this we can easily see that $\nabla_x b^T x = b$. This should be compared to the analogous situation in single variable calculus, where $\partial / (\partial x) ax = a$.

Now consider the quadratic function $$f(x) = x^T A x$$ for $$A \in S^n$$. Remember that

$$ f(x) = \sum_{i=1}^{n} \sum_{j=1}^{n} x_i A_{ij} x_j $$

so

$$
\frac{\partial f(x)}{\partial x_k} = \frac{\partial}{\partial x_k} \sum_{i=1}^{n} \sum_{j=1}^{n} x_i A_{ij} x_j = \sum_{i=1}^{n} x_i A_{ik} + \sum_{j=1}^{n} A_{kj} x_j = 2 \sum_{j=1}^{n} A_{kj} x_j
$$

where the last equality follows since $$A$$ is symmetric (which we can safely assume, since it is appearing in a quadratic form). Note that the $$k$$th entry of $$\nabla_x f(x)$$ is just the inner product of the $$k$$th row of $$A$$ and $$x$$. Therefore, $$\nabla_x x^T A x = 2Ax$$. Again, this should remind you of the analogous fact in single-variable calculus, that $$\partial / (\partial x) a x^2 = 2ax$$

Finally, lets look at the Hessian of the quadratic function $$f(x) = x^T A x$$ (it should be obvious that the Hessian of a linear function $$b^T x$$ is zero). This is even easier than determining the gradient of the function, since

$$ \frac{\partial^2 f(x)}{\partial x_k \partial x_l} = \frac{\partial^2}{\partial x_k \partial x_l} \sum_{i=1}^{n} \sum_{j=1}^{n} x_i A_{ij} x_j = A_{kl} + A_{lk} = 2A_{kl} $$

Therefore, it should be clear that $$\nabla_x^2 x^T A x = 2A$$, which should be entirely expected (and again analogous to the single-variable fact that $$\partial^2 / (\partial x^2) ax^2 = 2a$$).

To recap, if $$A$$ symmetric

* $$\nabla_x b^T x = b$$

* $$\nabla_x x^T A x = 2Ax$$

* $$\nabla_x^2 x^T A x = 2A$$

### Least Squares

Suppose we are given matrices $$A \in R^{m \times n}$$ (for simplicity we assume $$A$$ is full rank) and a vector $$b \in R^m$$ such that $$b \notin R(A)$$. In this situation we will not be able to find a vector $$x \in R^n$$, such that $$Ax = b$$, so instead we want to find a vector $$X$$ such that $Ax$ is as close as possible to $$b$$, as measured by the square of the Euclidean norm $${\Vert Ax-b \Vert}_2^2$$.

Using the fact that $${\Vert x \Vert}_2^2 = x^T x$$, we have

$$ {\Vert Ax-b \Vert}_2^2 = (Ax - b)^T (Ax - b) = x^T A^T Ax - 2 b^T Ax + b^T b $$

Taking the gradient with respect to $$x$$ we have

$$ \nabla_x (x^T A^T Ax - 2 b^T Ax + b^T b) = \nabla_x x^T A^T Ax - \nabla_x 2 b^T Ax + \nabla_x  b^T b = 2 A^T Ax - 2 A^T b $$

Setting this last expression equal to zero and solving for $$x$$ gives the normal equations

$$ x = (A^T A)^{-1} A^T b $$

### Gradients of the Determinant

Now lets consider gradient of the determinant respect to a matrix, namely for $$A \in R^{n \times n}$$, we want to find $$\nabla_A \vert A \vert$$. From the definition of determinants that

$$ \vert A \vert = \sum_{i=1}^{n} (-1)^{i+j} a_{ij} \vert A_{ij} \vert \text{ for any } j \in (1, \dots, n) $$

so

$$ \frac{\partial}{\partial A_{kl}} \vert A \vert = \frac{\partial}{\partial A_{kl}} \sum_{i=1}^{n} (-1)^{i+j} a_{ij} \vert A_{ij} \vert = (-1)^{k+l} \vert A_{kl} \vert = {(adj(A))}_{lk} $$

From this it immediately follows from the properties of the adjoint that

$$ \nabla_A \vert A \vert = (adj(A))^T = \vert A \vert A^{-T} $$

Now lets consider the function $$f : S_{+ +}^n \to R, f(A) = log \vert A \vert$$. Note that we have to restrict the domain of $$f$$ to be the positive definite matrices, since this ensures that $$\vert A \vert > 0$$, so that the log of $$\vert A \vert$$ is a real number. In this case we can use the chain rule (nothing fancy, just the ordinary chain rule from single-variable calculus) to see that

$$ \frac{\partial log \vert A \vert}{\partial A_{ij}} = \frac{\partial log \vert A \vert}{\partial \vert A \vert} \frac{\partial \vert A \vert}{\partial A_{ij}} = \frac{1}{\vert A \vert} \frac{\partial \vert A \vert}{\partial A_{ij}} $$

From this is should be obvious that

$$ \nabla_A log \vert A \vert = \frac{1}{\vert A \vert} \nabla_A \vert A \vert = A^{-1} $$

where we can drop the transpose in the last expression because $$A$$ is symmetric. Note the similarity to the single-valued case , where $$\partial / (\partial x) logx = 1 / x$$.

### Eigenvalues as Optimization

Finally, we use matrix calculus to solve an optimization problem in a way that leads directly to eigenvalue/eigenvector analysis. Consider the following equality constrained optimization problem:

$$ max_{x \in R^n} \text{ } x^T A x \text{ subject to } {\Vert x \Vert}_2^2 = 1 $$

for a symmetric matrix $$A \in S^n$$. A standard way of solving optimization problems with equality constraints is by forming the **Lagrangian**. The Lagrangian in this case can be given by

$$ L(x, \lambda) = x^T A x  + \lambda (1 - x^T x) = x^T A x + \lambda - \lambda x^T x $$

where $$\lambda$$ is called the Lagrange multiplier. It can be established that for $$x ^ \ast$$ to be a optimal point to the problem, the gradient of the Lagrangian has to be zero at $$x ^ \ast$$. That is,

$$ \nabla_x L(x, \lambda) = \nabla_x (x^T A x + \lambda - \lambda x^T x) = 2Ax - 2 \lambda x = 0 $$

Notice that this is just the linear equation $$A x = \lambda x$$. This shows that the only points which can possibly maximize (or minimize) $$x^T A x$$ assuming $$x^T x = 1$$ are the eigenvectors of $$A$$.
