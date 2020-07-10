---
# Page settings
layout: default
keywords:
comments: false

# Hero section
title: Probability Theory
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

## Probability Theory

### Probability Space

Formally, a **probability space** is defined by the triple $(\Omega, F, P)$, where

* $$\Omega$$ is the **space of possible outcomes** (or **outcome space**)
* $$F \subseteq 2^\Omega$$ (the power set of $$\Omega$$) is the **space of (measurable) events** (or **event space**)
* $$P$$ is the **probability measure** (or **probability distribution**) that maps an event $$E \in F$$ to a real value between $$0$$ and $$1$$ (think of $$P$$ as a function)

Given the outcome space $$\Omega$$, there is some restrictions as to what subset of $$2^\Omega$$ can be considered an event space $$F$$:

* The trivial event $$\Omega$$ and the empty event $$\emptyset$$ is in $$F$$
* The event space $$F$$ is closed under (countable) union, i.e., if $$\alpha, \beta \in F$$, then $$\alpha \cup \beta \in F$$
* The event space $$F$$ is closed under complement, i.e., if $$\alpha \in F$$, then $$(\Omega \setminus \alpha) \in F$$

Given an event space $$F$$, the probability measure $$P$$ must satisfy certain axioms.

* (non-negativity) For all $$\alpha \in F, P(\alpha) \ge 0$$
* (trivial event) $$P(\Omega) = 1$$
* (additivity) For all $$\alpha, \beta \in F$$ and $$\alpha \cap \beta = \emptyset, P(\alpha \cup \beta) = P(\alpha) + P(\beta)$$

### Random Variables

The most important fact about **random variables** is that they are **not** variables. They are actually **functions** that map outcomes (in the outcome space) to real values.

In a sense, random variables allow us to abstract away from the formal notion of event space, as we can define random variables that capture the appropriate events.

Consider the event space of odd or even in dice throw. We could have defined a random variable that takes on value $$1$$ if outcome is odd and $$0$$ otherwise. These type of binary random variables are very common in practice, and are known as **indicator variables**, taking its name from its use to indicate whether a certain event has happened.

So why did we introduce event space? That is because when one studies probability theory (more rigorously) using measure theory, the distinction between outcome space and event space will be very important. In any case, it is good to keep in mind that event space is not always simply the power set of the outcome space.

We will talk mostly about probability with respect to random variables. Random variables allow us to provide a more uniform treatment of probability theory. For notations, the probability of a random variable $$X$$ taking on the value of $$a$$ will be denoted by either

$$ P(X = a) \text{ or } P_X (a) $$

We will also denote the range of a random variable $$X$$ by $$Val(X)$$.

### Distributions, Joint Distributions, and Marginal Distributions

The **distribution** of a variable is formally refers to the probability of a random variable taking on certain values. For notation, we will use $$P(X)$$ to denote the distribution of the random variable $$X$$.

We speak about the distribution of more than one variables at a time. We call these distributions **joint distributions**, as the probability is determined jointly by all the variables involved. We will denote the probability of $$X$$ taking value $$a$$ and $$Y$$ taking value $$b$$ by either the long hand of $$P(X = a, Y = b)$$, or the short hand of $$P_{X,Y} (a, b)$$. We refer to their joint distribution by $$P(X, Y)$$.

Given a joint distribution, say over random variables $$X$$ and $$Y$$, we can talk about the **marginal distribution** of $$X$$ or that of $$Y$$. The marginal distribution refers to the probability distribution of a random variable on its own. To find out the marginal distribution of a random variable, we sum out all the other random variables from the distribution.

$$ P(X) = \sum_{b \in Val(Y)} P(X, Y = b) $$

### Conditional Distributions

**Conditional distributions** are one of the key tools in probability theory for reasoning about uncertainty. They specify the distribution of a random variable when the value of another random variable is known (or more generally, when some event is known to be true).

Formally, conditional probability of $$X = a$$ given $$Y = b$$ is defined as

$$ P(X = a \mid Y = b) = \frac{P(X = a, Y = b)}{P(Y = b)} $$

Note that this is not defined when the probability of $$Y = b$$ is $$0$$.

The idea of conditional probability extends naturally to the case when the distribution of a random variable is conditioned on several variables, namely

$$ P(X = a \mid Y = b, Z = c) = \frac{P(X = a, Y = b, Z = c)}{P(Y = b, Z = c)} $$

As for notations, we write $$P(X \mid Y = b)$$ to denote the distribution of random variable $$X$$ when $$Y = b$$. We may also write $$P(X \mid Y)$$ to denote a set of distributions of $$X$$, one for each of the different values that $$Y$$ can take.

### Independence

In probability theory, **independence** means that the distribution of a random variable does not change on learning the value of another random variable.

Mathematically, a random variable $$X$$ is independent of $$Y$$ when

$$ P(X) = P(X \mid Y) $$

It is easy to verify that if $$X$$ is independent of $$Y$$, then $$Y$$ is also independent of $$X$$. As a notation, we write $$X \perp Y$$ if $$X$$ and $$Y$$ are independent.

An equivalent mathematical statement about the independence of random variables $$X$$ and $$Y$$ is

$$ P(X,Y) = P(X) P(Y) $$

Sometimes we also talk about **conditional independence**, meaning that if we know the value of a random variable (or more generally, a set of random variables), then some other random variables will be independent of each other. Formally, we say $$X$$ and $$Y$$ are conditionally independent given $$Z$$ if

$$ P(X \mid Z) = P(X \mid Y, Z) $$

or, equivalently,

$$ P(X, Y \mid Z) = P(X \mid Z) P(Y \mid Z) $$

### Chain Rule and Bayes Rule

The **Chain Rule** is often used to evaluate the joint probability of some random variables, and is especially useful when there are (conditional) independence across variables.

$$ P(X_1, X_2, \dots, X_n) = P(X_1) P(X_2 \mid X_1) \dots P(X_n \mid X_1, X_2, \dots, X_{n−1}) $$

The **Bayes Rule** allows us to compute the conditional probability $$P(X \mid Y)$$ from $$P(Y \mid X)$$, in a sense inverting the conditions.

$$ P(X \mid Y) = \frac{P(Y \mid X) P(X)}{P(Y)} $$

Extending the Bayes Rule to the case of multiple random variables as following

$$ P(X,Y \mid Z) = \frac{P(Z \mid X,Y) P(X,Y)}{P(Z)} = \frac{P(Y,Z \mid X) P(X)}{P(Z)} $$

$$ P(X \mid Y,Z) = \frac{P(Y \mid X,Z) P(X,Z)}{P(Y,Z)} = \frac{P(Y \mid X,Z) P(X \mid Z) P(Z)}{P(Y \mid Z) P(Z)} = \frac{P(Y \mid X,Z) P(X \mid Z)}{P(Y \mid Z)} $$

Using definition of marginal distribution and chain rule, we can get **Law Of Total Probability**

$$ P(Y) = \sum_{a \in Val(X)} P(X = a, Y) = \sum_{a \in Val(X)} P(Y \mid X = a) P(X = a) $$

## Probability Distribution

In a broad sense, there are two classes of distribution that require seemingly different treatments (these can be unified using measure theory). Namely, **discrete distributions** and **continuous distributions**.

### Discrete Distribution: Probability Mass Function

By a discrete distribution, we mean that the random variable of the underlying distribution can take on only **finitely** many different values (or that the outcome space is finite).

To define a discrete distribution, we can simply enumerate the probability of the random variable taking on each of the possible values. This enumeration is known as the **probability mass function**, as it divides up a unit mass (the total probability) and places them on the different values a random variable can take. This can be extended analogously to joint distributions and conditional distributions.

### Continuous Distribution: Probability Density Function

By a continuous distribution, we mean that the random variable of the underlying distribution can take on **infinitely** many different values (or that the outcome space is infinite).

To define a continuous distribution, we will make use of **probability density function** (PDF). A probability density function, $$f$$, is a non-negative, integrable function such that

$$ \int_{Val(X)} f(x) dx = 1 $$

The probability of a random variable $$X$$ distributed according to a PDF $$F$$ is computed as follows

$$ P(a \leq X \leq b) = \int_{a}^{b} f(x) dx $$

Note that this, in particular, implies that the probability of a continuously distributed random variable taking on any given single value is zero.

To extend the definition of continuous distribution to joint distribution, the probability density function is extended to take multiple arguments, namely,

$$ P(a_1 \le X_1 \le b_1, a_2 \le X_2 \le b_2, \dots, a_n \le X_n \le b_n) = \int_{a_1}^{b_1} \int_{a_2}^{b_2} \dots \int_{a_n}^{b_n} f(x_1, x_2, \dots, x_n) dx_1 dx_2 \dots dx_n $$

To extend the definition of conditional distribution to continuous random variables, we ran into the problem that the probability of a continuous random variable taking on a single value is $$0$$, so bayes rule is not well defined, since the denominator equals $$0$$. To define the conditional distribution of a continuous variable, let $$f(x, y)$$ be the joint distribution of $$X$$ and $$Y$$. We can show that the PDF, $$f(y \mid x)$$, underlying the distribution $$P(Y \mid X)$$ is given by

$$ f(y \mid x) = \frac{f(x,y)}{f(x)} $$

$$ P(a \le Y \le b \mid X = c) = \int_{a}^{b} f(y \mid c) dy = \int_{a}^{b} \frac{f(c,y)}{f(c)} dy $$

Sometimes we will also speak about **cumulative distribution function**. It is a function that gives the probability of a random variable being smaller than some value. A cumulative distribution function $$F$$ is related to the underlying probability density function $$F$$ as follows:

$$ F(b) = P(X \leq b) = \int_{-\infty}^{b} f(x) dx $$

and hence $$F(x) = \int f(x) dx$$ (in the sense of indefinite integral).

## Expectations and Variance

### Expectations

One of the most common operations we perform on a random variable is to compute its **expectation**, also known as its **mean** or **expected value**. The expectation of a random variable, denoted by $$E(X)$$, is given by

$$ E(X) = \sum_{a \in Val(X)} a P(X = a) \text{ or } E(X) = \int_{Val(X)} x f(x) dx $$

When working with indicator variables, a useful identify is the following:

$$ E(X) = P(X = 1) \text{ for indicator variable } X $$

When working with the sums of random variables, one of the most important rule is the **linearity of expectations**. Let $$X_1, X_2, \dots, X_n$$ be (possibly dependent) random variables.

$$ E(X_1 + X_2 + \dots + X_n) = E(X_1) + E(X_2) + \dots + E(X_n) $$

The linearity of expectations is very powerful because there are no restrictions on whether the random variables are independent or not.

When we work on products of random variables, however, there is very little we can say in general. However, when the random variables are independent. Let $$X$$ and $$Y$$ be independent random variables

$$ E(XY) = E(X) E(Y) $$

### Variance

The **variance** of a distribution is a measure of the spread of a distribution. It is defined as follows:

$$ Var(X) = E((X - E(X))^2) $$

The variance of a random variable is often denoted by $\sigma^2$. The reason that this is squared is because we often want to find out $\sigma$, known as the **standard deviation**. The variance and the standard deviation is related by $\sigma = \sqrt{Var(X)}$.

To find out the variance of a random variable $$X$$, it’s often easier to compute the following instead

$$ Var(X) = E(X^2) - (E(X))^2 $$

Note that unlike expectation, variance is not a linear function of a random variable $$X$$. In fact, we can verify that the variance of $$(aX + b)$$ is

$$ Var(aX + b) = a^2 Var(X) $$

If random variables $$X$$ and $$Y$$ are independent, then

$$ Var(X + Y) = Var(X) + Var(Y) \text{ if } X \perp Y $$

Sometimes we also talk about the **covariance** of two random variables. This is a measure of how closely related two random variables are. Its definition is as follows.

$$ Cov(X,Y) = E((X − E(X))(Y − E(Y))) $$

## Important Probability Distributions

### Uniform distribution

The **uniform distribution** has all intervals of the same length on the distribution are equally probable. The corresponding probability density function would be

$$
f(x) =
  \begin{cases}
    \frac{1}{b-a}   & \text{ if } a \leq x \leq b \\
    0               & \text{ otherwise } \\
  \end{cases}
$$

### Bernoulli

The **Bernoulli distribution** is one of the most basic distribution. A random variable distributed according to the Bernoulli distribution can take on two possible values, $$0$$ and $$1$$. It can be specified by a single parameter $$p$$, and by convention we take $$P$$ to be $$P(X = 1)$$. It is often used to indicate whether a experiment is successful or not.

Sometimes it is useful to write the probability distribution of a Bernoulli random variable $$X$$ as follows

$$ P(X) = p^x (1-p)^{1-x} $$

### Binomial

The **binomial distribution** with parameters $$n$$ and $$P$$ is the discrete probability distribution of the number of successes in a sequence of $$n$$ independent experiments. In general, the random variable $$X$$ follows the binomial distribution with parameters $$n \in N$$ and $$p \in [0,1]$$, we write $$X \sim B(n, p)$$. The probability of getting exactly $$k$$ successes in $$n$$ experiments is given by the probability mass function:

$$ P(X = k) = C_n^k p^k (1-p)^{n-k} $$

$$ C_n^k = \frac{n!}{k! (n-k)!} $$

### Poisson

The **Poisson distribution** is a discrete probability distribution that expresses the probability of a given number of events occurring in a fixed interval of time or space if these events occur with a known constant rate $$\lambda$$ and independently of the time since the last event.

$$ P(X = k) = \frac{exp(-\lambda) \lambda^k}{k!} $$

The mean value of a Poisson random variable is $$\lambda$$, and its variance is also $\lambda$.

### Gaussian

The **Gaussian distribution**, also known as the **normal distribution**, is one of the most versatile distributions in probability theory, and appears in a wide variety of contexts. It can be used to approximate the binomial distribution when the number of experiments is large, or the Poisson distribution when the average arrival rate is high. It is also related to the Law of Large Numbers. For many problems, we will also often assume that when noise in the system is Gaussian distributed.

The Gaussian distribution is determined by two parameters: the mean $$\mu$$ and the variance $$\sigma$$. The probability density function is given by

$$ f(x) = \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{(x - \mu)^2}{2\sigma^2} \bigg) $$

We will sometimes work with multi-variate Gaussian distributions. A $$k$$-dimensional multi-variate Gaussian distribution is parametrized by $$(\mu, \Sigma)$$, where $$\mu$$ is now a vector of means in $$R^k$$, and $$\Sigma$$ is the covariance matrix in $$R^{k \times k}$$, in other words, $$\Sigma_{ii} = Var(X_i)$$ and $$\Sigma_{ij} = Cov(X_i, X_j)$$. The probability density function is now defined over vectors of input, given by

$$ f(X) = \frac{1}{\sqrt{(2\pi)^k \vert \Sigma \vert}} exp \bigg( -\frac{1}{2} (X - \mu)^T \Sigma^{-1} (X - \mu) \bigg) $$

When the covariances are zero, the determinant $$\vert \Sigma \vert$$ will simply be the product of the variances, and the inverse $$\Sigma^{−1}$$ can be found by taking the inverse of the diagonal entries of $$\Sigma$$.
