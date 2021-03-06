---
# Page settings
layout: default
keywords:
comments: false

# Hero section
title: Machine Learning
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

# Introduction to Machine Learning

## What Is Machine Learning

A field of computer science that gives computers the ability to learn without being explicitly programmed. - Arthur Samuel (1959)

A computer program is said to learn from experience E with respect to some class of tasks T and performance measure P if its performance at tasks in T, as measured by P, improves with experience E. - Tom Mitchell (1998)

## Supervised vs Unsupervised

Machine learning tasks are typically classified into two broad categories, depending on whether there is a learning feedback available to a learning system:

* Supervised learning: The computer is presented with example inputs and their desired outputs, given by a "teacher", and the goal is to learn a general rule that maps inputs to outputs.

    - Regression: modeling the relationship between inputs and outputs.

    - Classification: inputs are divided into two or more classes, and the learner must produce a model that assigns unseen inputs to one or more (multi-label classification) of these classes.

* Unsupervised learning: No outputs are given to the learning algorithm, leaving it on its own to find structure in its inputs.

    - Clustering: a set of inputs is to be divided into groups. The groups are not known beforehand.

# Supervised Learning

To establish notation for future use, we'll use $$x^{(i)}$$ to denote the "input" variables, also called input **features**, and $$y^{(i)}$$ to denote the "output" or **target** variable that we are trying to predict. A pair $$(x^{(i)}, y^{(i)})$$ is called a **training example**, and the dataset that we'll be using to learn - a list of $$m$$ training examples $$[(x^{(i)}, y^{(i)}); i = 1, \dots, m]$$ - is called a **training set**. We will also use $$X$$ denote the space of input values, and $$Y$$ the space of output values.

To describe the supervised learning problem slightly more formally, our goal is, given a training set, to learn a function $$h : X \to Y$$ so that $$h(x)$$ is a "good" predictor for the corresponding value of $$y$$. This function $$h$$ is called a **hypothesis**.

When the target variable that we're trying to predict is continuous, we call the learning problem a **regression** problem. When $$Y$$ can take on only a small number of discrete values, we call it a **classification** problem.

## Linear Regression

To perform linear regression, we must decide how we're going to represent functions/hypotheses $$h$$ in a computer. As an initial choice, lets say we decide to approximate $$Y$$ as a linear function of $$x$$:

$$ h_{\theta}(x) = \theta_0 + \theta_1 x_1 + \theta_2 x_2 + \dots + \theta_n x_n $$

Here, the $$\theta_i$$ 's are the **parameters** (also called **weights**) parameterizing the space of linear functions mapping from $$X$$ to $$Y$$. To simplify our notation, we also introduce the convention of letting $$x_0 = 1$$ (this is the **intercept term**), so that

$$ h_{\theta}(x) = \sum_{i=0}^{n} \theta_i x_i = \theta^T x $$

where on the right-hand side above we are viewing $$\theta$$ and $$X$$ both as vectors, and here $$n$$ is the number of input variables (not counting $$x_0$$).

Now, given a training set, how do we pick, or learn, the parameters $$\theta$$. One reasonable method seems to be to make $$h(x)$$ close to $$y$$, at least for the training examples we have. To formalize this, we will define a function that measures, for each value of the $$\theta$$ 's, how close the $$h_{\theta}(x^{(i)})$$ 's are to the corresponding $$y^{(i)}$$ 's. We define the **cost function**:

$$ J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \Big( h_{\theta}(x^{(i)}) - y^{(i)} \Big)^2 $$

### Least Mean Squares

We want to choose $$\theta$$ so as to minimize $$J(\theta)$$. To do so, lets use a search algorithm that starts with some "initial guess" for $$\theta$$, and that repeatedly changes $$\theta$$ to make $$J(\theta)$$ smaller, until hopefully we converge to a value of $$\theta$$ that minimizes $$J(\theta)$$. Specifically, lets consider the **gradient descent** algorithm, which starts with some initial $$\theta$$, and repeatedly performs the update:

$$ \theta_j = \theta_j - \alpha \frac{\partial}{\partial \theta_j} J(\theta) $$

(This update is simultaneously performed for all values of $$j = 0, \dots, n$$.)

Here, $$\alpha$$ is called the **learning rate**. This is a very natural algorithm that repeatedly takes a step in the direction of steepest decrease of $$J$$.

$$ \frac{\partial}{\partial \theta_j} J(\theta) = \sum_{i=1}^{m} \Big( h_{\theta}(x^{(i)}) - y^{(i)} \Big) x_j^{(i)} $$

Repeat until convergence, for every $$j$$

$$ \theta_j = \theta_j + \alpha \sum_{i=1}^{m} \Big( y^{(i)} - h_{\theta}(x^{(i)}) \Big) x_j^{(i)} $$

The rule is called the **LMS** update rule (LMS stands for "least mean squares"), and is also known as the **Widrow-Hoff** learning rule, the magnitude of the update is proportional to the **error term** $$\Big( y^{(i)} - h_\theta (x^{(i)}) \Big)$$.

This method looks at every example in the entire training set on every step, and is called **batch gradient descent**. Note that, while gradient descent can be a local minima in general, the optimization problem for linear regression has only one global (assuming the learning rate $$\alpha$$ is not too large), and no other local, optima; Indeed, $$J$$ is a convex quadratic function.

the contours of a quadratic function and the trajectory taken by batch gradient descent

{% include image.html description="contours" image="machine-learning/contours.png" caption="false"%}

There is an alternative to batch gradient descent that also works very well. Consider the following algorithm:

Loop for $$i=1$$ to $$m$$, for every $$j$$

$$ \theta_j = \theta_j + \alpha \Big( y^{(i)} - h_{\theta}(x^{(i)}) \Big) x_j^{(i)} $$

In this algorithm, we repeatedly run through the training set, and each time we encounter a training example, we update the parameters according to the gradient of the error with respect to that single training example only. This algorithm is called **stochastic gradient descent (SGD)**. Whereas batch gradient descent has to scan through the entire training set before taking a single step - a costly operation if $$m$$ is large - stochastic gradient descent can start making progress right away, and continues to make progress with each example it looks at. Often, stochastic gradient descent gets $$\theta$$ "close" to the minimum much faster than batch gradient descent. For these reasons, particularly when the training set is large, stochastic gradient descent is often preferred over batch gradient descent. **Mini-batch stochastic gradient descent (mini-batch SGD)** is a compromise between **full-batch** iteration and SGD. A mini-batch is typically between 10 and 1,000 examples, chosen at random. Mini-batch SGD reduces the amount of noisy gradients in SGD but is still more efficient than full-batch.

Note however that stochastic gradient descent may never "converge" to the minimum, and the parameters $$\theta$$ will keep oscillating around the minimum of $$J(\theta)$$; but in practice most of the values near the minimum will be reasonably good approximations to the true minimum. While it is more common to run stochastic gradient descent as we have described it and with a fixed learning rate $$\alpha$$, by slowly letting the learning rate $$\alpha$$ decrease to zero as the algorithm runs, it is also possible to ensure that the parameters will converge to the global minimum rather then merely oscillate around the minimum.

### Normal Equations

Gradient descent gives one way of minimizing $$J$$. Lets discuss a second way of doing so, this time performing the minimization explicitly and without resorting to an iterative algorithm. In this method, we will minimize $$J$$ by explicitly taking its derivatives with respect to the $$\theta_j$$ 's, and setting them to zero.

Giving a training set, define the **design matrix** $$X$$ to be the $$m \times n$$ matrix, actually $$m \times (n + 1)$$ if we include the intercept term, that contains the training examples input values in its rows; Also, let $$Y$$ be the $$m$$-dimensional vector containing all the target values from the training set;

Then

$$ J(\theta) = \frac{1}{2} \sum_{i=1}^{m} \Big( h_{\theta}(x^{(i)}) - y^{(i)} \Big)^2 = \frac{1}{2} {\Vert X \theta - Y \Vert}_2^2 = \frac{1}{2} (X \theta - Y)^T (X \theta - Y) $$

$$ \nabla_\theta J(\theta) = X^T X \theta - X^T Y $$

To minimize $$J$$, we set its derivatives to zero, and obtain the **normal equations**:

$$ X^T X \theta = X^T Y $$

Thus, the value of $$\theta$$ that minimizes $$J(\theta)$$ is given in closed form by the equation

$$ \theta = (X^T X)^{-1} X^T Y $$

### Probabilistic Interpretation

Let us assume that the target variables and the inputs are related via the equation

$$ y^{(i)} = \theta^T x^{(i)} + \epsilon^{(i)} $$

where $$\epsilon^{(i)}$$ is an error term that captures either unmodeled effects, or random noise. Let us further assume that the $$\epsilon^{(i)}$$ are distributed IID (independently and identically distributed) according to a Gaussian distribution with mean zero and variance $$\sigma^2$$. We can write this assumption as $$\epsilon^{(i)} \sim N(0, \sigma^2)$$. I.e., the density of $$\epsilon^{(i)}$$ is given by

$$ p(\epsilon^{(i)}) = \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{(\epsilon^{(i)})^2}{2\sigma^2} \bigg) $$

Then

$$ p(y^{(i)} \mid x^{(i)};\theta) = \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \bigg) $$

The notation $$p(y^{(i)} \mid x^{(i)};\theta)$$ indicates that this is the distribution of $$y^{(i)}$$ given $$x^{(i)}$$ and parameterized by $$\theta$$. We can also write the distribution of $$y^{(i)}$$ as $$(y^{(i)} \mid x^{(i)};\theta) \sim N(\theta^T x^{(i)}, \sigma^2)$$

Given the design matrix $$X$$ and outputs $$Y$$, then the **likelihood** function of $$\theta$$

$$ L(\theta) = L(\theta;X,Y) = P(Y \mid X;\theta) = \prod_{i=1}^{m} p(y^{(i)} \mid x^{(i)};\theta) = \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \bigg) $$

The principal of **maximum likelihood** says that we should choose $$\theta$$ so as to make the data as high probability as possible. I.e., we should choose $$\theta$$ to maximize $$L(\theta)$$. Instead of maximizing $$L(\theta)$$, we maximize the **log likelihood** $$\ell(\theta)$$:

$$ \begin{align*}
\ell(\theta) = log L(\theta) &= log \prod_{i=1}^{m} \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \bigg) \\
&= \sum_{i=1}^{m} log \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{(y^{(i)} - \theta^T x^{(i)})^2}{2\sigma^2} \bigg) \\
&= m log \frac{1}{\sqrt{2\pi} \sigma} - \frac{1}{\sigma^2} \frac{1}{2} \sum_{i=1}^{m} \Big( y^{(i)} - \theta^T x^{(i)} \Big)^2 \\
\end{align*} $$

Hence, maximizing $$\ell(\theta)$$ gives the same answer as minimizing

$$ \frac{1}{2} \sum_{i=1}^{m} \Big( y^{(i)} - \theta^T x^{(i)} \Big)^2 $$

which we recognize to be $$J(\theta)$$, our original least-squares cost function.

### Locally Weighted Linear Regression

{% include image.html description="LWR" image="machine-learning/LWR.png" caption="false"%}

The choice of features is important to ensuring good performance of a learning algorithm. The **locally weighted linear regression (LWR)** algorithm which, assuming there is sufficient training data, makes the choice of features less critical.

The locally weighted linear regression algorithm does the following:

* Fit $$\theta$$ to minimize $$\displaystyle\sum_{i=1}^{m} w^{(i)} \Big( y^{(i)} - \theta^T x^{(i)} \Big)^2$$
* Output $$\theta^T x$$

Here, the $$w^{(i)}$$ 's are non-negative valued **weights**. Intuitively, if $$w^{(i)}$$ is large for a particular value of $$i$$, then in picking $$\theta$$, we'll try hard to make $$\Big( y^{(i)} - \theta^T x^{(i)} \Big)^2$$ small. If $$w^{(i)}$$ is small, then the $$\Big( y^{(i)} - \theta^T x^{(i)} \Big)^2$$ error term will be pretty much ignored in the fit.

A fairly standard choice for the weights is

$$ w^{(i)} = exp \bigg( -\frac{(x^{(i)} - x)^2}{2 \tau^2} \bigg) $$

If $$X$$ is vector-valued, this is generalized for an appropriate choice of $$\tau$$ or $$\sum$$.

$$ w^{(i)} = exp \bigg( -\frac{(x^{(i)} - x)^T (x^{(i)} - x)}{2 \tau^2} \bigg) \text{ or } w^{(i)} = exp \bigg( -\frac{(x^{(i)} - x)^T \sum^{-1} (x^{(i)} - x)}{2} \bigg) $$

Note that the weights depend on the particular point $$X$$ at which we're trying to evaluate $$x$$. Moreover, if $$\vert x^{(i)} - x \vert$$ is small, then $$w^{(i)}$$ is close to $$1$$; and if $$\vert x^{(i)} - x \vert$$ is large, then $$w^{(i)}$$ is close to $$0$$. Hence, $$\theta$$ is chosen giving a much higher weight to the errors on training examples close to the query point $$x$$. The parameter $$\tau$$ controls how quickly the weight of a training example falls off with distance of its $$x^{(i)}$$ from the query point $$x$$. $$\tau$$ is called the **bandwidth parameter**.

In linear regression once we've fit the $$\theta_i$$ 's and stored them away, we no longer need to keep the training data around to make future predictions. In contrast, to make predictions using locally weighted linear regression, we need to keep the entire training set around.

## Classification and Logistic Regression

Classification problem is just like the regression problem, except that the values $$Y$$ we now want to predict take on only a small number of discrete values. For now, we will focus on the **binary classification** problem in which $$Y$$ can take on only two values, $$0$$ and $$1$$. $$0$$ is also called the **negative class**, and $$1$$ the **positive class**, and they are sometimes also denoted by the symbols "-" and "+". Given $$x^{(i)}$$, the corresponding $$y^{(i)}$$ is also called the **label** for the training example.

### Logistic Regression

In logistic regression we will choose

$$ \begin{align*}
h_\theta(x) &= g(z) = \frac{1}{1 + e^{-z}} \text{ with } z = \theta^T x \\
&= g(\theta^T x) = \frac{1}{1 + e^{-\theta^T x}}
\end{align*} $$

$$g(z)$$ is called the **logistic function** or the **sigmoid function**. The plot of $$g(z)$$:

{% include image.html description="logistic-function" image="machine-learning/logistic-function.png" caption="true"%}

$$g(z)$$ tends towards $$1$$ as $$z \to +\infty$$, and $$g(z)$$ tends towards $$0$$ as $$z \to -\infty$$. $$g(z)$$ is always bounded between $$0$$ and $$1$$.

a useful property of the derivative of the sigmoid function

$$ g'(z) = g(z)(1 - g(z)) $$

assume that

$$ \begin{align*}
P(y=1 \mid x;\theta) &= h_\theta(x) \\
P(y=0 \mid x;\theta) &= 1 - h_\theta(x) \\
\end{align*} $$

so that

$$ P(y \mid x;\theta) = (h_\theta(x))^y (1 - h_\theta(x))^{1-y} $$

so the likelihood of the parameters as

$$ \begin{align*}
L(\theta)
&= P(Y \mid X;\theta) \\
&= \prod_{i=1}^{m} P(y^{(i)} \mid x^{(i)};\theta) \\
&= \prod_{i=1}^{m} (h_\theta(x^{(i)}))^{y^{(i)}} (1 - h_\theta(x^{(i)}))^{1-y^{(i)}} \\
\end{align*} $$

to maximize the log likelihood

$$ \begin{align*}
\ell (\theta)
&= log \text{ } L(\theta) \\
&= \sum_{i=1}^{m} \bigg( y^{(i)} log \text{ } h_\theta(x^{(i)}) + (1 - y^{(i)}) log(1 - h_\theta(x^{(i)})) \bigg) \\
\end{align*} $$

we can use **gradient ascent**:

$$ \theta = \theta + \alpha \nabla_\theta \ell (\theta) $$

take derivatives to derive the stochastic gradient ascent rule:

$$ \begin{align*}
\frac{\partial}{\partial \theta_j} \ell(\theta)
&= \bigg( y \frac{1}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)} \bigg) \frac{\partial}{\partial \theta_j} g(\theta^T x) \\
&= \bigg( y \frac{1}{g(\theta^T x)} - (1 - y) \frac{1}{1 - g(\theta^T x)} \bigg) g(\theta^T x) (1 - g(\theta^T x)) \frac{\partial}{\partial \theta_j} \theta^T x \\
&= \bigg( y(1 - g(\theta^T x)) - (1 - y)g(\theta^T x) \bigg) x_j \\
&= (y - h_\theta(x)) x_j \\
\end{align*} $$

so that:

$$ \theta_j = \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)} $$

### Perceptron Learning Algorithm

Output values that are either $$0$$ or $$1$$ or exactly, change the definition of $$g$$ to be the threshold function:

$$
g(z) =
\begin{cases}
  1 & \text{ if } z \geq 0 \\
  0 & \text{ if } z < 0 \\
\end{cases}
$$

let $$h_\theta (x) = g(\theta^T x)$$ and use the update rule

$$ \theta_j = \theta_j + \alpha (y^{(i)} - h_\theta(x^{(i)})) x_j^{(i)} $$

then we have the **perceptron learning algorithm**.

Note however that even though the perceptron may be cosmetically similar to the other algorithms we talked about, it is actually a very different type of algorithm than logistic regression and least squares linear regression; in particular, it is difficult to endow the perceptron's predictions with meaningful probabilistic interpretations, or derive the perceptron as a maximum likelihood estimation algorithm.

### Newton Method For Maximizing Log Likelihood

**Newton method** for finding a zero of a function. Specifically, suppose we have some function $$f : R \to R$$, and we wish to find a value of $$\theta$$ so that $$f(\theta) = 0$$. Here, $$\theta \in R$$ is a real number. Newton method performs the following update:

$$ \theta = \theta - \frac{f(\theta)}{f'(\theta)} $$

Approximating the function $$f$$ via a linear function that is tangent to $$f$$ at the current guess $\theta$, solving for where that linear function equals to zero, and letting the next guess for $$\theta$$ be where that linear function is zero.

Picture of the Newton method in action:

{% include image.html description="newton-method" image="machine-learning/newton-method.png" caption="true"%}

The maxima of $$\ell$$ correspond to points where its derivative $\ell'(\theta)$ is zero. So, by letting $$f(\theta) = \ell'(\theta)$$, we can use the same algorithm to maximize $$\ell$$, and we obtain update rule:

$$ \theta = \theta - \frac{\ell'(\theta)}{\ell''(\theta)} $$

In logistic regression setting, $$\theta$$ is vector-valued, so we need to generalize Newton method to this setting. The generalization of Newton method to this multidimensional setting also called the **Newton-Raphson method** is given by

$$ \theta = \theta - H^{-1} \nabla_\theta \ell(\theta) $$

Here, $$\nabla_\theta \ell(\theta)$$ is, as usual, the vector of partial derivatives of $$\ell(\theta)$$ with respect to the $$\theta_i$$ 's; and $$h$$ is a $$(n+1) \times (n+1)$$ matrix that include the intercept term called the **Hessian**, whose entries are given by

$$ H_{ij} = \frac{\partial^2 \ell(\theta)}{\partial \theta_i \partial \theta_j} $$

Newton-Raphson method typically faster convergence than batch gradient descent, and requires many fewer iterations to get very close to the minimum. One iteration of Newton-Raphson method can, however, be more expensive than one iteration of batch gradient descent, since it requires finding and inverting a $$(n+1) \times (n+1)$$ Hessian matrix; but so long as $$n$$ is not too large, it is usually much faster overall. When Newton-Raphson method is applied to maximize the logistic regression log likelihood function $$\ell(\theta)$$, the resulting method is also called **Fisher scoring**.

## Generalized Linear Models

In the regression, we had $$(y^{(i)} \mid x^{(i)};\theta) \sim N(\mu, \sigma^2)$$, and in the classification, we had $$(y^{(i)} \mid x^{(i)};\theta) \sim Bernoulli(\phi)$$, the definitions of $$\mu$$ and $$\phi$$ as functions of $$X$$ and $$\theta$$. Both of these methods are special cases of a broader family of models, called **Generalized Linear Models (GLMs)**.

### The Exponential Family

To defining exponential family distributions. We say that a class of distributions is in the exponential family if it can be written in the form

$$ p(y;\eta) = b(y) exp \Big( \eta^T T(y) - a(\eta) \Big) $$

$$\eta$$ is called the **natural parameter** (also called the **canonical parameter**) of the distribution; $$T(y)$$ is the **sufficient statistic** (for the distributions we consider, it will often be the case that $$T(y) = y$$); and $$a(\eta)$$ is the **log partition function**. The quantity $$e^{-a(\eta)}$$ essentially plays the role of a normalization constant, that makes sure the distribution $$p(y;\eta)$$ sums/integrates over $$Y$$ to $$1$$.

A fixed choice of $$T$$, $$a$$ and $$b$$ defines a family (or set) of distributions that is parameterized by $$\eta$$; as we vary $$\eta$$, we then get different distributions within this family.

We now show that the Bernoulli and the Gaussian distributions are exponential family distributions.

The Bernoulli distribution with mean $\phi$, written $$Bernoulli(\phi)$$, specifies a distribution over $$Y$$ $$\in$$ $$[0, 1]$$, so that $$P(y = 1; \phi) = \phi; P(y = 0; \phi) = 1 - \phi$$. As we varying $$\phi$$, we obtain Bernoulli distributions with different means. We now show that this class of Bernoulli distributions, ones obtained by varying $$\phi$$, is in the exponential family; i.e., that there is a choice of $$T$$, $$a$$ and $$b$$ so that the exponential family becomes exactly the class of Bernoulli distributions.

Write the Bernoulli distribution as:

$$ \begin{align*}
P(y;\phi)
&= \phi^y (1 - \phi)^{1 - y} \\
&= exp \Big( y log\phi + (1 - y) log(1 - \phi) \Big) \\
&= exp \Big( \big( log(\frac{\phi}{1 - \phi}) \big) y + log(1 - \phi) \Big) \\
\end{align*} $$

Thus, the natural parameter is given by $$\eta^T = \eta = log \big( \phi/(1 - \phi) \big)$$. Invert this definition for $$\eta$$ by solving for $$\phi$$ in terms of $$\eta$$, we obtain $$\phi = 1/(1 + e^{-\eta})$$. To complete the formulation of the Bernoulli distribution as an exponential family distribution, we also have

$$ \begin{align*}
   T(y) &= y \\
a(\eta) &= -log(1 - \phi) \\
        &= log(1 + e^\eta) \\
   b(y) &= 1 \\
\end{align*} $$

The Gaussian distribution $$N(\mu, \sigma^2)$$ with mean $$\mu$$ and variance $$\sigma^2$$:

$$ \begin{align*}
p(y;\eta) = p(y;\mu,\sigma)
&= \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{(y - \mu)^2}{2\sigma^2} \bigg) \\
&= \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{y^2}{2\sigma^2} \bigg) exp \bigg( -\frac{\mu^2-2\mu y}{2\sigma^2} \bigg) \\
&= \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{y^2}{2\sigma^2} \bigg) exp \bigg( \frac{\mu}{\sigma}\frac{y}{\sigma} - \frac{\mu^2}{2\sigma^2} \bigg) \\
\end{align*} $$

so the Gaussian is in the exponential family, with

$$ \begin{align*}
\eta^T = \eta &= \frac{\mu}{\sigma} \text{ invert } \mu = \sigma \eta \\
         T(y) &= \frac{y}{\sigma} \\
      a(\eta) &= \frac{\mu^2}{2\sigma^2} = \frac{\eta^2}{2} \\
         b(y) &= \frac{1}{\sqrt{2\pi} \sigma} exp \bigg( -\frac{y^2}{2\sigma^2} \bigg) \\
\end{align*} $$

### Constructing GLMs

Make the following assumptions/design choices about the conditional distribution of $$Y$$ given $$x$$:

* $$(y \mid x; \theta) \sim ExponentialFamily(\eta)$$. I.e., given $$X$$ and $$\theta$$, the distribution of $$Y$$ follows some exponential family distribution, with parameter $$\eta$$.

* Given $$x$$, our goal is to predict the expected value of $$T(y)$$. So the prediction $$h(x)$$ output by our learned hypothesis $$h$$ to satisfy $$h(x) = E[T(y) \mid x]$$.

* The natural parameter $$\eta$$ and the inputs $$X$$ are related linearly: $$\eta = \theta^T x$$. Or, if $$\eta$$ is vector-valued, then $$\eta_i = \theta_i^T x$$.

#### Constructing GLMs - Ordinary Least Squares

To show that ordinary least squares is a special case of the GLM family of models, consider the setting where the target variable $$Y$$ (also called the **response variable** in GLM terminology) is continuous, and we model the conditional distribution of $$Y$$ given $$X$$ as a Gaussian $$N(\mu, \sigma^2)$$. So, we let the $$ExponentialFamily(\eta)$$ distribution be the Gaussian distribution. In the formulation of the Gaussian as an exponential family distribution, $$\mu / \sigma = \eta$$. So:

$$ \begin{align*}
h_\theta (x)
&= E[T(y) \mid x;\theta] = E[\frac{y}{\sigma} \mid x;\theta] \\
&= \frac{\mu}{\sigma} = \eta = \theta^T x \\
\end{align*} $$

#### Constructing GLMs - Logistic Regression

Consider logistic regression. Here we are interested in binary classification, so $$y \in [0, 1]$$. Given that $$Y$$ is binary-valued, it therefore seems natural to choose the Bernoulli family of distributions to model the conditional distribution of $$Y$$ given $$x$$. In our formulation of the Bernoulli distribution as an exponential family distribution, we had $$\phi = 1/(1 + e^{-\eta})$$. Furthermore, note that if $$(y \mid x;\theta) \sim Bernoulli(\phi)$$, then $$E[T(y) \mid x;\theta] = E[y \mid x;\theta] = \phi$$. So:

$$ \begin{align*}
h_\theta (x)
&= E[T(y) \mid x;\theta] = E[y \mid x;\theta] \\
&= \phi \\
&= \frac{1}{1 + e^{-\eta}} \\
&= \frac{1}{1 + e^{-\theta^T x}} \\
\end{align*} $$

#### Constructing GLMs - Softmax Regression

Consider a classification problem in which the response variable $$Y$$ can take on any one of $$k$$ values, so $$y \in [1, 2, \dots, k]$$. We will model it as distributed according to a multinomial distribution.

Derive a GLM for modelling multinomial data. To do so, Begin by expressing the multinomial as an exponential family distribution.

To parameterize a multinomial over $$k$$ possible outcomes, use $$k$$ parameters $$\phi_1, \dots, \phi_k$$ specifying the probability of each of the outcomes. However, these parameters would be redundant, or more formally, they would not be independent (since knowing any $$k - 1$$ of the $$\phi_i$$ 's uniquely determines the last one, as they must satisfy $$\sum_{i=1}^{k} \phi_i = 1$$). So, we will instead parameterize the multinomial with only $$k - 1$$ parameters, $$\phi_1, \dots, \phi_{k-1}$$, where $$\phi_i = P(y = i)$$, and $$P(y = k) = 1 - \sum_{i=1}^{k-1} \phi_i$$. For notational convenience, we will also let $$\phi_k = 1 - \sum_{i=1}^{k-1} \phi_i$$, but we should keep in mind that this is not a parameter, and that it is fully specified by $$\phi_1, \dots, \phi_{k-1}$$.

To express the multinomial as an exponential family distribution, we will define $$T(y) \in R^{k-1}$$ as follows:

$$
T(1) =
\begin{bmatrix}
  1 \\
  0 \\
  0 \\
  \vdots \\
  0 \\
\end{bmatrix},
T(2) =
\begin{bmatrix}
  0 \\
  1 \\
  0 \\
  \vdots \\
  0 \\
\end{bmatrix},
T(3) =
\begin{bmatrix}
  0 \\
  0 \\
  1 \\
  \vdots \\
  0 \\
\end{bmatrix},
\dots,
T(k-1) =
\begin{bmatrix}
  0 \\
  0 \\
  0 \\
  \vdots \\
  1 \\
\end{bmatrix},
T(k) =
\begin{bmatrix}
  0 \\
  0 \\
  0 \\
  \vdots \\
  0 \\
\end{bmatrix}
$$

$$T(y)$$ is a $$k - 1$$ dimensional vector, rather than a real number. We will write $${(T(y))}_i$$ to denote the $$i$$-th element of the vector $$T(y)$$. So, we can also write the relationship between $$T(y)$$ and $$Y$$ as $${(T(y))}_i = 1[y=i]$$. Further, we have that $$E[{(T(y))}_i] = E[1[y=i]] = P(y = i) = \phi_i$$.

We are now ready to show that the multinomial is a member of the exponential family. We have:

$$ \begin{align*}
P(y;\phi_1, \dots, \phi_{k-1})
&= \phi_1^{1\{y=1\}} \phi_2^{1\{y=2\}} \dots \phi_k^{1\{y=k\}} \\
&= \phi_1^{1\{y=1\}} \phi_2^{1\{y=2\}} \dots \phi_k^{1 - \sum_{i=1}^{k-1} 1\{y=i\}} \\
&= \phi_1^{ {(T(y))}_1 } \phi_2^{ {(T(y))}_2 } \dots \phi_k^{1 - \sum_{i=1}^{k-1} {(T(y))}_i} \\
&= exp \bigg( {(T(y))}_1 log(\phi_1) + {(T(y))}_2 log(\phi_2) + \dots + \Big( 1 - \sum_{i=1}^{k-1} {(T(y))}_i \Big) log(\phi_k) \bigg) \\
&= exp \bigg( {(T(y))}_1 log({\phi_1}/{\phi_k}) + {(T(y))}_2 log({\phi_2}/{\phi_k}) + \dots + {(T(y))}_{k-1} log({\phi_{k-1}}/{\phi_k}) + log(\phi_k) \bigg) \\
&= b(y)exp(\eta^T T(y) - a(\eta)) \\
\end{align*} $$

where

$$ \begin{align*}
\eta &=
\begin{bmatrix}
  log(\phi_1/\phi_k) \\
  log(\phi_2/\phi_k) \\
  \vdots \\
  log(\phi_{k-1}/\phi_k)
\end{bmatrix} \\
\\
T(y) &=
\begin{bmatrix}
  1\{y=1\} \\
  1\{y=2\} \\
  \vdots \\
  1\{y=k-1\} \\
\end{bmatrix} \\
\\
a(\eta) &= -log(\phi_k) \\
\\
b(y) &= 1 \\
\end{align*} $$

This completes our formulation of the multinomial as an exponential family distribution.

For $$i = 1, \dots, k$$

$$ \eta_i = log \frac{\phi_i}{\phi_k} $$

For convenience, we have also defined $\eta_k = log(\phi_k/\phi_k) = 0$. We therefore have that

$$ \begin{align*}
e^{\eta_i} &= \frac{\phi_i}{\phi_k} \\
\\
\phi_i &= \phi_k e^{\eta_i} \\
\\
\sum_{i=1}^{k} \phi_i &= \phi_k \sum_{i=1}^{k} e^{\eta_i} = 1 \\
\\
\phi_k &= \frac{1}{\sum_{i=1}^{k} e^{\eta_i}} \\
\\
\phi_i &= \frac{e^{\eta_i}}{\sum_{j=1}^{k} e^{\eta_j}} \\
\end{align*} $$

This function mapping from the $$\eta$$ 's to the $$\phi$$ 's is called the **softmax** function.

To complete our model, so, have $$\eta_i = \theta_i^T x$$ (for $$i = 1, \dots, k - 1$$), where $$\theta_1, \dots, \theta_{k-1} \in R^{n+1}$$ are the parameters of our model. For notational convenience, we can also define $$\theta_k = 0$$, so that $$\eta_k = \theta_k^T x = 0$$, as given previously. Hence, our model assumes that the conditional distribution of $$Y$$ given $$X$$ is given by

$$ \begin{align*}
P(y = i \mid x; \theta) &= \phi_i \\
\\
&= \frac{e^{\eta_i}}{\sum_{j=1}^{k} e^{\eta_j}} \\
\\
&= \frac{e^{\theta_i^T x}}{\sum_{j=1}^{k} e^{\theta_j^T x}} \\
\end{align*} $$

This model, which applies to classification problems where $$y \in [1, \dots, k]$$, is called **softmax regression**. It is a generalization of logistic regression.

Our hypothesis will output

$$ \begin{align*}
h_\theta(x) &= E[T(y) \mid x;\theta] \\
\\
&= E
\left[
\begin{array}{c|c}
  1\{y=1\} \\
  1\{y=2\} \\
  \vdots & x;\theta \\
  1\{y=k-1\} \\
\end{array}
\right] \\
\\
&=
\left[
\begin{array}{c}
  \phi_1 \\
  \phi_2 \\
  \vdots \\
  \phi_{k-1} \\
\end{array}
\right] \\
\\
&=
\left[
\begin{array}{c}
  \frac{exp(\theta_1^T x)}{\sum_{j=1}^{k} exp(\theta_j^T x)} \\
  \frac{exp(\theta_2^T x)}{\sum_{j=1}^{k} exp(\theta_j^T x)} \\
  \vdots \\
  \frac{exp(\theta_{k-1}^T x)}{\sum_{j=1}^{k} exp(\theta_j^T x)} \\
\end{array}
\right] \\
\end{align*} $$

In other words, our hypothesis will output the estimated probability that $$P(y = i \mid x; \theta)$$, for every value of $$i = 1, \dots, k$$. (Even though $$h_\theta (x)$$ as defined above is only $$k - 1$$ dimensional, clearly $$P(y = k \mid x; \theta)$$ can be obtained as $$1 - \sum_{i=1}^{k-1} \phi_i$$.)

Lastly, lets discuss parameter fitting. Similar to our original derivation of ordinary least squares and logistic regression, if we have a training set of $$m$$ examples $$[(x^{(i)}, y^{(i)}); i = 1, \dots, m]$$ and would like to learn the parameters $$\theta_i$$ of this model, we would begin by writing down the log-likelihood

$$ \begin{align*}
\ell (\theta)
&= \sum_{i=1}^{m} log P(y^{(i)} \mid x^{(i)}, \theta) \\
&= \sum_{i=1}^{m} log \prod_{l=1}^{k} \bigg( \frac{e^{\theta_l^T x^{(i)}}}{\sum_{j=1}^{k} e^{\theta_j^T x^{(i)}}} \bigg)^{1\{y^{(i)}=l\}} \\
\end{align*} $$

We can now obtain the maximum likelihood estimate of the parameters by maximizing $\ell (\theta)$ in terms of $\theta$, using a method such as gradient ascent or Newton method.

## Generative Learning Algorithms

Algorithms that try to learn $$p(y \mid x)$$ directly (such as logistic regression), or algorithms that try to learn mappings directly from the space of inputs $$X$$ to the labels $$[0, 1]$$, (such as the perceptron algorithm) are called **discriminative learning algorithms**. Here, we'll talk about algorithms that instead try to model $$p(x \mid y)$$ and $$p(y)$$. These algorithms are called **generative learning algorithms**.

After modeling $$p(y)$$ (called the **class priors**) and $$p(x \mid y)$$, our algorithm can then use Bayes rule to derive the posterior distribution on $$Y$$ given $$x$$:

$$ p(y \mid x) = \frac{p(x \mid y)p(y)}{p(x)} $$

Here, the denominator can also be expressed in terms of the quantities $$p(x \mid y)$$ and $$p(y)$$ that we've learned. Actually, if were calculating $$p(y \mid x)$$ in order to make a prediction, then we don't actually need to calculate the denominator, since

$$ \mathop{argmax}\limits_{y} \text{ } p(y \mid x) = \mathop{argmax}\limits_{y} \text{ } \frac{p(x \mid y)p(y)}{p(x)} = \mathop{argmax}\limits_{y} \text{ } p(x \mid y)p(y) $$

### Gaussian Discriminant Analysis

In Gaussian discriminant analysis (GDA), we'll assume that $$p(x \mid y)$$ is distributed according to a multivariate normal distribution. Lets talk briefly about the properties of multivariate normal distributions before moving on to the GDA model itself.

#### The Multivariate Normal Distribution

The multivariate normal distribution in $$n$$ dimensions, also called the multivariate Gaussian distribution, is parameterized by a **mean vector** $$\mu \in R^n$$ and a **covariance matrix** $$\Sigma \in R^{n \times n}$$, where $$\Sigma \ge 0$$ is symmetric and positive semi-definite. Also written $$N (\mu, \Sigma)$$, its density is given by:

$$ p(x; \mu, \Sigma) = \frac{1}{(2\pi)^{n/2} {\vert \Sigma \vert}^{1/2}} exp \bigg( -\frac{1}{2} (x - \mu)^T {\Sigma}^{-1} (x - \mu) \bigg) $$

For a vector-valued random variable $$X \sim N(\mu, \Sigma)$$, the mean is given by $$\mu$$:

$$ \mu = E[X] = \int_x x \text{ } p(x; \mu, \Sigma) \text{ } dx $$

then the covariance matrix is given by $$\Sigma$$

$$ \Sigma = Cov(X) $$

#### The Gaussian Discriminant Analysis Model

When we have a classification problem in which the input features $$X$$ are continuous-valued random variables, we can then use the Gaussian Discriminant Analysis (GDA) model, which models $$p(x \mid y)$$ using a multivariate normal distribution. The model is:

$$ \begin{align*}
y & \sim Bernoulli(\phi) \\
\\
(x \mid y = 0) & \sim N (\mu_0, \Sigma) \\
\\
(x \mid y = 1) & \sim N (\mu_1, \Sigma) \\
\end{align*} $$

Writing out the distributions, this is:

$$ \begin{align*}
P(y) &= \phi^{y} (1 - \phi)^{1-y} \\
p(x \mid y = 0) &= \frac{1}{(2\pi)^{n/2} {\vert \Sigma \vert}^{1/2}} exp \bigg( -\frac{1}{2} (x - \mu_0)^T {\Sigma}^{-1} (x - \mu_0) \bigg) \\
p(x \mid y = 1) &= \frac{1}{(2\pi)^{n/2} {\vert \Sigma \vert}^{1/2}} exp \bigg( -\frac{1}{2} (x - \mu_1)^T {\Sigma}^{-1} (x - \mu_1) \bigg) \\
\end{align*} $$

Here, the parameters of our model are $$\phi, \Sigma, \mu_0$$ and $$\mu_1$$. (Note that while there're two different mean vectors $$\mu_0$$ and $$\mu_1$$, this model is usually applied using only one covariance matrix $$\Sigma$$.) The log-join-likelihood of the data is given by

$$ \begin{align*}
\ell (\phi ,\mu_0 ,\mu_1 ,\Sigma)
&= log \prod_{i=1}^{m} p(x^{(i)}, y^{(i)};\phi ,\mu_0 ,\mu_1 ,\Sigma) \\
&= log \prod_{i=1}^{m} p(x^{(i)} \mid y^{(i)};\mu_0 ,\mu_1 ,\Sigma) p(y^{(i)};\phi) \\
\end{align*} $$

By maximizing $$\ell$$ with respect to the parameters, we find the maximum likelihood estimate of the parameters to be:

$$ \begin{align*}
\phi &= \frac{1}{m} \sum_{i=1}^{m} 1\{y^{(i)}=1\} \\
\\
\mu_0 &= \frac{\sum_{i=1}^{m} 1\{y^{(i)}=0\} x^{(i)}}{\sum_{i=1}^{m} 1\{y^{(i)}=0\}} \\
\\
\mu_1 &= \frac{\sum_{i=1}^{m} 1\{y^{(i)}=1\} x^{(i)}}{\sum_{i=1}^{m} 1\{y^{(i)}=1\}} \\
\\
\Sigma &= \frac{1}{m} \sum_{i=1}^{m} (x^{(i)} - \mu_{y^{(i)}}) (x^{(i)} - \mu_{y^{(i)}})^T \\
\end{align*} $$

Pictorially, what the algorithm is doing can be seen in as follows:

{% include image.html description="gda" image="machine-learning/gda.png" caption="true"%}

Shown in the figure are the training set, as well as the contours of the two Gaussian distributions that have been fit to the data in each of the two classes. Note that the two Gaussians have contours that are the same shape and orientation, since they share a covariance matrix $$\Sigma$$, but they have different means $$\mu_0$$ and $$\mu_1$$. Also shown in the figure is the straight line giving the decision boundary at which $$p(y = 1 \mid x) = 0.5$$. On one side of the boundary, we'll predict $$y = 1$$ to be the most likely outcome, and on the other side, we'll predict $$y = 0$$.

#### Discussion: GDA and Logistic Regression

The GDA model has an interesting relationship to logistic regression. If we view the quantity $$P(y = 1 \mid x; \phi ,\mu_0 ,\mu_1 ,\Sigma)$$ as a function of $$x$$, we'll find that it can be expressed in the form of logistic regression

$$ P(y = 1 \mid x; \phi ,\mu_0 ,\mu_1 ,\Sigma) = \frac{1}{1 + exp(-\theta^T x)} $$

because:

$$ \begin{align*}
p(y=1 \mid x) &= \frac{p(x \mid y=1) p(y=1)}{p(x)} \\
&= \frac{p(x \mid y=1) p(y=1)}{p(x \mid y=0) p(y=0) + p(x \mid y=1) p(y=1)} \\
&= \frac{1}{1 + \frac{p(x \mid y=0) p(y=0)}{p(x \mid y=1) p(y=1)}}
\end{align*} $$

then:

$$ \begin{align*}
&= \frac{p(x \mid y=0) p(y=0)}{p(x \mid y=1) p(y=1)} \\
&= exp \bigg( -\frac{1}{2} (x - \mu_0)^T {\Sigma}^{-1} (x - \mu_0) + \frac{1}{2} (x - \mu_1)^T {\Sigma}^{-1} (x - \mu_1) \bigg) \times \frac{1 - \phi}{\phi} \\
&= exp \bigg( \frac{1}{2} \Big( x^T {\Sigma}^{-1} \mu_0 - x^T {\Sigma}^{-1} \mu_1 +  \mu_0^T {\Sigma}^{-1} x - \mu_1^T {\Sigma}^{-1} x - \mu_0^T {\Sigma}^{-1} \mu_0 + \mu_1^T {\Sigma}^{-1} \mu_1 \Big) \bigg) \times exp \bigg( \log(\frac{1-\phi}{\phi}) \bigg) \\
&= exp \bigg( (\mu_0 - \mu_1)^T {\Sigma}^{-1} x - \frac{\mu_0^T {\Sigma}^{-1} \mu_0 - \mu_1^T {\Sigma}^{-1} \mu_1}{2} +\log(\frac{1-\phi}{\phi}) \bigg) \\
&= exp \bigg((\mu_0 - \mu_1)^T {\Sigma}^{-1} x + \Big( \log(\frac{1-\phi}{\phi}) - \frac{\mu_0^T {\Sigma}^{-1} \mu_0 - \mu_1^T {\Sigma}^{-1} \mu_1}{2} \Big) x_0 \bigg) \\
\end{align*} $$

so:

$$ \theta = -
\left[
\begin{array}{c}
  log(\frac{1-\phi}{\phi}) - \frac{\mu_0^T {\Sigma}^{-1} \mu_0 - \mu_1^T {\Sigma}^{-1} \mu_1}{2} \\
  {\Sigma}^{-1} (\mu_0 - \mu_1) \\
\end{array}
\right]
$$

where $$\theta$$ is some appropriate function of $$\phi ,\mu_0 ,\mu_1 ,\Sigma$$. (uses the convention of redefining the $$x^{(i)}$$ 's to be $$n + 1$$ dimensional vectors by adding the extra coordinate $$x_0^{(i)} = 1$$) This is exactly the form that logistic regression - a discriminative algorithm - used to model $$P(y = 1 \mid x)$$.

When would we prefer one model over another? GDA and logistic regression will, in general, give different decision boundaries when trained on the same dataset. Which is better?

We just argued that if $$p(x \mid y)$$ is multivariate gaussian (with shared $$\Sigma$$), then $$P(y \mid x)$$ necessarily follows a logistic function. The converse, however, is not true; i.e., $$P(y \mid x)$$ being a logistic function does not imply $$p(x \mid y)$$ is multivariate gaussian. This shows that GDA makes stronger modeling assumptions about the data than does logistic regression. It turns out that when these modeling assumptions are correct, when $$p(x \mid y)$$ is indeed gaussian (with shared $$\Sigma$$), then GDA will find better fits to the data, and is a better model.

In contrast, by making significantly weaker assumptions, logistic regression is also more robust and less sensitive to incorrect modeling assumptions. There are many different sets of assumptions that would lead to $$P(y \mid x)$$ taking the form of a logistic function. For example, if $$(x \mid y = 0) \sim Poisson(\lambda_0)$$, and $$(x \mid y = 1) \sim Poisson(\lambda_1)$$, then $$P(y \mid x)$$ will be logistic. Logistic regression will also work well on Poisson data like this. But if we were to use GDA on such data, and fit Gaussian distributions to such non-Gaussian data, then the results will be less predictable, and GDA may or may not do well.

Turns out if you assume $$p(x \mid y=1) \sim ExponentialFamily(\eta_1)$$ and $$p(x \mid y=0) \sim ExponentialFamily(\eta_0)$$, then this implies that $$P(y = 1 \mid x)$$ is also logistic. So the same exponential family distribution for the two classes with different natural parameters then the posterior $$P(y = 1 \mid x)$$ would be logistic, and so this shows the robustness of logistic regression to the choice of modeling assumptions because it could be that the data can be anyone of exponential family distribution. So it's the robustness of logistic regression to modeling assumptions.

To summarize: GDA makes stronger modeling assumptions, and is more data efficient (requires less training data to learn well) when the modeling assumptions are correct or at least approximately correct. Logistic regression makes weaker assumptions, and is significantly more robust to deviations from modeling assumptions. For this reason, in practice logistic regression is used more often than GDA.

### Naive Bayes

#### Naive Bayes

In GDA, the feature vectors $$X$$ were continuous, real-valued vectors. Lets now talk about a different learning algorithm to be used in **text classification** in which the $$x_i$$ 's are discrete-valued.

We will represent the text via a feature vector whose length is equal to the number of words in the dictionary. Specifically, if the text contains the $$i$$-th word of the dictionary, then we will set $$x_i = 1$$; otherwise, we let $$x_i = 0$$.

For instance, the vector

$$
x =
\left[
\begin{array}{c}
  1 \\
  0 \\
  0 \\
  \vdots \\
  1 \\
  \vdots \\
  0 \\
\end{array}
\right]
\begin{array}{c}
  a \\
  coat \\
  cold \\
  \vdots \\
  buy \\
  \vdots \\
  zoo \\
\end{array}
$$

is used to represent the text that contains the words "a" and "buy," but not "coat", "cold" or "zoo". The set of words encoded into the feature vector is called the **vocabulary**, so the dimension of $$X$$ is equal to the size of the vocabulary.

Actually, rather than looking through an english dictionary for the list of all english words, in practice it is more common to look through our training set and encode in our feature vector only the words that occur at least once there. Apart from reducing the number of words modeled and hence reducing our computational and space requirements, this also has the advantage of allowing us to model/include as a feature many words that may appear in the text but that you won't find in a dictionary. Sometimes, we also exclude the very high frequency words (which will be words like "the", "of", "and"; these high frequency, "content free" words are called **stop words**) since they occur in so many documents and do little to indicate the meaning of the text.

Having chosen our feature vector, we now want to build a discriminative model. So, we have to model $$p(x \mid y)$$. But if we have, say, a vocabulary of $$50000$$ words, then $$x \in [0, 1]^{50000}$$ ($$x$$ is a $$50000$$-dimensional vector of $$0$$ 's and $$1$$ 's), and if we were to model $$X$$ explicitly with a multinomial distribution over the $$2^{50000}$$ possible outcomes, then we'd end up with $$(2^{50000} -1)$$ parameter vectors. This is clearly too many parameters.

To model $$p(x \mid y)$$, we will therefore make a very strong assumption. We will assume that the $$x_i$$ 's are conditionally independent given $$y$$. This assumption is called the **Naive Bayes (NB) assumption**, and the resulting algorithm is called the **Naive Bayes classifier**. (Note that this is not the same as saying that $$x_i$$ 's are independent, which would have been written $$p(x_i) = p(x_i \mid x_j)$$; rather, we are only assuming that $$x_i$$ 's are conditionally independent given $$y$$.)

we now have:

$$ \begin{align*}
p(x_1, \dots, x_{50000} \mid y)
&= p(x_1 \mid y)p(x_2 \mid y,x_1)p(x_3 \mid y,x_1,x_2) \dots p(x_{50000} \mid y,x_1, \dots, x_{49999}) \\
&= p(x_1 \mid y)p(x_2 \mid y)p(x_3 \mid y) \dots p(x_{50000} \mid y) \\
&= \prod_{i=1}^{n} p(x_i \mid y) \\
\end{align*} $$

Our model is parameterized by $\phi_i \mid_{y=1} = p(x_i = 1 \mid y = 1)$, $\phi_i \mid_{y=0} = p(x_i = 1 \mid y = 0)$, and $\phi_y = p(y = 1)$. As usual, given a training set $[(x^{(i)}, y^{(i)}); i = 1, \dots, m]$, we can write down the joint likelihood of the data:

$$ L(\phi_y, \phi_i  \mid {}_{y=0}, \phi_i \mid {}_{y=1}) = \prod_{i=1}^{m} p(x^{(i)}, y^{(i)}) $$

Maximizing this with respect to $\phi_y$, $\phi_i \mid_{y=0}$ and $\phi_i \mid_{y=1}$ gives the maximum likelihood estimates:

$$ \begin{align*}
\phi_j \mid {}_{y=1} &= \frac{\sum_{i=1}^{m} 1\{x_j^{(i)}=1, y^{(i)}=1\}}{\sum_{i=1}^{m} 1\{y^{(i)}=1\}} \\
\\
\phi_j \mid {}_{y=0} &= \frac{\sum_{i=1}^{m} 1\{x_j^{(i)}=1, y^{(i)}=0\}}{\sum_{i=1}^{m} 1\{y^{(i)}=0\}} \\
\\
\phi_y &= \frac{\sum_{i=1}^{m} 1\{y^{(i)}=1\}}{m} \\
\end{align*} $$

Having fit all these parameters, to make a prediction on a new example with features $$x$$, we then simply calculate

$$ \begin{align*}
p(y = 1 \mid x) &= \frac{p(x \mid y = 1)p(y = 1)}{p(x)} \\
&= \frac{(\prod_{i=1}^{n}p(x_i \mid y=1))p(y = 1)}{(\prod_{i=1}^{n}p(x_i \mid y=1))p(y = 1) + (\prod_{i=1}^{n}p(x_i \mid y=0))p(y = 0)} \\
\end{align*} $$

and pick whichever class has the higher posterior probability.

Lastly, we note that while we have developed the Naive Bayes algorithm mainly for the case of problems where the features $$x_i$$ are binary-valued, the generalization to where $$x_i$$ can take values in $$[1, 2, \dots, k_i]$$ is straightforward. Here, we would simply model $$p(x_i \mid y)$$ as multinomial rather than as Bernoulli. Indeed, even if some original input attribute were continuous valued, it is quite common to discretize it - that is, turn it into a small set of discrete values by mapping continuous valued to different range intervals - and apply Naive Bayes. When the original, continuous-valued attributes are not well-modeled by a multivariate normal distribution, discretizing the features and using Naive Bayes (instead of GDA) will often result in a better classifier.

#### Laplace Smoothing

The Naive Bayes algorithm as we have described it will work fairly well for many problems, but there is a simple change that makes it work much better, especially for text classification. Lets briefly discuss a problem with the algorithm in its current form, and then talk about how we can fix it.

Consider the text has a word that you never seen it before. Assuming that word was the 35000th in the dictionary, Naive Bayes text classification therefore had picked its maximum likelihood estimates of the parameters $$\phi_{35000} \mid {}_y$$ to be

$$ \begin{align*}
\phi_{35000} \mid {}_{y=1} &= \frac{\sum_{i=1}^{m} 1\{x_{35000}^{(i)} = 1, y^{(i)}=1\}}{\sum_{i=1}^{m} 1\{y^{(i)}=1\}} = 0\\
\\
\phi_{35000} \mid {}_{y=0} &= \frac{\sum_{i=1}^{m} 1\{x_{35000}^{(i)} = 1, y^{(i)}=0\}}{\sum_{i=1}^{m} 1\{y^{(i)}=0\}} = 0 \\
\end{align*} $$

I.e., because the $$35000$$th word never seen before in any training examples, it thinks the probability of seeing it is zero. Hence, when trying to classify the text containing that word, it calculates the class posterior probabilities, and obtains

$$ \begin{align*}
p(y = 1 \mid x)
&= \frac{(\prod_{i=1}^{n}p(x_i \mid y=1))p(y = 1)}{(\prod_{i=1}^{n}p(x_i \mid y=1))p(y = 1) + (\prod_{i=1}^{n}p(x_i \mid y=0))p(y = 0)} \\
&= \frac{0}{0} \\
\end{align*} $$

This is because each of the terms $$\prod_{i=1}^{n} p(x_i \mid y)$$ includes a term $$p(x_{35000} \mid y) = 0$$ that is multiplied into it. Hence, our algorithm obtains $$0/0$$, and doesn't know how to make a prediction.

Stating the problem more broadly, it is statistically a bad idea to estimate the probability of some event to be zero just because you haven't seen it before in your finite training set. Take the problem of estimating the mean of a multinomial random variable $$z$$ taking values in $$[1, \dots, k]$$. We can parameterize our multinomial with $$\phi_i = p(z = i)$$. Given a set of $$m$$ independent observations $$[z^{(1)}, \dots, z^{(m)}]$$, the maximum likelihood estimates are given by

$$ \phi_j = \frac{\sum_{i=1}^{m} 1\{z^{(i)}=j\}}{m} $$

As we saw previously, if we were to use these maximum likelihood estimates, then some of the $$\phi_j$$ 's might end up as zero, which was a problem. To avoid this, we can use **Laplace Smoothing**, which replaces the above estimate with

$$ \phi_j = \frac{1 + \sum_{i=1}^{m} 1\{z^{(i)}=j\}}{k + m} $$

Here, we've added $$1$$ to the numerator, and $$k$$ to the denominator. Note that $$\sum_{j=1}^{k} \phi_j = 1$$ still holds, which is a desirable property since the $$\phi_j$$ 's are estimates for probabilities that we know must sum to $$1$$. Also, $$\phi_j \neq 0$$ for all values of $$j$$, solving our problem of probabilities being estimated as zero. Under certain (arguably quite strong) conditions, it can be shown that the Laplace smoothing actually gives the optimal estimator of the $$\phi_j$$ 's.

Returning to our Naive Bayes classifier, with Laplace smoothing, we therefore obtain the following estimates of the parameters:

$$ \begin{align*}
\phi_j \mid {}_{y=1} &= \frac{1 + \sum_{i=1}^{m} 1\{x_j^{(i)}=1, y^{(i)}=1\}}{2 + \sum_{i=1}^{m} 1\{y^{(i)}=1\}} \\
\\
\phi_j \mid {}_{y=0} &= \frac{1 + \sum_{i=1}^{m} 1\{x_j^{(i)}=1, y^{(i)}=0\}}{2 + \sum_{i=1}^{m} 1\{y^{(i)}=0\}} \\
\end{align*} $$

In practice, it usually doesn't matter much whether we apply Laplace smoothing to $$\phi_y$$ or not, since we will typically have a fair fraction each of different classification, so $$\phi_y$$ will be a reasonable estimate of $$p(y = 1)$$ and will be quite far from $$0$$ anyway.

#### Event models for text classification

To close off our discussion of generative learning algorithms, lets talk about one more model that is specifically for text classification. While Naive Bayes as we've presented it will work well for many classification problems, for text classification, there is a related model that does even better.

In the specific context of text classification, Naive Bayes as presented uses the what's called the **multivariate Bernoulli event model**. In this model, we assumed that the way a text is generated is that first it is randomly determined (according to the class priors $$p(y)$$) its classification. Then, runs through the dictionary, deciding whether to include each word $$i$$ in that text independently and according to the probabilities $$p(x_i = 1 \mid y) = \phi_i \mid_y$$. Thus, the probability of a text was given by $$p(y) \prod_{i=1}^{n} p(x_i \mid y)$$.

Here's a different model, called the **multinomial event model**. To describe this model, we will use a different notation and set of features for representing texts. We let $$x_i$$ denote the identity of the $$i$$-th word in the text. Thus, $$x_i$$ is now an integer taking values in $$[1, \dots, \vert V \vert]$$, where $$\vert V \vert$$ is the size of our vocabulary (dictionary). A text of $$n$$ words is now represented by a vector $$(x_1, x_2, \dots, x_n)$$ of length $$n$$; note that $$n$$ can vary for different texts.

In the multinomial event model, we assume that the way a text is generated is via a random process in which classification is first determined (according to $$p(y)$$) as before. Then, the text by first generating $$x_1$$ from some multinomial distribution over words. Next, the second word $$x_2$$ is chosen independently of $$x_1$$ but from the same multinomial distribution, and similarly for $$x_3$$, $$x_4$$, and so on, until all $$n$$ words of the text have been generated. Thus, the overall probability of a text is given by $$p(y) \prod_{i=1}^{n} p(x_i \mid y)$$. Note that this formula looks like the one we had earlier for the probability of a text under the multivariate Bernoulli event model, but that the terms in the formula now mean very different things. In particular $$x_i \mid y$$ is now a multinomial, rather than a Bernoulli distribution.

The parameters for our new model are $$\phi_y = p(y)$$ as before, $$\phi_i \mid_{y=1} = p(x_j = i \mid y = 1)$$ and $$\phi_i \mid_{y=0} = p(x_j = i \mid {y = 0})$$ (for any $$j$$). Note that we have assumed that $$p(x_j \mid y)$$ is the same for all values of $$j$$ (i.e., that the distribution according to which a word is generated does not depend on its position $$j$$ within the text).

If we are given a training set $$[(x^{(i)}, y^{(i)}); i = 1, \dots, m]$$ where $$x^{(i)} = (x_1^{(i)}, x_2^{(i)}, \dots, x_{n_i}^{(i)})$$ (here, $$n_i$$ is the number of words in the $$i$$-training example), the likelihood of the data is given by

$$ \begin{align*}
L(\phi_y, \phi_i \mid {}_{y=0}, \phi_i \mid {}_{y=1})
&= \prod_{i=1}^{m} p(x^{(i)}, y^{(i)}) \\
&= \prod_{i=1}^{m} \bigg( \prod_{j=1}^{n_i} p(x_j^{(i)} \mid y^{(i)}; \phi_i \mid {}_{y=0}, \phi_i \mid {}_{y=1}) \bigg) p(y^{(i)};\phi_y) \\
\end{align*} $$

Maximizing this yields the maximum likelihood estimates of the parameters

$$ \begin{align*}
\phi_k \mid {}_{y=1} &= \frac{\sum_{i=1}^{m} \sum_{j=1}^{n_i} 1\{x_j^{(i)}=k, y^{(i)}=1\}}{\sum_{i=1}^{m} 1\{y^{(i)}=1\} n_i} \\
\\
\phi_k \mid {}_{y=0} &= \frac{\sum_{i=1}^{m} \sum_{j=1}^{n_i} 1\{x_j^{(i)}=k, y^{(i)}=0\}}{\sum_{i=1}^{m} 1\{y^{(i)}=0\} n_i} \\
\\
\phi_y &= \frac{\sum_{i=1}^{m} 1\{y^{(i)}=1\}}{m} \\
\end{align*} $$

If we were to apply Laplace smoothing (which needed in practice for good performance) when estimating $$\phi_k \mid_{y=0}$$ and $$\phi_k \mid_{y=1}$$, we add $$1$$ to the numerators and $$\vert V \vert$$ to the denominators, and obtain:

$$ \begin{align*}
\phi_k \mid {}_{y=1} &= \frac{1 + \sum_{i=1}^{m} \sum_{j=1}^{n_i} 1\{x_j^{(i)}=k, y^{(i)}=1\}}{\vert V \vert + \sum_{i=1}^{m} 1\{y^{(i)}=1\} n_i} \\
\\
\phi_k \mid {}_{y=0} &= \frac{1 + \sum_{i=1}^{m} \sum_{j=1}^{n_i} 1\{x_j^{(i)}=k, y^{(i)}=0\}}{\vert V \vert + \sum_{i=1}^{m} 1\{y^{(i)}=0\} n_i} \\
\end{align*} $$

While not necessarily the very best classification algorithm, the Naive Bayes classifier often works surprisingly well. It is often also a very good "first thing to try," given its simplicity and ease of implementation.

## Support Vector Machines

SVMs are among the best (and many believe is indeed the best) "off-the-shelf" supervised learning algorithm. To tell the SVM story, we'll need to first talk about margins and the idea of separating data with a large "gap". Next, we'll talk about the optimal margin classifier, which will lead us into a digression on Lagrange duality. We'll also see kernels, which give a way to apply SVMs efficiently in very high dimensional (such as infinite-dimensional) feature spaces, and finally, we'll close off the story with the SMO algorithm, which gives an efficient implementation of SVMs.

### Margins: Intuition

We'll start our story on SVMs by talking about margins. This section will give the intuitions about margins and about the "confidence" of our predictions;

Consider logistic regression, where the probability $$p(y = 1 \mid x; \theta)$$ is modeled by $$h_\theta (x) = g(\theta^T x)$$. We would then predict "1" on an input $$X$$ if and only if $$h_\theta (x) \geq 0.5$$, or equivalently, if and only if $$\theta^T x \geq 0$$. Consider a positive training example ($$y = 1$$). The larger $$\theta^T x$$ is, the larger also is $$h_\theta (x) = p(y = 1 \mid x; \theta)$$, and thus also the higher our degree of "confidence" that the label is $$1$$. Thus, informally we can think of our prediction as being a very confident one that $$y = 1$$ if $$\theta^T x \gg 0$$. Similarly, we think of logistic regression as making a very confident prediction of $$y = 0$$, if $$\theta^T x \ll 0$$. Given a training set, again informally it seems that we'd have found a good fit to the training data if we can find $$\theta$$ so that $$\theta^T x^{(i)} \gg 0$$ whenever $$y^{(i)} = 1$$, and $$\theta^T x^{(i)} \ll 0$$ whenever $$y^{(i)} = 0$$, since this would reflect a very confident (and correct) set of classifications for all the training examples. This seems to be a nice goal to aim for, and we'll soon formalize this idea using the notion of functional margins.

For a different type of intuition, consider the following figure, in which x's represent positive training examples, o's denote negative training examples, a decision boundary (this is the line given by the equation $$\theta^T x = 0$$, and is also called the **separating hyperplane**) is also shown, and three points have also been labeled A, B and C.

{% include image.html description="hyperplane" image="machine-learning/hyperplane.png" caption="true"%}

Notice that the point A is very far from the decision boundary. If we are asked to make a prediction for the value of $$Y$$ at at A, it seems we should be quite confident that $$y = 1$$ there. Conversely, the point C is very close to the decision boundary, and while it's on the side of the decision boundary on which we would predict $$y = 1$$, it seems likely that just a small change to the decision boundary could easily have caused out prediction to be $$y = 0$$. Hence, we're much more confident about our prediction at A than at C. The point B lies in-between these two cases, and more broadly, we see that if a point is far from the separating hyperplane, then we may be significantly more confident in our predictions. Again, informally we think it'd be nice if, given a training set, we manage to find a decision boundary that allows us to make all correct and confident (meaning far from the decision boundary) predictions on the training examples. We'll formalize this later using the notion of geometric margins.

### Notation

To make our discussion of SVMs easier, we'll first need to introduce a new notation for talking about classification. We will be considering a linear classifier for a binary classification problem with labels $$Y$$ and features $$x$$. From now, we'll use $$y \in [-1, 1]$$ (instead of $$[0, 1]$$) to denote the class labels. Also, rather than parameterizing our linear classifier with the vector $$\theta$$, we will use parameters $$w$$, $$b$$, and write our classifier as

$$ h_{w,b} (x) = g(w^T x + b) $$

Here, $$g(z) = 1$$ if $$z \ge 0$$, and $$g(z) = -1$$ otherwise. This "$$w, b$$" notation allows us to explicitly treat the intercept term $$b$$ separately from the other parameters. (We also drop the convention we had previously of letting $$x_0 = 1$$ be an extra coordinate in the input feature vector.) Thus, $$b$$ takes the role of what was previously $$\theta_0$$, and $$w$$ takes the role of $$[\theta_1 \dots \theta_n]^T$$.

Note also that, from our definition of $$g$$ above, our classifier will directly predict either $$1$$ or $$-1$$ (cf. the perceptron algorithm), without first going through the intermediate step of estimating the probability of $$Y$$ being $$1$$ (which was what logistic regression did).

### Functional and Geometric Margins

Lets formalize the notions of the functional and geometric margins. Given a training example $$(x^{(i)}, y^{(i)})$$, we define the **functional margin** of $$(w, b)$$ with respect to the training example

$$ \hat{\gamma}^{(i)} = y^{(i)} (w^T x + b) $$

Note that if $$y^{(i)} = 1$$, then for the functional margin to be large (i.e., for our prediction to be confident and correct), then we need $$w^T x + b$$ to be a large positive number. Conversely, if $$y^{(i)} = -1$$, then for the functional margin to be large, then we need $$w^T x + b$$ to be a large negative number. Moreover, if $$y^{(i)} (w^T x + b) > 0$$, then our prediction on this example is correct. (Check this yourself.) Hence, a large functional margin represents a confident and a correct prediction.

For a linear classifier with the choice of $$g$$ given above (taking values in $$[-1, 1]$$), there's one property of the functional margin that makes it not a very good measure of confidence, however. Given our choice of $$g$$, we note that if we replace $$w$$ with $$2w$$ and $$b$$ with $$2b$$, then since $$g(w^T x + b) = g(2w^T x + 2b)$$, this would not change $$h_{w,b} (x)$$ at all. I.e., $$g$$, and hence also $$h_{w,b} (x)$$, depends only on the sign, but not on the magnitude, of $$w^T x + b$$. However, replacing $$(w, b)$$ with $$(2w, 2b)$$ also results in multiplying our functional margin by a factor of $$2$$. Thus, it seems that by exploiting our freedom to scale $$w$$ and $$b$$, we can make the functional margin arbitrarily large without really changing anything meaningful. Intuitively, it might therefore make sense to impose some sort of normalization condition such as that $${\Vert w \Vert}_2 = 1$$; i.e., we might replace $$(w, b)$$ with $$(w/{\Vert w \Vert}_2, b/{\Vert w \Vert}_2)$$, and instead consider the functional margin of $$(w/{\Vert w \Vert}_2, b/{\Vert w \Vert}_2)$$. We'll come back to this later.

Given a training set $$S = [(x^{(i)}, y^{(i)}); i = 1, \dots, m]$$, we also define the function margin of $$(w, b)$$ with respect to $$S$$ as the smallest of the functional margins of the individual training examples. Denoted by $$\hat{\gamma}$$, this can therefore be written:

$$ \hat{\gamma} = \mathop{min}\limits_{i=1, \dots, m} \hat{\gamma}^{(i)} $$

Next, lets talk about **geometric margins**. Consider the picture below:

{% include image.html description="geometric-margins" image="machine-learning/geometric-margins.png" caption="true"%}

The decision boundary corresponding to $$(w, b)$$ is shown, along with the vector $$w$$. Note that $$w$$ is orthogonal to the separating hyperplane. (You should convince yourself that this must be the case.) Consider the point at A, which represents the input $$x^{(i)}$$ of some training example with label $$y^{(i)} = 1$$. Its distance to the decision boundary, $$\gamma^{(i)}$$, is given by the line segment AB.

How can we find the value of $$\gamma^{(i)}$$. Well, $$(w/\Vert w \Vert_2)$$ is a unit-length vector pointing in the same direction as $$w$$. Since A represents $$x^{(i)}$$, we therefore find that the point B is given by $$(x^{(i)} - \gamma^{(i)} · w/\Vert w \Vert_2)$$. But this point lies on the decision boundary, and all points $$X$$ on the decision boundary satisfy the equation $$w^T x + b = 0$$. Hence,

$$ w^T \bigg( x^{(i)} - \gamma^{(i)} \frac{w}{\Vert w \Vert_2} \bigg) + b = 0 $$

Solving for $$\gamma^{(i)}$$ yields

$$ \gamma^{(i)} = \frac{w^T x^{(i)} + b}{\Vert w \Vert_2} = \bigg( \frac{w}{\Vert w \Vert_2} \bigg)^T x^{(i)} + \frac{b}{\Vert w \Vert_2} $$

This was worked out for the case of a positive training example at A in the figure, where being on the "positive" side of the decision boundary is good. More generally, we define the geometric margin of $$(w, b)$$ with respect to a training example $$(x^{(i)}, y^{(i)})$$ to be

$$ \gamma^{(i)} = y^{(i)} \bigg( \Big(\frac{w}{\Vert w \Vert_2}\Big)^T x^{(i)} + \frac{b}{\Vert w \Vert_2} \bigg) $$

Note that if $${\Vert w \Vert}_2 = 1$$, then the functional margin equals the geometric margin - this thus gives us a way of relating these two different notions of margin. Also, the geometric margin is invariant to rescaling of the parameters; i.e., if we replace $$w$$ with $$2w$$ and $$b$$ with $$2b$$, then the geometric margin does not change. This will in fact come in handy later. Specifically, because of this invariance to the scaling of the parameters, when trying to fit $$w$$ and $$b$$ to training data, we can impose an arbitrary scaling constraint on $$w$$ without changing anything important; for instance, we can demand that $${\Vert w \Vert}_2 = 1$$, or $$\vert w_1 \vert = 5$$, or $$\vert w_1 + b \vert + \vert w_2 \vert = 2$$, and any of these can be satisfied simply by rescaling $$w$$ and $$b$$.

Finally, given a training set $$S = [(x^{(i)}, y^{(i)}); i = 1, \dots, m]$$, we also define the geometric margin of $$(w, b)$$ with respect to $$S$$ to be the smallest of the geometric margins on the individual training examples:

$$ \gamma = \mathop{min}\limits_{i=1, \dots, m} \gamma^{(i)} $$

### The optimal margin classifier

Given a training set, it seems from our previous discussion that a natural desideratum is to try to find a decision boundary that maximizes the (geometric) margin, since this would reflect a very confident set of predictions on the training set and a good "fit" to the training data. Specifically, this will result in a classifier that separates the positive and the negative training examples with a "gap" (geometric margin).

For now, we will assume that we are given a training set that is linearly separable; i.e., that it is possible to separate the positive and negative examples using some separating hyperplane. How we find the one that achieves the maximum geometric margin. We can pose the following optimization problem:

$$ \begin{align*}
\mathop{max}\limits_{\gamma, w, b} \text{ } & \gamma \\
s.t. \text{ } & y^{(i)} (w^T x^{(i)} + b) \ge \gamma, \text{ } i = 1, \dots, m \\
& {\Vert w \Vert}_2 = 1 \\
\end{align*} $$

I.e., we want to maximize $$\gamma$$, subject to each training example having functional margin at least $$\gamma$$. The $${\Vert w \Vert}_2 = 1$$ constraint moreover ensures that the functional margin equals to the geometric margin, so we are also guaranteed that all the geometric margins are at least $$\gamma$$. Thus, solving this problem will result in ($$w, b$$) with the largest possible geometric margin with respect to the training set.

If we could solve the optimization problem above, we'd be done. But the $${\Vert w \Vert}_2 = 1$$ constraint is a nasty (non-convex) one, and this problem certainly isn't in any format that we can plug into standard optimization software to solve. So, lets try transforming the problem into a nicer one. Consider:

$$ \begin{align*}
\mathop{max}\limits_{\gamma, w, b} \text{ } & \frac{\hat{\gamma}}{\Vert w \Vert_2} \\
s.t. \text{ } & y^{(i)} (w^T x^{(i)} + b) \ge \hat{\gamma}, \text{ } i = 1, \dots, m \\
\end{align*} $$

Here, we're going to maximize $$\hat{\gamma}/{\Vert w \Vert_2}$$, subject to the functional margins all being at least $$\hat{\gamma}$$. Since the geometric and functional margins are related by $$\gamma = \hat{\gamma}/{\Vert w \Vert_2}$$, this will give us the answer we want. Moreover, we've gotten rid of the constraint $${\Vert w \Vert}_2 = 1$$ that we didn't like. The downside is that we now have a nasty (again, non-convex) objective $$\frac{\hat{\gamma}}{\Vert w \Vert_2}$$ function; and, we still don't have any off-the-shelf software that can solve this form of an optimization problem.

Lets keep going. Recall our earlier discussion that we can add an arbitrary scaling constraint on $$w$$ and $$b$$ without changing anything. This is the key idea we'll use now. We will introduce the scaling constraint that the functional margin of $$w, b$$ with respect to the training set must be $$1$$:

$$ \hat{\gamma} = 1 $$

Since multiplying $$w$$ and $$b$$ by some constant results in the functional margin being multiplied by that same constant, this is indeed a scaling constraint, and can be satisfied by rescaling $$w, b$$. Plugging this into our problem above, and noting that maximizing $$\hat{\gamma}/{\Vert w \Vert_2} = 1/{\Vert w \Vert_2}$$ is the same thing as minimizing $${\Vert w \Vert}_2^2$$, we now have the following optimization problem:

$$ \begin{align*}
\mathop{min}\limits_{\gamma, w, b} \text{ } & \frac{1}{2} {\Vert w \Vert}_2^2 \\
s.t. \text{ } & y^{(i)} (w^T x^{(i)} + b) \ge 1, \text{ } i = 1, \dots, m \\
\end{align*} $$

We've now transformed the problem into a form that can be efficiently solved. The above is an optimization problem with a convex quadratic objective and only linear constraints. Its solution gives us the **optimal margin classifier**. This optimization problem can be solved using commercial quadratic programming (QP) code. You may be familiar with linear programming, which solves optimization problems that have linear objectives and linear constraints. QP software is also widely available, which allows convex quadratic objectives and linear constraints.

While we could call the problem solved here, what we will instead do is make a digression to talk about Lagrange duality. This will lead us to our optimization problem's dual form, which will play a key role in allowing us to use kernels to get optimal margin classifiers to work efficiently in very high dimensional spaces. The dual form will also allow us to derive an efficient algorithm for solving the above optimization problem that will typically do much better than generic QP software.

### Lagrange duality

Lets temporarily put aside SVMs and maximum margin classifiers, and talk about solving constrained optimization problems.

Consider a problem of the following form:

$$ \begin{align*}
\min_{w} \text{ } & f(w) \\
s.t. \text{ } & h_i (w) = 0, \text{ } i = 1, \dots, l \\
\end{align*} $$

Some of you may recall how the method of Lagrange multipliers can be used to solve it. In this method, we define the **Lagrangian** to be

$$ L(w,\beta) = f(w) + \sum_{i=1}^{l} \beta_i h_i(w) $$

Here, the $$\beta_i$$ 's are called the **Lagrange multipliers**. We would then find and set $$L$$'s partial derivatives to zero:

$$ \frac{\partial L}{\partial w_i} = 0 \text{ } \frac{\partial L}{\partial \beta_i} = 0 $$

and solve for $$w$$ and $$\beta$$.

In this section, we will generalize this to constrained optimization problems in which we may have inequality as well as equality constraints. Due to time constraints, we won't really be able to do the theory of Lagrange duality justice in this class, but we will give the main ideas and results, which we will then apply to our optimal margin classifier's optimization problem.

Consider the following, which we'll call the **primal** optimization problem:

$$ \begin{align*}
\mathop{min}\limits_{w} \text{ } & f(w) \\
s.t. \text{ } & g_i (w) \le 0, \text{ } i = 1, \dots, k \\
& h_i (w) = 0, \text{ } i = 1, \dots, l \\
\end{align*} $$

To solve it, we start by defining the **generalized Lagrangian**

$$ L(w, \alpha, \beta) = f(w) + \sum_{i=1}^{k} \alpha_i g_i(w) + \sum_{i=1}^{l} \beta_i h_i(w) $$

Here, the $$\alpha_i$$ 's and $$\beta_i$$ 's are the Lagrange multipliers. Consider the quantity

$$ \theta_P (w) = \mathop{max}\limits_{\alpha, \beta: \alpha_i \ge 0} L(w, \alpha, \beta) $$

Here, the "$$P$$" subscript stands for "primal." Let some $$w$$ be given. If $$w$$ violates any of the primal constraints (i.e., if either $$g_i (w) > 0$$ or $$h_i (w) \neq 0$$ for some $$i$$), then you should be able to verify that

$$ \begin{align*}
\theta_P (w)
&= \mathop{max}\limits_{\alpha, \beta: \alpha_i \ge 0} f(w) + \sum_{i=1}^{k} \alpha_i g_i(w) + \sum_{i=1}^{l} \beta_i h_i(w) \\
&= + \infty \\
\end{align*} $$

Conversely, if the constraints are indeed satisfied for a particular value of $$w$$, then $$\theta_P (w) = f(w)$$. Hence,

$$
\theta_P (w) =
  \begin{cases}
    f(w) & \text{ if $w$ satisfies primal constraints } \\
    + \infty       & \text{ otherwise } \\
   \end{cases}
$$

Thus, $$\theta_P$$ takes the same value as the objective in our problem for all values of $$w$$ that satisfies the primal constraints, and is positive infinity if the constraints are violated. Hence, if we consider the minimization problem

$$ \mathop{min}\limits_{w} \theta_P (w) = \mathop{min}\limits_{w} \mathop{max}\limits_{\alpha, \beta: \alpha_i \ge 0} L(w, \alpha, \beta) $$

we see that it is the same problem (i.e., and has the same solutions as) our original, primal problem. For later use, we also define the optimal value of the objective to be $$p^{\ast} = \min_w \theta_P (w)$$; we call this the **value** of the primal problem.

Now, lets look at a slightly different problem. We define

$$ \theta_D (\alpha, \beta) = \mathop{min}\limits_{w} L(w, \alpha, \beta) $$

Here, the "$$D$$" subscript stands for "dual." Note also that whereas in the definition of $$\theta_P$$ we were optimizing (maximizing) with respect to $$\alpha, \beta$$, here are minimizing with respect to $$w$$.

We can now pose the **dual** optimization problem:

$$ \mathop{max}\limits_{\alpha, \beta: \alpha_i \ge 0} \theta_D (\alpha, \beta) = \mathop{max}\limits_{\alpha, \beta: \alpha_i \ge 0} \mathop{min}\limits_{w} L(w, \alpha, \beta) $$

This is exactly the same as our primal problem shown above, except that the order of the "max" and the "min" are now exchanged. We also define the optimal value of the dual problem's objective to be $$d^{\ast}=\max_{\alpha, \beta: \alpha_i \ge 0} \theta_D (w)$$.

How are the primal and the dual problems related? It can easily be shown that

$$ d^{\ast} = \mathop{max}\limits_{\alpha, \beta: \alpha_i \ge 0} \mathop{min}\limits_{w} L(w, \alpha, \beta) \le \mathop{min}\limits_{w} \mathop{max}\limits_{\alpha, \beta: \alpha_i \ge 0} L(w, \alpha, \beta) = p^{\ast} $$

(You should convince yourself of this; this follows from the "max min" of a function always being less than or equal to the "min max.") However, under certain conditions, we will have

$$ d^{\ast} = p^{\ast} $$

so that we can solve the dual problem in lieu of the primal problem. Lets see what these conditions are.

Suppose $$f$$ and the $$g_i$$ 's are convex, and the $$h_i$$ 's are **affine**. Suppose further that the constraints $$g_i$$ are strictly feasible; this means that there exists some $$w$$ so that $$g_i (w) < 0$$ for all $$i$$.

When $$f$$ has a Hessian, then it is convex if and only if the hessian is positive semi-definite. For instance, $$f(w) = w^Tw$$ is convex; similarly, all linear and affine functions are also convex.

I.e., there exists $$a_i , b_i$$, so that $$h_i (w/) = a_i^T w + b_i$$. **Affine** means the same thing as linear, except that we also allow the extra intercept term $$b_i$$.

Under our above assumptions, there must exist $$w^{\ast}, \alpha^{\ast}, \beta^{\ast}$$ so that $$w^{\ast}$$ is the solution to the primal problem, $$\alpha^{\ast}, \beta^{\ast}$$ are the solution to the dual problem, and moreover $$p^{\ast} = d^{\ast} = L(w^{\ast}, \alpha^{\ast}, \beta^{\ast})$$. Moreover, $$w^{\ast}, \alpha^{\ast}, \beta^{\ast}$$ satisfy the **Karush-Kuhn-Tucker (KKT) conditions**, which are as follows:

$$ \begin{align*}
\frac{\partial}{\partial w_i} L(w^{\ast}, \alpha^{\ast}, \beta^{\ast}) &= 0, \enspace i = 1, \dots, n \\
\frac{\partial}{\partial \beta_i} L(w^{\ast}, \alpha^{\ast}, \beta^{\ast}) &= 0, \enspace i = 1, \dots, l \\
\alpha_i^{\ast} g_i(w^{\ast}) &= 0, \enspace i = 1, \dots, k \\
g_i(w^{\ast}) &\le 0, \enspace i = 1, \dots, k \\
\alpha_i^{\ast} &\ge 0, \enspace i = 1, \dots, k \\
\end{align*} $$

Moreover, if some $$w^{\ast}, \alpha^{\ast}, \beta^{\ast}$$ satisfy the KKT conditions, then it is also a solution to the primal and dual problems.

We draw attention to $$a_i^{\ast} g_i(w^{\ast}) = 0$$, which is called the KKT **dual complementarity** condition. Specifically, it implies that if $$\alpha_i^{\ast} > 0$$, then $$g_i (w^{\ast}) = 0$$. (I.e., the "$$g_i (w) \le 0$$" constraint is **active**, meaning it holds with equality rather than with inequality.) Later on, this will be key for showing that the SVM has only a small number of "support vectors"; the KKT dual complementarity condition will also give us our convergence test when we talk about the SMO algorithm.

### Optimal margin classifiers

Previously, we posed the following (primal) optimization problem for finding the optimal margin classifier:

$$ \begin{align*}
\mathop{min}\limits_{\gamma, w, b} \text{ } & \frac{1}{2} {\Vert w \Vert}_2^2 \\
s.t. \text{ } & y^{(i)} (w^T x^{(i)} + b) \ge 1, \text{ } i = 1, \dots, m \\
\end{align*} $$

We can write the constraints as

$$ g_i(w) = - y^{(i)} (w^T x^{(i)} + b) + 1 \le 0 $$

We have one such constraint for each training example. Note that from the KKT dual complementarity condition, we will have $$\alpha_i > 0$$ only for the training examples that have functional margin exactly equal to one (i.e., the ones corresponding to constraints that hold with equality, $$g_i (w) = 0$$). Consider the figure below, in which a maximum margin separating hyperplane is shown by the solid line.

{% include image.html description="multi-hyperplane" image="machine-learning/multi-hyperplane.png" caption="true"%}

The points with the smallest margins are exactly the ones closest to the decision boundary; here, these are the three points (one negative and two positive examples) that lie on the dashed lines parallel to the decision boundary. Thus, only three of the $$\alpha_i$$ 's - namely, the ones corresponding to these three training examples - will be non-zero at the optimal solution to our optimization problem. These three points are called the **support vectors** in this problem. The fact that the number of support vectors can be much smaller than the size the training set will be useful later.

Lets move on. Looking ahead, as we develop the dual form of the problem, one key idea to watch out for is that we'll try to write our algorithm in terms of only the inner product $$\langle x^{(i)}, x^{(j)} \rangle$$ (think of this as $$(x^{(i)})^T (x^{(j)})$$) between points in the input feature space. The fact that we can express our algorithm in terms of these inner products will be key when we apply the kernel trick.

When we construct the Lagrangian for our optimization problem we have:

$$ L(w, b, \alpha) = \frac{1}{2} {\Vert w \Vert}_2^2 - \sum_{i=1}^{m} \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1] $$

Note that there're only "$$\alpha_i$$" but no "$$\beta_i$$" Lagrange multipliers, since the problem has only inequality constraints.

Lets find the dual form of the problem. To do so, we need to first minimize $$L(w, b, \alpha)$$ with respect to $$w$$ and $$b$$ (for fixed $$\alpha$$), to get $$\theta_D$$, which we'll do by setting the derivatives of $$L$$ with respect to $$w$$ and $$b$$ to zero. We have:

$$ \nabla_w L(w, b, \alpha) = w - \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)} = 0 $$

This implies that

$$ w = \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)} $$

As for the derivative with respect to $$b$$, we obtain

$$ \frac{\partial}{\partial b} L(w, b, \alpha) = \sum_{i=1}^{m} a_i y^{(i)} = 0 $$

If we take the definition of $$w = \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)}$$ and plug that back into the Lagrangian, and simplify, we get

$$ L(w, b, \alpha) = \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} y^{(i)} y^{(j)} \alpha_i \alpha_j (x^{(i)})^T (x^{(j)}) - b \sum_{i=1}^{m} \alpha_i y^{(i)}  $$

But from Equation $$\sum_{i=1}^{m} a_i y^{(i)} = 0$$, the last term must be zero, so we obtain

$$ L(w, b, \alpha) = \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} y^{(i)} y^{(j)} \alpha_i \alpha_j (x^{(i)})^T x^{(j)} $$

Recall that we got to the equation above by minimizing $$L$$ with respect to $$w$$ and $$b$$. Putting this together with the constraints $$\alpha_i \ge 0$$ (that we always had) and the constraint $$\sum_{i=1}^{m} a_i y^{(i)} = 0$$, we obtain the following dual optimization problem:

$$ \begin{align*}
max_\alpha \text{ } & W(\alpha) = \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \\
s.t. \text{ } & \alpha_i \ge 0, \text{ } i = 1, \dots, m \\
& \sum_{i=1}^{m} \alpha_i y^{(i)} = 0 \\
\end{align*} $$

You should also be able to verify that the conditions required for $$p^{\ast} = d^{\ast}$$ and the KKT conditions to hold are indeed satisfied in our optimization problem. Hence, we can solve the dual in lieu of solving the primal problem. Specifically, in the dual problem above, we have a maximization problem in which the parameters are the $$\alpha_i$$ 's. We'll talk later about the specific algorithm that we're going to use to solve the dual problem, but if we are indeed able to solve it (i.e., find the $$\alpha$$'s that maximize $$W(\alpha)$$ subject to the constraints), then we can use $$w = \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)}$$ to go back and find the optimal $$w$$'s as a function of the $$\alpha$$'s. Having found $$w^{\ast}$$, by considering the primal problem, it is also straightforward to find the optimal value for the intercept term $$b$$ as

$$ b^{\ast} = - \frac{max_{i:y^{(i)}=-1} {w^{\ast}}^{T} x^{(i)} + min_{i:y^{(i)}=1} {w^{\ast}}^{T} x^{(i)}}{2} $$

Before moving on, lets also take a more careful look at $$w = \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)}$$, which gives the optimal value of $$w$$ in terms of (the optimal value of) $$\alpha$$. Suppose we've fit our model's parameters to a training set, and now wish to make a prediction at a new point input $$x$$. We would then calculate $$w^T x + b$$, and predict $$y = 1$$ if and only if this quantity is bigger than zero. But using $$w = \sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)}$$, this quantity can also be written:

$$ \begin{align*}
w^T x + b
&= \bigg(\sum_{i=1}^{m} \alpha_i y^{(i)} x^{(i)} \bigg)^T x + b \\
&= \sum_{i=1}^{m} \alpha_i y^{(i)} \langle x^{(i)}, x \rangle + b \\
\end{align*} $$

Hence, if we've found the $$\alpha_i$$ 's, in order to make a prediction, we have to calculate a quantity that depends only on the inner product between $$X$$ and the points in the training set. Moreover, we saw earlier that the $$\alpha_i$$ 's will all be zero except for the support vectors. Thus, many of the terms in the sum above will be zero, and we really need to find only the inner products between $$X$$ and the support vectors (of which there is often only a small number) in order to make our prediction.

By examining the dual form of the optimization problem, we gained significant insight into the structure of the problem, and were also able to write the entire algorithm in terms of only inner products between input feature vectors. In the next section, we will exploit this property to apply the kernels to our classification problem. The resulting algorithm, **support vector machines**, will be able to efficiently learn in very high dimensional spaces.

### Kernels

Back in our discussion of linear regression, we had a problem in which the input $$x$$ was the living area of a house, and we considered performing regression using the features $$x$$, $$x^2$$ and $$x^3$$ say to obtain a cubic function. To distinguish between these two sets of variables, we'll call the "original" input value the input **attributes** of a problems. When that is mapped to some new set of quantities that are then passed to the learning algorithm, we'll call those new quantities the input **features**. We will also let $$\phi$$ denote the **feature mapping**, which maps from the attributes to the features. For instance, in our example, we had

$$
\phi(x) =
\begin{bmatrix}
  x \\
  x^2 \\
  x^3 \\
\end{bmatrix}
$$

Rather than applying SVMs using the original input attributes $$x$$, we may instead want to learn using some features $$\phi(x)$$. To do so, we simply need to go over our previous algorithm, and replace $$X$$ everywhere in it with $$\phi(x)$$.

Since the algorithm can be written entirely in terms of the inner products $$\langle x,z \rangle$$, this means that we would replace all those inner products with $$\langle \phi(x),\phi(z) \rangle$$. Specificically, given a feature mapping $$\phi$$, we define the corresponding **Kernel** to be

$$ K(x,z) = \phi(x)^T \phi(z) $$

Then, everywhere we previously had $$\langle x,z \rangle$$ in our algorithm, we could simply replace it with $$K(x,z)$$, and our algorithm would now be learning using the features $$\phi$$.

Now, given $$\phi$$, we could easily compute $$K(x,z)$$ by finding $$\phi(x)$$ and $$\phi(z)$$ and taking their inner product. But what's more interesting is that often, $$K(x,z)$$ may be very inexpensive to calculate, even though $$\phi(x)$$ itself may be very expensive to calculate (perhaps because it is an extremely high dimensional vector). In such settings, by using in our algorithm an efficient way to calculate $$K(x,z)$$, we can get SVMs to learn in the high dimensional feature space given by $$\phi$$, but without ever having to explicitly find or represent vectors $$\phi(x)$$.

Lets see an example. Suppose $$x,z \in R^n$$, and consider

$$ K(x,z) = (x^T z)^2 $$

We can also write this as

$$ \begin{align*}
K(x,z)
&= \bigg( \sum_{i=1}^{n} x_i z_i \bigg) \bigg( \sum_{j=1}^{n} x_i z_i \bigg) \\
&= \sum_{i=1}^{n} \sum_{j=1}^{n} (x_i x_j) (z_i z_j) \\
\end{align*} $$

Thus, we see that $$K(x,z) = \phi(x)^T \phi(z)$$, where the feature mapping $$\phi$$ is given (shown here for the case of $$n = 3$$) by

$$
\phi(x) =
\begin{bmatrix}
  x_1 x_1 \\
  x_1 x_2 \\
  x_1 x_3 \\
  x_2 x_1 \\
  x_2 x_2 \\
  x_2 x_3 \\
  x_3 x_1 \\
  x_3 x_2 \\
  x_3 x_3 \\
\end{bmatrix}
$$

Note that whereas calculating the high-dimensional $$\phi(x)$$ requires $$O(n^2)$$ time, finding $$K(x,z)$$ takes only $$O(n)$$ time - linear in the dimension of the input  attributes.

For a related kernel, also consider

$$ \begin{align*}
K(x,z)
&= (x^T z + c)^2 \\
&= \sum_{i=1}^{n} \sum_{j=1}^{n} (x_i x_j) (z_i z_j) + \sum_{i=1}^{n} (\sqrt{2c} x_i) (\sqrt{2c} z_i) + c^2 \\
\end{align*} $$

This corresponds to the feature mapping (again shown for $$n = 3$$)

$$
\phi(x) =
\begin{bmatrix}
  x_1 x_1 \\
  x_1 x_2 \\
  x_1 x_3 \\
  x_2 x_1 \\
  x_2 x_2 \\
  x_2 x_3 \\
  x_3 x_1 \\
  x_3 x_2 \\
  x_3 x_3 \\
  \sqrt{2c} x_1 \\
  \sqrt{2c} x_2 \\
  \sqrt{2c} x_3 \\
  c \\
\end{bmatrix}
$$

and the parameter $$c$$ controls the relative weighting between the $$x_i$$ first order and the $$x_i x_j$$ second order terms.

More broadly, the kernel $$K(x,z) = (x^T z + c)^d$$ corresponds to a feature mapping to an $$\binom{n+d}{d}$$ feature space, corresponding of all monomials of the form $$x_{i1} x_{i2} \dots x_{ik}$$ that are up to order $$d$$. However, despite working in this $$O(n^d)$$ - dimensional space, computing $$K(x,z)$$ still takes only $$O(n)$$ time, and hence we never need to explicitly represent feature vectors in this very high dimensional feature space.

Now, lets talk about a slightly different view of kernels. Intuitively, if $$\phi(x)$$ and $$\phi(z)$$ are close together, then we might expect $$K(x,z) = \phi(x)^T \phi(z)$$ to be large. Conversely, if $$\phi(x)$$ and $$\phi(z)$$ are far apart - say nearly orthogonal to each other - then $$K(x,z) = \phi(x)^T \phi(z)$$ will be small. So, we can think of $$K(x,z)$$ as some measurement of how similar are $$\phi(x)$$ and $$\phi(z)$$, or of how similar are $$X$$ and $$z$$.

Given this intuition, suppose that for some learning problem that you're working on, you've come up with some function $$K(x,z)$$ that you think might be a reasonable measure of how similar $$X$$ and $$z$$ are. For instance, perhaps you chose

$$ K(x,z) = exp \bigg( -\frac{ {\Vert x - z \Vert}_2^2 }{2 \sigma^2} \bigg) $$

This is a resonable measure of $$X$$ and $$z$$ 's similarity, and is close to $$1$$ when $$X$$ and $$z$$ are close, and near $$0$$ when $$X$$ and $$z$$ are far apart. Can we use this definition of $$K$$ as the kernel in an SVM? In this particular example, the answer is yes. (This kernel is called the **Gaussian kernel**, and corresponds to an infinite dimensional feature mapping $$\phi$$.) But more broadly, given some function $$K$$, how can we tell if it's a valid kernel; i.e., can we tell if there is some feature mapping $$\phi$$ so that $$K(x,z) = \phi(x)^T \phi(z)$$ for all $$x, z$$.

Suppose for now that $$K$$ is indeed a valid kernel corresponding to some feature mapping $$\phi$$. Now, consider some finite set of m points (not necessarily the training set) $$[x^{(1)}, \dots, x^{(m)}]$$, and let a square, $$m$$-by-$$m$$ matrix $$K$$ be defined so that its $$(i, j)$$ - entry is given by $$K_{ij} = K(x^{(i)}, x^{(j)})$$. This matrix is called the **Kernel matrix**. Note that we've overloaded the notation and used $$K$$ to denote both the kernel function $$K(x,z)$$ and the kernel matrix $$K$$, due to their obvious close relationship.

Now, if $$K$$ is a valid Kernel, then $$K_{ij} = K(x^{(i)}, x^{(j)}) = \phi(x^{(i)})^T \phi(x^{(j)}) = \phi(x^{(j)})^T \phi(x^{(i)}) = K(x^{(j)}, x^{(i)}) = K_{ji}$$, and hence $$K$$ must be symmetric. Moreover, letting $$\phi_k (x)$$ denote the $$k$$-th coordinate of the vector $$\phi(x)$$, we find that for any vector $$z$$, we have

$$ \begin{align*}
z^T K z
&= \sum_{i} \sum_{j} z_i K_{ij} z_j \\
&= \sum_{i} \sum_{j} z_i \phi(x^{(i)})^T \phi(x^{(j)}) z_j \\
&= \sum_{i} \sum_{j} z_i \sum_{k} {(\phi(x^{(i)}))}_k {(\phi(x^{(j)}))}_k z_j \\
&= \sum_{i} \sum_{j} \sum_{k} z_i {(\phi(x^{(i)}))}_k {(\phi(x^{(j)}))}_k z_j \\
&= \sum_{k} \bigg(\sum_{i} z_i {(\phi(x^{(i)}))}_k \bigg)^2 \\
&\ge 0 \\
\end{align*} $$

Since $$z$$ was arbitrary, this shows that $$K$$ is positive semi-definite ($K \ge 0$).

Hence, we've shown that if $$K$$ is a valid kernel (i.e., if it corresponds to some feature mapping $$\phi$$), then the corresponding Kernel matrix $$K \in R^{m×m}$$ is symmetric positive semidefinite. More generally, this turns out to be not only a necessary, but also a sufficient, condition for $$K$$ to be a valid kernel (also called a **Mercer kernel**). The following result is due to **Mercer**.

**Theorem (Mercer)**. Let $$K : R^n \times R^n \to R$$ be given. Then for $$K$$ to be a valid (Mercer) kernel, it is necessary and sufficient that for any $$[x^{(1)}, \dots, x^{(m)}]$$, ($$m < \infty$$), the corresponding kernel matrix is symmetric positive semi-definite.

Given a function $$K$$, apart from trying to find a feature mapping $$\phi$$ that corresponds to it, this theorem therefore gives another way of testing if it is a valid kernel.

Consider the digit recognition problem, in which given an image (16x16 pixels) of a handwritten digit (0-9), we have to figure out which digit it was. Using either a simple polynomial kernel $$K(x,z) = (x^T z)^ d$$ or the Gaussian kernel, SVMs were able to obtain extremely good performance on this problem. This was particularly surprising since the input attributes $$X$$ were just a 256-dimensional vector of the image pixel intensity values, and the system had no prior knowledge about vision, or even about which pixels are adjacent to which other ones. Another example that we briefly talked about in lecture was that if the objects $$X$$ that we are trying to classify are strings, then it seems hard to construct a reasonable, "small" set of features for most learning algorithms, especially if different strings have different lengths. However, consider letting $$\phi(x)$$ be a feature vector that counts the number of occurrences of each length-$$k$$ substring in $$x$$. If we're considering strings of english alphabets, then there're $$26^k$$ such strings. Hence, $$\phi(x)$$ is a $$26^k$$ dimensional vector; even for moderate values of $$k$$, this is probably too big for us to efficiently work with. (e.g., $$26^4 ≈ 460000$$.) However, using (dynamic programming-ish) string matching algorithms, it is possible to efficiently compute $$K(x,z) = \phi(x)^T \phi(z)$$, so that we can now implicitly work in this $$26^k$$ -dimensional feature space, but without ever explicitly computing feature vectors in this space.

The application of kernels to support vector machines should already be clear and so we won't dwell too much longer on it here. Keep in mind however that the idea of kernels has significantly broader applicability than SVMs. Specifically, if you have any learning algorithm that you can write in terms of only inner products $$\langle x,z \rangle$$ between input attribute vectors, then by replacing this with $$K(x,z)$$ where $$K$$ is a kernel, you can "magically" allow your algorithm to work efficiently in the high dimensional feature space corresponding to $$K$$. For instance, this kernel trick can be applied with the perceptron to to derive a kernel perceptron algorithm. Many of the algorithms that we'll see later in this class will also be amenable to this method, which has come to be known as the "kernel trick".

### Regularization and the non-separable case

The derivation of the SVM as presented so far assumed that the data is linearly separable. While mapping data to a high dimensional feature space via $$\phi$$ does generally increase the likelihood that the data is separable, we can't guarantee that it always will be so. Also, in some cases it is not clear that finding a separating hyperplane is exactly what we'd want to do, since that might be susceptible to outliers. For instance, the left figure below shows an optimal margin classifier, and when a single outlier is added in the upper-left region (right figure), it causes the decision boundary to make a dramatic swing, and the resulting classifier has a much smaller margin.

{% include image.html description="regularization-and-non-separable" image="machine-learning/regularization-and-non-separable.png" caption="true"%}

To make the algorithm work for non-linearly separable datasets as well as be less sensitive to outliers, we reformulate our optimization (using $$\ell_1$$ **regularization**) as follows:

$$ \begin{align*}
\mathop{min}\limits_{\gamma, w, b} \text{ } & \frac{1}{2} {\Vert w \Vert}^2 + C \sum_{i=1}^{m} \xi_i \\
s.t. \text{ } & y^{(i)} (w^T x^{(i)} + b) \ge 1 - \xi_i, \text{ } i = 1, \dots, m \\
& \xi_i \ge 0, \text{ } i = 1, \dots, m \\
\end{align*} $$

Thus, examples are now permitted to have (functional) margin less than $$1$$, and if an example whose functional margin is $$1 - \xi_i$$, we would pay a cost of the objective function being increased by $$C \xi_i$$. The parameter $$C$$ controls the relative weighting between the twin goals of making the $${\Vert w \Vert}^2$$ large (which we saw earlier makes the margin small) and of ensuring that most examples have functional margin at least $$1$$.

As before, we can form the Lagrangian:

$$ L(w,b,\xi,\alpha,r) = \frac{1}{2} w^T w + C \sum_{i=1}^{m} \xi_i - \sum_{i=1}^{m} \alpha_i [y^{(i)}(w^T x^{(i)} + b) - 1 + \xi_i] - \sum_{i=1}^{m} r_i \xi_i $$

Here, the $$\alpha_i$$ 's and $$r_i$$ 's are our Lagrange multipliers (constrained to be $$\ge 0$$). We won't go through the derivation of the dual again in detail, but after setting the derivatives with respect to $$w$$ and $$b$$ to zero as before, substituting them back in, and simplifying, we obtain the following dual form of the problem:

$$ \begin{align*}
max_\alpha \text{ } & W(\alpha) = \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \\
s.t. \text{ } & 0 \le \alpha_i \le C, \text{ } i = 1, \dots, m \\
& \sum_{i=1}^{m} \alpha_i y^{(i)} = 0 \\
\end{align*} $$

As before, we also have that $$w$$ can be expressed in terms of the $$\alpha_i$$ 's, so that after solving the dual problem, we can continue to make our predictions. Note that, somewhat surprisingly, in adding $$\ell_1$$ regularization, the only change to the dual problem is that what was originally a constraint that $$0 \le \alpha_i$$ has now become $$0 \le \alpha_i \le C$$. The calculation for $$b^{\ast}$$ also has to be modified;

Also, the KKT dual-complementarity conditions (which in the next section will be useful for testing for the convergence of the SMO algorithm) are:

$$
\begin{align}
\alpha_i = 0 & \implies \text{ } y^{(i)} (w^T x^{(i)} + b) \ge 1 \\
\alpha_i = C & \implies \text{ } y^{(i)} (w^T x^{(i)} + b) \le 1 \\
0 < \alpha_i < C & \implies \text{ } y^{(i)} (w^T x^{(i)} + b) = 1 \\
\end{align}
$$

Now, all that remains is to give an algorithm for actually solving the dual problem, which we will do in the next section.

### The SMO algorithm

The SMO (sequential minimal optimization) algorithm, due to John Platt, gives an efficient way of solving the dual problem arising from the derivation of the SVM. Partly to motivate the SMO algorithm, and partly because it’s interesting in its own right, lets first take another digression to talk about the coordinate ascent algorithm.

#### Coordinate ascent

Consider trying to solve the unconstrained optimization problem

$$ \mathop{max}\limits_{\alpha} \text{ } W(\alpha_1, \alpha_2, \dots, \alpha_m) $$

Here, we think of $$W$$ as just some function of the parameters $$\alpha_i$$ 's, and for now ignore any relationship between this problem and SVMs. We’ve already seen two optimization algorithms, gradient ascent and Newton’s method. The new algorithm we’re going to consider here is called **coordinate ascent**:

Loop until convergence $$\Rightarrow$$ Loop for $$i=1$$ to $$m$$

$$
\alpha_i := argmax_{\hat{\alpha}_i} \text{ } W(\alpha_1, \dots ,\alpha_{i-1}, \hat{\alpha}_i, \alpha_{i+1}, \dots ,\alpha_m) \\
$$

Thus, in the innermost loop of this algorithm, we will hold all the variables except for some $$\alpha_i$$ fixed, and reoptimize $$W$$ with respect to just the parameter $$\alpha_i$$. In the version of this method presented here, the inner-loop reoptimizes the variables in order $$\alpha_1, \alpha_2, \dots, \alpha_m, \alpha_1, \alpha_2, \dots$$ (A more sophisticated version might choose other orderings; for instance, we may choose the next variable to update according to which one we expect to allow us to make the largest increase in $$W(\alpha)$$.)

When the function $$W$$ happens to be of such a form that the "arg max" in the inner loop can be performed efficiently, then coordinate ascent can be a fairly efficient algorithm. Here’s a picture of coordinate ascent in action:

{% include image.html description="coordinate-ascent" image="machine-learning/coordinate-ascent.png" caption="true"%}

The ellipses in the figure are the contours of a quadratic function that we want to optimize. Coordinate ascent was initialized at $$(2, −2)$$, and also plotted in the figure is the path that it took on its way to the global maximum. Notice that on each step, coordinate ascent takes a step that’s parallel to one of the axes, since only one variable is being optimized at a time.

#### SMO

We close off the discussion of SVMs by sketching the derivation of the SMO algorithm. Some details will be left to the homework, and for others you may refer to the paper excerpt handed out in class.

Here's the (dual) optimization problem that we want to solve:

$$
\begin{align}
max_\alpha \text{ } & W(\alpha) = \sum_{i=1}^{m} \alpha_i - \frac{1}{2} \sum_{i=1}^{m} \sum_{j=1}^{m} y^{(i)} y^{(j)} \alpha_i \alpha_j \langle x^{(i)}, x^{(j)} \rangle \\
s.t. \text{ } & 0 \le \alpha_i \le C, i = 1, \dots, m \\
& \sum_{i=1}^{m} \alpha_i y^{(i)} = 0 \\
\end{align}
$$

Lets say we have set of $$\alpha_i$$ 's that satisfy the above constraints. Now, suppose we want to hold $$\alpha_2, \dots, \alpha_m$$ fixed, and take a coordinate ascent step and reoptimize the objective with respect to $$\alpha_1$$ . Can we make any progress. The answer is no, because the last constraint ensures that

$$ \alpha_1 y^{(1)} = - \sum_{i=2}^{m} \alpha_i y^{(i)} $$

Or we equivalently have

$$ \alpha_1 = - \frac{\sum_{i=2}^{m} \alpha_i y^{(i)}}{y^{(1)}} $$

(This step used the fact that $$y^{(1)} \in [−1, 1]$$, and hence $$(y^{(1)})^2 = 1$$.) Hence, $$\alpha_1$$ is exactly determined by the other $$\alpha_i$$ 's, and if we were to hold $$\alpha_2, \dots, \alpha_m$$ fixed, then we can’t make any change to $$\alpha_1$$ without violating the last constraint in the optimization problem.

Thus, if we want to update some subject of the $$\alpha_i$$ 's, we must update at least two of them simultaneously in order to keep satisfying the constraints. This motivates the SMO algorithm, which simply does the following:

Repeat till convergence:

* Select some pair $$\alpha_i$$ and $$\alpha_j$$ to update next (using a heuristic that tries to pick the two that will allow us to make the biggest progress towards the global maximum).

* Reoptimize $$W(\alpha)$$ with respect to $$\alpha_i$$ and $$\alpha_j$$ , while holding all the other $$\alpha_k$$ 's ($$k \neq i, j$$) fixed.

To test for convergence of this algorithm, we can check whether the KKT conditions are satisfied to within some tol. Here, tol is the convergence tolerance parameter, and is typically set to around $$0.01$$ to $$0.001$$.

The key reason that SMO is an efficient algorithm is that the update to $$\alpha_i$$, $$\alpha_j$$ can be computed very efficiently. Lets now briefly sketch the main ideas for deriving the efficient update.

Lets say we currently have some setting of the $$\alpha_i$$ 's that satisfy the constraints, and suppose we’ve decided to hold $$\alpha_3, \dots, \alpha_m$$ fixed, and want to reoptimize $$W (\alpha_1, \alpha_2, \dots, \alpha_m)$$ with respect to $$\alpha_1$$ and $$\alpha_2$$ (subject to the constraints). We require that

$$ \alpha_1 y^{(1)} + \alpha_2 y^{(2)} = - \sum_{i-3}^{m} \alpha_i y^{(i)} $$

Since the right hand side is fixed (as we’ve fixed $$\alpha_3, \dots, \alpha_m$$), we can just let it be denoted by some constant $$\zeta$$:

$$ \alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta $$

We can thus picture the constraints on $$\alpha_1$$ and $$\alpha_2$$ as follows:

{% include image.html description="smo" image="machine-learning/smo.png" caption="true"%}

We know that $$\alpha_1$$ and $$\alpha_2$$ must lie within the box $$[0, C] \times [0, C]$$ shown. Also plotted is the line $$\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta$$, on which we know $$\alpha_1$$ and $$\alpha_2$$ must lie. Note also that, from these constraints, we know $$L \le \alpha_2 \le H$$; otherwise, $$(\alpha_1, \alpha_2)$$ can’t simultaneously satisfy both the box and the straight line constraint. In this example, $$L = 0$$. But depending on what the line $$\alpha_1 y^{(1)} + \alpha_2 y^{(2)} = \zeta$$ looks like, this won’t always necessarily be the case; but more generally, there will be some lower-bound $$L$$ and some upper-bound $$h$$ on the permissable values for $$\alpha_2$$ that will ensure that $$\alpha_1, \alpha_2$$ lie within the box $$[0, C] \times [0, C]$$.

We can also write $$\alpha_1$$ as a function of $$\alpha_2$$ :

$$ \alpha_1 = \frac{\zeta - \alpha_2 y^{(2)}}{y^{(1)}} $$

(Check this derivation yourself; we again used the fact that $$y^{(1)} \in [−1, 1]$$ so that $$(y^{(1)})^2 = 1$$.) Hence, the objective $$W(\alpha)$$ can be written

$$ W(\alpha_1, \alpha_2, \dots, \alpha_m) = W \bigg( \frac{\zeta - \alpha_2 y^{(2)}}{y^{(1)}}, \alpha_2, \dots, \alpha_m \bigg) $$

Treating $$\alpha_3, \dots, \alpha_m$$ as constants, you should be able to verify that this is just some quadratic function in $$\alpha_2$$. I.e., this can also be expressed in the form $$a \alpha_2^2 + b \alpha_2 + c$$ for some appropriate $$a$$, $$b$$, and $$c$$. If we ignore the "box" constraints (or, equivalently, that $$L \le \alpha_2 \le H$$), then we can easily maximize this quadratic function by setting its derivative to zero and solving.

We’ll let $$\alpha_2^{new,unclipped}$$ denote the resulting value of $$\alpha_2$$. You should also be able to convince yourself that if we had instead wanted to maximize $$W$$ with respect to $$\alpha_2$$ but subject to the box constraint, then we can find the resulting value optimal simply by taking $$\alpha_2^{new,unclipped}$$ and "clipping" it to lie in the $$[L, H]$$ interval, to get

$$
\alpha_2^{new} =
\begin{cases}
H & \text{ if } \alpha_2^{new,unclipped} > H \\
\alpha_2^{new,unclipped} & \text{ if } L \le \alpha_2^{new,unclipped} \le H \\
L & \text{ if } \alpha_2^{new,unclipped} < L \\
\end{cases}
$$

Finally, having found the $$\alpha_2^{new}$$, we can go back and find the optimal value of $$\alpha_1^{new}$$$.

There’re a couple more details that are quite easy but that we’ll leave you to read about yourself in Platt’s paper: One is the choice of the heuristics used to select the next $$\alpha_i, \alpha_j$$ to update; the other is how to update b as the SMO algorithm is run.

# Learning Theory

## Bias/Variance Tradeoff

When talking about linear regression, we discussed the problem of whether to fit a simple model such as the linear $$y = \theta_0 + \theta_1 x$$, or a more complex model such as the polynomial $$y = \theta_0 + \theta_1 x + \dots + \theta_5 x^5$$. We saw the following example:

{% include image.html description="LWR" image="machine-learning/LWR.png" caption="true"%}

Fitting a $$5$$th order polynomial to the examples (rightmost figure) did not result in a good model. Specifically, even though the $$5$$th order polynomial did a very good job predicting $$Y$$ from $$X$$ for the examples in the training set, we do not expect the model shown to be a good one for predicting the examples not in the training set. In other words, what’s has been learned from the training set does not generalize well to other examples. The **generalization error** of a hypothesis is its expected error on examples not necessarily in the training set.

Both the models in the leftmost and the rightmost figures above have large generalization error. However, the problems that the two models suffer from are very different. If the relationship between $$Y$$ and $$X$$ is not linear, then even if we were fitting a linear model to a very large amount of training data, the linear model would still fail to accurately capture the structure in the data. Informally, we define the **bias** of a model to be the expected generalization error even if we were to fit it to a very (say, infinitely) large training set. Thus, for the problem above, the linear model suffers from large bias, and may underfit (i.e., fail to capture structure exhibited by) the data.

Apart from bias, there's a second component to the generalization error, consisting of the **variance** of a model fitting procedure. Specifically, when fitting a 5th order polynomial as in the rightmost figure, there is a large risk that we’re fitting patterns in the data that happened to be present in our small, finite training set, but that do not reflect the wider pattern of the relationship between $$X$$ and $$y$$. This could be, say, because in the training set we just happened by chance to get a slightly more-expensive-than-average here, and a slightly less-expensive-than-average there, and so on. By fitting these spurious patterns in the training set, we might again obtain a model with large generalization error. In this case, we say the model has large variance.

Often, there is a tradeoff between bias and variance. If our model is too simple and has very few parameters, then it may have large bias (but small variance); if it is too complex and has very many parameters, then it may suffer from large variance (but have smaller bias). In the example above, fitting a quadratic function does better than either of the extremes of a first or a fifth order polynomial.

## Preliminaries

In this set of notes, we begin our foray into learning theory. Apart from being interesting and enlightening in its own right, this discussion will also help us hone our intuitions and derive rules of thumb about how to best apply learning algorithms in different settings. We will also seek to answer a few questions: First, can we make formal the bias/variance tradeoff that was just discussed? The will also eventually lead us to talk about model selection methods, which can, for instance, automatically decide what order polynomial to fit to a training set. Second, in machine learning it’s really generalization error that we care about, but most learning algorithms fit their models to the training set. Why should doing well on the training set tell us anything about generalization error? Specifically, can we relate error on the training set to generalization error? Third and finally, are there conditions under which we can actually prove that learning algorithms will work well?

We start with two simple but very useful lemmas.

Lemma. (**The union bound**). Let $$A_1, A_2, \dots, A_k$$$ be $$k$$ different events (that may not be independent). Then

$$ P(A_1 \cup \dots \cup A_k) \leq P(A_1) + \dots + P(A_k) $$

In probability theory, the union bound is usually stated as an axiom (and thus we won’t try to prove it), but it also makes intuitive sense: The probability of any one of $$k$$ events happening is at most the sums of the probabilities of the $$k$$ different events.

Lemma. (**Hoeffding inequality**). Let $$Z_1, \dots, Z_m$$ be $$m$$ independent and identically distributed (IID) random variables drawn from a Bernoulli($$\phi$$) distribution. I.e., $$P(Z_i = 1) = \phi$$, and $$P(Z_i = 0) = 1 − \phi$$. Let $$\hat{\phi} = (1/m) \sum_{i=1}^m Z_i$$ be the mean of these random variables, and let any $$\gamma > 0$$ be fixed. Then

$$ P(\mid \phi − \hat{\phi} \mid > \gamma) \leq 2 exp(−2 \gamma^2 m) $$

This lemma (which in learning theory is also called the **Chernoff bound**) says that if we take $$\hat{\phi}$$ - the average of $$m$$ Bernoulli($$\phi$$) random variables - to be our estimate of $$\phi$$, then the probability of our being far from the true value is small, so long as $$m$$ is large. Another way of saying this is that if you have a biased coin whose chance of landing on heads is $$\phi$$, then if you toss it $$m$$ times and calculate the fraction of times that it came up heads, that will be a good estimate of $$\phi$$ with high probability (if $$m$$ is large).

Using just these two lemmas, we will be able to prove some of the deepest and most important results in learning theory.

To simplify our exposition, lets restrict our attention to binary classification in which the labels are $$y \in [0, 1]$$. Everything we’ll say here generalizes to other, including regression and multi-class classification, problems.

We assume we are given a training set $$S = [(x^{(i)}, y^{(i)}); i = 1, \dots, m]$$ of size $$m$$, where the training examples $$(x^{(i)}, y^{(i)})$$ are drawn iid from some probability distribution $$D$$. For a hypothesis $$h$$, we define the **training error** (also called the **empirical risk** or **empirical error** in learning theory) to be

$$ \hat{\varepsilon}(h) = \frac{1}{m} \sum_{i=1}^{m} 1 \{ h(x^{(i)}) \neq y^{(i)} \} $$

This is just the fraction of training examples that $$h$$ misclassifies. When we want to make explicit the dependence of $$\hat{\varepsilon}(h)$$ on the training set $$S$$, we may also write this a $$\hat{\varepsilon}_{S}(h)$$. We also define the generalization error to be

$$ \varepsilon(h) = P_{(x,y) \sim D} (h(x) \neq y) $$

I.e. this is the probability that, if we now draw a new example $$(x, y)$$ from the distribution $$D$$, $$h$$ will misclassify it.

Note that we have assumed that the training data was drawn from the same distribution $$D$$ with which we’re going to evaluate our hypotheses (in the definition of generalization error). This is sometimes also referred to as one of the **PAC** assumptions.

Consider the setting of linear classification, and let $$h_\theta (x) = 1 [ \theta^T x \geq 0 ]$$. What’s a reasonable way of fitting the parameters $$\theta$$? One approach is to try to minimize the training error, and pick

$$ \hat{\theta} = \mathop{argmin}\limits_{\theta} \text{ } \hat{\varepsilon} (h_{\theta}) $$

We call this process **empirical risk minimization** (ERM), and the resulting hypothesis output by the learning algorithm is $$\hat{h} = h_{\hat{\theta}}$$. We think of ERM as the most basic learning algorithm, and it will be this algorithm that we focus on in these notes. (Algorithms such as logistic regression can also be viewed as approximations to empirical risk minimization.)

In our study of learning theory, it will be useful to abstract away from the specific parameterization of hypotheses and from issues such as whether we’re using a linear classifier. We define the **hypothesis class** $$h$$ used by a learning algorithm to be the set of all classifiers considered by it. For linear classification, $$H = [h_{\theta} : h_{\theta} (x) = 1[\theta^T x \geq 0], \theta \in R^{n+1}]$$ is thus the set of all classifiers over $$X$$ (the domain of the inputs) where the decision boundary is linear. More broadly, if we were studying, say, neural networks, then we could let $$h$$ be the set of all classifiers representable by some neural network architecture.

Empirical risk minimization can now be thought of as a minimization over the class of functions $$H$$, in which the learning algorithm picks the hypothesis:

$$ \hat{h} = \mathop{argmin}\limits_{h \in H} \text{ } \hat{\varepsilon} (h) $$

## The case of finite H

Lets start by considering a learning problem in which we have a finite hypothesis class $$H = [h_1, \dots, h_k]$$ consisting of $$k$$ hypotheses. Thus, $$h$$ is just a set of $$k$$ functions mapping from $$X$$ to [0, 1], and empirical risk minimization selects $$\hat{h}$$ to be whichever of these $$k$$ functions has the smallest training error.

We would like to give guarantees on the generalization error of $$\hat{h}$$. Our strategy for doing so will be in two parts: First, we will show that $$\hat{\varepsilon}(h)$$ is a reliable estimate of $$\varepsilon(h)$$ for all $$h$$. Second, we will show that this implies an upper-bound on the generalization error of $$\hat{h}$$.

Take any one, fixed, $$h_i \in H$$. Consider a Bernoulli random variable $$Z$$ whose distribution is defined as follows. We’re going to sample $$(x, y) \sim D$$. Then, we set $$Z = 1[h_i (x) \neq y]$$. I.e., we’re going to draw one example, and let $$Z$$ indicate whether $$h_i$$ misclassifies it. Similarly, we also define $$Z_j = 1[h_i (x^{(j)}) \neq y^{(j)}]$$. Since our training set was drawn iid from $$D$$, $$Z$$ and the $$Z_j$$ ’s have the same distribution.

We see that the misclassification probability on a randomly drawn example - that is, $$\varepsilon(h)$$ - is exactly the expected value of $$Z$$ (and $$Z_j$$). Moreover, the training error can be written

$$ \hat{\varepsilon} (h_i) = \frac{1}{m} \sum_{j=1}^{m} Z_j $$

Thus, $$\hat{\varepsilon}(h_i)$$ is exactly the mean of the m random variables $$Z_j$$ that are drawn iid from a Bernoulli distribution with mean $$\varepsilon(h_i)$$. Hence, we can apply the Hoeffding inequality, and obtain

$$ P(\mid \varepsilon(h_i) - \hat{\varepsilon}(h_i) \mid > \gamma) \leq 2 exp(−2 \gamma^2 m) $$

This shows that, for our particular $$h_i$$, training error will be close to generalization error with high probability, assuming $$m$$ is large. But we don’t just want to guarantee that $$\varepsilon(h_i)$$ will be close to $$\hat{\varepsilon}(h_i)$$ (with high probability) for just only one particular $$h_i$$. We want to prove that this will be true for simultaneously for all $$h \in H$$. To do so, let $$A_i$$ denote the event that $$\mid \varepsilon(h_i) - \hat{\varepsilon}(h_i) \mid > \gamma$$. We’ve already show that, for any particular $$A_i$$, it holds true that $$P(A_i) \leq 2 exp(−2 \gamma^2 m)$$. Thus, using the union bound, we have that

$$ \begin{align*}
P(\exists h \in H. \mid \varepsilon(h_i) - \hat{\varepsilon}(h_i) \mid > \gamma) &= P(A_1 \cup \dots \cup A_k) \\
& \leq \sum_{i=1}^{k} P(A_i) \\
& \leq \sum_{i=1}^{k} 2 exp(-2 \gamma^2 m) \\
&= 2k \text{ } exp(-2 \gamma^2 m) \\
\end{align*} $$

If we subtract both sides from $$1$$, we find that

$$ \begin{align*}
P(\neg \exists h \in H. \mid \varepsilon(h_i) - \hat{\varepsilon}(h_i) \mid > \gamma) &= P( \forall h \in H. \mid \varepsilon(h_i) - \hat{\varepsilon}(h_i) \mid \leq \gamma ) \\
& \geq 1 - 2k \text{ } exp(-2 \gamma^2 m) \\
\end{align*} $$

So, with probability at least $$1 - 2k \text{ } exp(−2 \gamma^2 m)$$, we have that $$\varepsilon(h)$$ will be within $$\gamma$$ of $$\hat\varepsilon(h)$$ for all $$h \in H$$. This is called a **uniform convergence** result, because this is a bound that holds simultaneously for all (as opposed to just one) $$h \in H$$.

In the discussion above, what we did was, for particular values of $$m$$ and $$\gamma$$, given a bound on the probability that, for some $$h \in H, \mid \varepsilon(h_i) - \hat{\varepsilon}(h_i) \mid > \gamma$$. There are three quantities of interest here: $$m$$, $$\gamma$$, and the probability of error; we can bound either one in terms of the other two.

For instance, we can ask the following question: Given $$\gamma$$ and some $$\delta > 0$$, how large must $$m$$ be before we can guarantee that with probability at least $$1 - \delta$$, training error will be within $$\gamma$$ of generalization error? By setting $$\delta = 2k \text{ } exp(−2 \gamma^2 m)$$ and solving for $$m$$, we find that if

$$ m \geq \frac{1}{2 \gamma^2} log \frac{2k}{\delta} $$

then with probability at least $$1 - \delta$$, we have that $$\mid \varepsilon(h) - \hat{\varepsilon}(h) \mid \leq \gamma$$ for all $$h \in H$$. (Equivalently, this show that the probability that $$\mid \varepsilon(h) - \hat{\varepsilon}(h) \mid > \gamma$$ for some $$h \in H$$ is at most $$\delta$$.) This bound tells us how many training examples we need in order make a guarantee. The training set size $$m$$ that a certain method or algorithm requires in order to achieve a certain level of performance is also called the algorithm’s **sample complexity**.

The key property of the bound above is that the number of training examples needed to make this guarantee is only logarithmic in $$k$$, the number of hypotheses in $$H$$. This will be important later.

Similarly, we can also hold $$m$$ and $$\delta$$ fixed and solve for $$\gamma$$ in the previous equation, and show that with probability $$1 - \delta$$, we have that for all $$h \in H$$

$$ \mid \hat{\varepsilon}(h) - \varepsilon(h) \mid \leq \sqrt{\frac{1}{2m} log \frac{2k}{\delta}} $$

Now, lets assume that uniform convergence holds, i.e., that $$\mid \varepsilon(h) - \hat{\varepsilon}(h) \mid \leq \gamma$$ for all $$h \in H$$. What can we prove about the generalization of our learning algorithm that picked $$\hat{h} = argmin_{h \in H} \hat{\varepsilon}(h)$$?

Define $$h^{\ast} = argmin_{h \in H} \varepsilon(h)$$ to be the best possible hypothesis in $$H$$. Note that $$h^{\ast}$$ is the best that we could possibly do given that we are using $$H$$, so it makes sense to compare our performance to that of $$h^{\ast}$$. We have:

$$ \begin{align*}
\varepsilon(\hat{h}) &\leq \hat{\varepsilon}(\hat{h}) + \gamma \\
&\leq \hat{\varepsilon}(h^{\ast}) + \gamma \\
&\leq \varepsilon(h^{\ast}) + 2 \gamma \\
\end{align*} $$

The first line used the fact that $$\mid \varepsilon(\hat{h}) - \hat{\varepsilon}(\hat{h}) \mid \leq \gamma$$ (by our uniform convergence assumption). The second used the fact that $$\hat{h}$$ was chosen to minimize $$\hat{\varepsilon}(h)$$, and hence $$\hat{\varepsilon}(\hat{h}) \leq \hat{\varepsilon}(h)$$ for all $$h$$, and in particular $$\hat{\varepsilon}(\hat{h}) \leq \hat{\varepsilon}(h^{\ast})$$. The third line used the uniform convergence assumption again, to show that $$\hat{\varepsilon}(h^{\ast}) \leq \varepsilon(h^{\ast}) + \gamma$$. So, what we’ve shown is the following: If uniform convergence occurs, then the generalization error of $$\hat{h}$$ is at most $$2\gamma$$ worse than the best possible hypothesis in $$H$$!

Lets put all this together into a theorem.

**Theorem**. Let $$\mid H \mid = k$$, and let any $$m$$, $$\delta$$ be fixed. Then with probability at least $$1 - \delta$$, we have that

$$ \varepsilon(\hat{h}) \leq \Big( \mathop{min}\limits_{h \in H} \varepsilon(h) \Big) + 2 \sqrt{\frac{1}{2m} log \frac{2k}{\delta}} $$

This is proved by letting $$\gamma$$ equal the $$\sqrt{.}$$ term, using our previous argument that uniform convergence occurs with probability at least $$1 - \delta$$, and then noting that uniform convergence implies $$\varepsilon(\hat{h})$$ is at most $$2\gamma$$ higher than $$\varepsilon(h^{\ast}) = min_{h \in H} \varepsilon(h)$$.

This also quantifies what we were saying previously saying about the bias/variance tradeoff in model selection. Specifically, suppose we have some hypothesis class $$H$$, and are considering switching to some much larger hypothesis class $$H \subseteq H'$$. If we switch to $$H'$$, then the first term $$min_{h} \varepsilon(h)$$ can only decrease (since we’d then be taking a min over a larger set of functions). Hence, by learning using a larger hypothesis class, our bias can only decrease. However, if $$k$$ increases, then the second $$2 \sqrt{·}$$ term would also increase. This increase corresponds to our variance increasing when we use a larger hypothesis class.

By holding $$\gamma$$ and $$\delta$$ fixed and solving for $$m$$ like we did before, we can also obtain the following sample complexity bound:

**Corollary**. Let $$\mid H \mid = k$$, and let any $$\delta$$, $$\gamma$$ be fixed. Then for $$\varepsilon(\hat{h}) \leq min_{h \in H} \varepsilon(h) + 2\gamma$$ to hold with probability at least $$1 - \delta$$, it suffices that

$$ \begin{align*}
m &\geq \frac{1}{2\gamma^2} log \frac{2k}{\delta} \\
&= O (\frac{1}{\gamma^2} log \frac{k}{\delta}) \\
\end{align*} $$

## The case of infinite H

We have proved some useful theorems for the case of finite hypothesis classes. But many hypothesis classes, including any parameterized by real numbers (as in linear classification) actually contain an infinite number of functions. Can we prove similar results for this setting?

Lets start by going through something that is not the right argument. Better and more general arguments exist, but this will be useful for honing our intuitions about the domain.

Suppose we have an $$h$$ that is parameterized by $$d$$ real numbers. Since we are using a computer to represent real numbers, and IEEE double-precision floating point (double’s in C) uses $$64$$ bits to represent a floating point number, this means that our learning algorithm, assuming we’re using double-precision floating point, is parameterized by $$64d$$ bits. Thus, our hypothesis class really consists of at most $$k = 2^{64d}$$ different hypotheses. From the Corollary at the end of the previous section, we therefore find that, to guarantee $$\varepsilon(\hat{h}) \leq \varepsilon(h^{\ast}) + 2 \gamma$$, with to hold with probability at least $$1 - \delta$$, it suffices that $$m > O(\frac{1}{\gamma^2} log \frac{2^{64d}}{\delta}) = O(\frac{d}{\gamma^2} log \frac{1}{\delta}) = O_{\gamma, \delta}(d)$$. (The $$\gamma, \delta$$ subscripts are to indicate that the last big $$O$$ is hiding constants that may depend on $$\gamma$$ and $$\delta$$). Thus, the number of training examples needed is at most linear in the parameters of the model.

The fact that we relied on $$64$$-bit floating point makes this argument not entirely satisfying, but the conclusion is nonetheless roughly correct: If what we’re going to do is try to minimize training error, then in order to learn well using a hypothesis class that has $$d$$ parameters, generally we’re going to need a linear number of training examples in $$d$$.

(At this point, it’s worth noting that these results were proved for an algorithm that uses empirical risk minimization. Thus, while the linear dependence of sample complexity on $$d$$ does generally hold for most discriminative learning algorithms that try to minimize training error or some approximation to training error, these conclusions do not always apply as readily to discriminative learning algorithms. Giving good theoretical guarantees on many non-ERM learning algorithms is still an area of active research).

The other part of our previous argument that’s slightly unsatisfying is that it relies on the parameterization of $$H$$. Intuitively, this doesn’t seem like it should matter: We had written the class of linear classifiers as $$h_\theta (x) = 1[\theta_0 + \theta_1 x_1 + \dots + \theta_n x_n \geq 0]$$, with $$n + 1$$ parameters $$\theta_0, \dots, \theta_n$$. But it could also be written $$h_{u,v} (x) = 1[(u_0^2 - v_0^2) + (u_1^2 - v_1^2) x_1 + \dots + (u_n^2 - v_n^2) x_n \geq 0]$$ with $$2n + 2$$ parameters $$u_i, v_i$$. Yet, both of these are just defining the same $$H$$: The set of linear classifiers in $$n$$ dimensions.

To derive a more satisfying argument, lets define a few more things.

Given a set $$S = [x^{(1)}, \dots, x^{(d)}]$$ (no relation to the training set) of points $$x^{(i)} \in X$$, we say that $$h$$ **shatters** $$S$$ if $$h$$ can realize any labeling on $$S$$. I.e., if for any set of labels $$[y^{(1)}, \dots, y^{(d)}]$$, there exists some $$h \in H$$ so that $$h(x^{(i)}) = y^{(i)}$$ for all $$i = 1, \dots, d$$.

Given a hypothesis class $$H$$, we then define its **Vapnik-Chervonenkis dimension**, written $$VC(H)$$, to be the size of the largest set that is shattered by $$H$$. (If $$h$$ can shatter arbitrarily large sets, then $$VC(H) = \infty$$).

For instance, consider the following set of three points:

{% include image.html description="VC3" image="machine-learning/vc-3.png" caption="true"%}

Can the set $$h$$ of linear classifiers in two dimensions ($$h(x) = 1[\theta_0 + \theta_0 x_1 + \theta_2 x_2 \geq 0]$$) can shatter the set above? The answer is yes. Specifically, we see that, for any of the eight possible labelings of these points, we can find a linear classifier that obtains "zero training error" on them:

{% include image.html description="vc-shatter-3" image="machine-learning/vc-shatter-3.png" caption="true"%}

Moreover, it is possible to show that there is no set of $$4$$ points that this hypothesis class can shatter. Thus, the largest set that $$h$$ can shatter is of size $$3$$, and hence $$VC(H) = 3$$.

More generally $$VC(H) = n + 1$$, where is $$h$$ is $$n$$ dimensions linear classifier hypothesis class.

Note that the $$VC$$ dimension of $$h$$ here is $$3$$ even though there may be sets of size $$3$$ that it cannot shatter. For instance, if we had a set of three points lying in a straight line (left figure), then there is no way to find a linear separator for the labeling of the three points shown below (right figure):

{% include image.html description="VC-Not-Shatter-3" image="machine-learning/vc-not-shatter-3.png" caption="true"%}

In order words, under the definition of the $$VC$$ dimension, in order to prove that $$VC(H)$$ is at least $$d$$, we need to show only that there’s at least one set of size $$d$$ that $$h$$ can shatter.

The following theorem, due to Vapnik, can then be shown. (This is, many would argue, the most important theorem in all of learning theory.)

**Theorem.** Let $$h$$ be given, and let $$d = VC(H)$$. Then with probability at least $$1 - \delta$$, we have that for all $$h \in H$$

$$ \mid \varepsilon(h) - \hat{\varepsilon(h)} \mid \leq O(\sqrt{\frac{d}{m} log \frac{m}{d} + \frac{1}{m} log \frac{1}{\delta}}) $$

Thus, with probability at least $$1 - \delta$$, we also have that:

$$\varepsilon(\hat{h}) \leq \varepsilon(h^{\ast}) + O(\sqrt{\frac{d}{m} log \frac{m}{d} + \frac{1}{m} log \frac{1}{\delta}}) $$

In other words, if a hypothesis class has finite $$VC$$ dimension, then uniform convergence occurs as $$m$$ becomes large. As before, this allows us to give a bound on $$\varepsilon(h)$$ in terms of $$\varepsilon(h^{\ast})$$. We also have the following corollary:

**Corollary.** For $$\mid \varepsilon(h) - \hat{\varepsilon(h)} \mid \leq \gamma$$ to hold for all $$h \in H$$ (and hence $$\varepsilon(\hat{h}) \leq \varepsilon(h^{\ast}) + 2\gamma)$$ with probability at least $$1 − \delta$$, it suffices that $$m = O_{\gamma, \delta}(d)$$.

In other words, the number of training examples needed to learn well using $$h$$ is linear in the $$VC$$ dimension of $$H$$. It turns out that, for most hypothesis classes, the $$VC$$ dimension (assuming a "reasonable" parameterization) is also roughly linear in the number of parameters. Putting these together, we conclude that (for an algorithm that tries to minimize training error) the number of training examples needed is usually roughly linear in the number of parameters of $$H$$.

# Regularization and model selection

Suppose we are trying select among several different models for a learning problem. For instance, we might be using a polynomial regression model $$h_{\theta}(x) = g(\theta_0 + \theta_1 x + \theta_2 x^2 + \dots + \theta_k x^k)$$, and wish to decide if $$k$$ should be $$0, 1, \dots, \text{ or } 10$$. How can we automatically select a model that represents a good tradeoff between bias and variance? Alternatively, suppose we want to automatically choose the bandwidth parameter $$\tau$$ for locally weighted regression, or the parameter $$C$$ for our $$\ell_1$$-regularized SVM. How can we do that?

For the sake of concreteness, in these notes we assume we have some finite set of models $$M = [M_1, \dots, M_d]$$ that we’re trying to select among. For instance, in our first example above, the model $$M_i$$ would be an $$i$$-th order polynomial regression model. Alternatively, if we are trying to decide between using an SVM, a neural network or logistic regression, then $$m$$ may contain these models.

If we are trying to choose from an infinite set of models, say corresponding to the possible values of the bandwidth $$\tau \in R^{+}$$, we may discretize $$\tau$$ and consider only a finite number of possible values for it. More generally, most of the algorithms described here can all be viewed as performing optimization search in the space of models, and we can perform this search over infinite model classes as well.

## Cross validation

Lets suppose we are, as usual, given a training set $$S$$. Given what we know about empirical risk minimization, here’s what might initially seem like a algorithm, resulting from using empirical risk minimization for model selection:

* Train each model $$M_i$$ on $$S$$, to get some hypothesis $$h_i$$.

* Pick the hypotheses with the smallest training error.

This algorithm does not work. Consider choosing the order of a polynomial. The higher the order of the polynomial, the better it will fit the training set $$S$$, and thus the lower the training error. Hence, this method will always select a high-variance, high-degree polynomial model, which we saw previously is often poor choice.

Here’s an algorithm that works better. In **hold-out cross validation** (also called **simple cross validation**), we do the following:

* Randomly split $$S$$ into $$S_{train}$$ (say, 70% of the data) and $$S_{cv}$$ (the remaining 30%). Here, $$S_{cv}$$ is called the hold-out cross validation set.

* Train each model $$M_i$$ on $$S_{train}$$ only, to get some hypothesis $$h_i$$.

* Select and output the hypothesis $$h_i$$ that had the smallest error $$\hat{\varepsilon}_{S_{cv}} (h_i)$$ on the hold out cross validation set. (Recall, $$\hat{\varepsilon}_{S_{cv}} (h)$$ denotes the empirical error of $$h$$ on the set of examples in $$S_{cv}$$.)

$$ {\hat{\varepsilon}}_{S_{cv}} (h_i) $$

By testing on a set of examples $$S_{cv}$$ that the models were not trained on, we obtain a better estimate of each hypothesis $$h_i$$'s true generalization error, and can then pick the one with the smallest estimated generalization error. Usually, somewhere between $$1/4 - 1/3$$ of the data is used in the hold out cross validation set, and 30% is a typical choice.

Optionally, last step in the algorithm may also be replaced with selecting the model $$M_i$$ according to $$argmin_i \hat{\varepsilon}_{S_{cv}} (h_i)$$, and then retraining $$M_i$$ on the entire training set $$S$$. (This is often a good idea, with one exception being learning algorithms that are be very sensitive to perturbations of the initial conditions and/or data. For these methods, $$M_i$$ doing well on $$S$$ train does not necessarily mean it will also do well on $$S_{cv}$$, and it might be better to forgo this retraining step.)

The disadvantage of using hold out cross validation is that it "wastes" about 30% of the data. Even if we were to take the optional step of retraining the model on the entire training set, it’s still as if we’re trying to find a good model for a learning problem in which we had 0.7$$m$$ training examples, rather than $$m$$ training examples, since we’re testing models that were trained on only 0.7$$m$$ examples each time. While this is fine if data is abundant and/or cheap, in learning problems in which data is scarce (consider a problem with $$m$$ = 20, say), we’d like to do something better.

Here is a method, called $$k$$-**fold cross validation**, that holds out less data each time:

* Randomly split $$S$$ into $$k$$ disjoint subsets of $$m/k$$ training examples each. Lets call these subsets $$S_1, \dots ,S_k$$.

* For each model $$M_i$$, we evaluate it as follows:

    - For $$j = 1, \dots, k$$

        - Train the model $$M_i$$ on $$S_1 \cup \dots \cup S_{j−1} \cup S_{j+1} \cup \dots S_k$$ (i.e., train on all the data except $$S_j$$) to get some hypothesis $$h_{ij}$$. Test the hypothesis $$h_{ij}$$ on $$S_j$$, to get $$\hat{\varepsilon}_{S_j}(h_{ij})$$.

    - The estimated generalization error of model $$M_i$$ is then calculated as the average of the $$\hat{\varepsilon}_{S_j}(h_{ij})$$'s (averaged over $$j$$).

* Pick the model $$M_i$$ with the lowest estimated generalization error, and retrain that model on the entire training set $$S$$. The resulting hypothesis is then output as our final answer.

A typical choice for the number of folds to use here would be $$k = 10$$. While the fraction of data held out each time is now $$1/k$$ - much smaller than before - this procedure may also be more computationally expensive than hold-out cross validation, since we now need train to each model $$k$$ times.

While $$k = 10$$ is a commonly used choice, in problems in which data is really scarce, sometimes we will use the extreme choice of $$k = m$$ in order to leave out as little data as possible each time. In this setting, we would repeatedly train on all but one of the training examples in $$S$$, and test on that held-out example. The resulting $$m = k$$ errors are then averaged together to obtain our estimate of the generalization error of a model. This method has its own name; since we’re holding out one training example at a time, this method is called **leave-one-out cross validation**.

Finally, even though we have described the different versions of cross validation as methods for selecting a model, they can also be used more simply to evaluate a single model or algorithm. For example, if you have implemented some learning algorithm and want to estimate how well it performs for your application (or if you have invented a novel learning algorithm and want to report in a technical paper how well it performs on various test sets), cross validation would give a reasonable way of doing so.

## Feature Selection

One special and important case of model selection is called feature selection. To motivate this, imagine that you have a supervised learning problem where the number of features $$n$$ is very large (perhaps $$n \gg m$$), but you suspect that there is only a small number of features that are "relevant" to the learning task. Even if you use the a simple linear classifier (such as the perceptron) over the $$n$$ input features, the VC dimension of your hypothesis class would still be $$O(n)$$, and thus overfitting would be a potential problem unless the training set is fairly large.

In such a setting, you can apply a feature selection algorithm to reduce the number of features. Given $$n$$ features, there are $$2^n$$ possible feature subsets (since each of the $$n$$ features can either be included or excluded from the subset), and thus feature selection can be posed as a model selection problem over $$2^n$$ possible models. For large values of $$n$$, it's usually too expensive to explicitly enumerate over and compare all $$2^n$$ models, and so typically some heuristic search procedure is used to find a good feature subset. The following search procedure is called **forward search**:

* Initialize $$F = \emptyset$$.

* Repeat:

    - For $$i = 1, \dots, n$$ if $$i \notin F$$, let $$F_{i} = F \cup [i]$$, and use some version of cross validation to evaluate features $$F_i$$. (I.e., train your learning algorithm using only the features in $$F_i$$, and estimate its generalization error.)

    - Set $$F$$ to be the best feature subset found on above step

* Select and output the best feature subset that was evaluated during the entire search procedure.

The outer loop of the algorithm can be terminated either when $$F = [1, \dots, n]$$ is the set of all features, or when $$\mid F \mid$$ exceeds some preset threshold (corresponding to the maximum number of features that you want the algorithm to consider using).

This algorithm described above one instantiation of **wrapper model feature selection**, since it is a procedure that "wraps" around your learning algorithm, and repeatedly makes calls to the learning algorithm to evaluate how well it does using different feature subsets. Aside from forward search, other search procedures can also be used. For example, **backward search** starts off with $$F = [1, \dots, n]$$ as the set of all features, and repeatedly deletes features one at a time (evaluating single-feature deletions in a similar manner to how forward search evaluates single-feature additions) until $$F = \emptyset$$.

Wrapper feature selection algorithms often work quite well, but can be computationally expensive given how that they need to make many calls to the learning algorithm. Indeed, complete forward search (terminating when $$F = {1, \dots, n}$$) would take about $$O(n^2)$$ calls to the learning algorithm.

**Filter feature selection** methods give heuristic, but computationally much cheaper, ways of choosing a feature subset. The idea here is to compute some simple score $$S(i)$$ that measures how informative each feature $$x_i$$ is about the class labels $$y$$. Then, we simply pick the $$k$$ features with the largest scores $$S(i)$$.

One possible choice of the score would be define $$S(i)$$ to be (the absolute value of) the correlation between $$x_i$$ and $$y$$, as measured on the training data. This would result in our choosing the features that are the most strongly correlated with the class labels. In practice, it is more common (particularly for discrete-valued features $$x_i$$) to choose $$S(i)$$ to be the **mutual information** $$MI(x_i, y)$$ between $$x_i$$ and $$y$$:

$$MI(x_i, y) = \sum_{x_i \in [0,1]} \sum_{y \in [0,1]} p(x_i, y) log \frac{p(x_i, y)}{p(x_i)p(y)}$$

(The equation above assumes that $$x_i$$ and $$Y$$ are binary-valued; more generally the summations would be over the domains of the variables.) The probabilities above $$p(x_i, y)$$, $$p(x_i)$$ and $$p(y)$$ can all be estimated according to their empirical distributions on the training set.

To gain intuition about what this score does, note that the mutual information can also be expressed as a **Kullback-Leibler (KL) divergence**:

$$MI(x_i, y) = KL\Big(p(x_i, y) \mid \mid p(x_i)p(y)\Big)$$

KL-divergence gives a measure of how different the probability distributions $$p(x_i, y)$$ and $$p(x_i )p(y)$$ are. If $$x_i$$ and $$Y$$ are independent random variables, then we would have $$p(x_i, y) = p(x_i)p(y)$$, and the KL-divergence between the two distributions will be zero. This is consistent with the idea if $$x_i$$ and $$Y$$ are independent, then $$x_i$$ is clearly very "non-informative" about $$y$$, and thus the score $$S(i)$$ should be small. Conversely, if $$x_i$$ is very "informative" about $$y$$, then their mutual information $$MI(x_i, y)$$ would be large.

One final detail: Now that you've ranked the features according to their scores $$S(i)$$, how do you decide how many features $$k$$ to choose? Well, one standard way to do so is to use cross validation to select among the possible values of $$k$$. For example, when applying naive Bayes to text classification a problem where $$n$$, the vocabulary size, is usually very large using this method to select a feature subset often results in increased classifier accuracy.

## Bayesian statistics and regularization
