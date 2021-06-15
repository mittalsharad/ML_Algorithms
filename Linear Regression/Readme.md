# Linear Regression

When starting the journey of Machine Learning, the very first algorithm that everyone starts with is the **Linear Regression**.

It is the most basic and and most well understood algorithm in Supervised ML algorithms. As the name suggests, it is used in Regression tasks i.e. it is used when we have continuous values in our dataset like car or house prices, room temperature data etc.

Also, as the name suggests, It tries to find a **best fit linear line between the independent variables(x) and dependent variable(y)**

**For example:**

Let's consider a data set consisting of cars, where for each car you had the number of miles a car had driven along with its price. In this case, let’s assume that you are trying to train a machine learning system that takes in the information about each car, namely the number of miles driven along with its associated price. Here for a given car, the miles driven is the input and the price is the output. This data could be represented as (X, Y ) coordinates. 


Plotting this data will give us something like this:


![image](https://user-images.githubusercontent.com/22586467/121916938-3a0ee880-cd52-11eb-944d-83f7eafcc18c.png)

In the above case, it seems that there is a linear relationship between the miles driven and the price. If we try to fit a resonable looking fit line, it would look something like this:

![image](https://user-images.githubusercontent.com/22586467/121917824-1ac48b00-cd53-11eb-801b-7caa1c5b8f2e.png)


We could describe the model for our car price dataset as a mathematical function of the form: `F (X) = a_1 · X + a_0`

Here, a_1 and a_0 are called **weights** and these are the values that determines how our function behaves on different inputs. 

All supervised learning algorithms have some set of weights that determine how the algorithm behaves on different inputs, 
and determining the right weights is really at the core of what we call learning.

## Types of Linear Regression Models

**1. Simple Linear Regression**: the equation we saw above is what we call a Simple Linear Regression. It is called Simple because we have only one independent
feature(miles driven in above example) and the model has to find the linear relationship of it with the dependent variable (car price).

Equation of Simple Linear Regression, ![image](https://user-images.githubusercontent.com/22586467/121919441-a559ba00-cd54-11eb-849f-4eef611e4f3e.png)

Here,
- bo is the intercept
- b1 is coefficient or slope
- x is the independent variable
- y is the dependent variable

**2. Multiple Linear Regression**: if we have more than one independent feature in our dataset, than the model is called Multiple Linear Regression.

Equation of Multiple Linear Regression: ![image](https://user-images.githubusercontent.com/22586467/121919829-0b464180-cd55-11eb-8ad1-e21153e60ebb.png)

Here,
- b1,b2,b3,b4…,bn are coefficients or slopes of the independent variables x1,x2,x3,x4…,xn
- y is the dependent variable.

**A Linear Regression model’s main aim is to find the best fit linear line and the optimal values of intercept and coefficients such that the error is minimized.**

**Error**: error is the difference between the actual values and the predicted values. And the goal of our function is to minimize this error.

Let's understand this with the help of an example:

![image](https://user-images.githubusercontent.com/22586467/121920432-a9d2a280-cd55-11eb-8aa1-e73cac246ab9.png)

Image source:Statistical tools for high-throughput data analysis

In the above diagram,

- x is our dependent variable which is plotted on the x-axis and y is the dependent variable which is plotted on the y-axis.
- Black dots are the data points i.e the actual values.
- bo is the intercept which is 10 and b1 is the slope of the x variable.
- The blue line is the best fit line predicted by the model i.e the predicted values lie on the blue line.

The vertical distance between the data point and the regression line is known as `error or residual`. Each data point has one residual and the sum of all the differences is known as the Sum of Residuals/Errors. 

-> Residual/Error = Actual values – Predicted Values

-> Sum of Residuals/Errors = Sum(Actual- Predicted Values)

-> Square of Sum of Residuals/Errors = (Sum(Actual- Predicted Values))2


## Cost Function
Training and evaluating a machine learning model involves using something called a cost function. In the case of supervised learning, a cost function is a measure of how much the predicted labels outputted by our model deviate from the true labels. Ideally we would like the deviation between the two to be small, and so we want to minimize the value of our cost function.

The cost function helps us to figure out the best possible values for a_0 and a_1 which would provide the best fit line for the data points. Since we want the best values for a_0 and a_1, we convert this search problem into a minimization problem where we would like to minimize the error between the predicted value and the actual value.

<p align="center">
  <img width="460" height="300" src="https://user-images.githubusercontent.com/22586467/121921615-c91dff80-cd56-11eb-8cf0-db456bbfd66c.png">
</p>
<p align="center">
    <em>Minimization and Cost Function</em>
</p>

We choose the above function to minimize the difference between the predicted values and ground truth measures the error difference. We square the error difference and sum over all data points and divide that value by the total number of data points. This provides the average squared error over all the data points. Therefore, this cost function is also known as the `Mean Squared Error(MSE)` function. Now, using this MSE function we are going to change the values of a_0 and a_1 such that the MSE value settles at the minima.

## Gradient Descent
How do we actually compute the weights that achieve the minimal values of our cost?

Gradient descent is a method of updating a_0 and a_1 to reduce the cost function(MSE). The idea is that we start with some values for a_0 and a_1 and then we change these values iteratively to reduce the cost. Gradient descent helps us on how to change the values.

![image](https://user-images.githubusercontent.com/22586467/121923348-76454780-cd58-11eb-9c9f-200ff14254fd.png)

To draw an analogy, imagine a pit in the shape of U and you are standing at the topmost point in the pit and your objective is to reach the bottom of the pit. There is a catch, you can only take a discrete number of steps to reach the bottom. If you decide to take one step at a time you would eventually reach the bottom of the pit but this would take a longer time. If you choose to take longer steps each time, you would reach sooner but, there is a chance that you could overshoot the bottom of the pit and not exactly at the bottom. In the gradient descent algorithm, the number of steps you take is the learning rate. This decides on how fast the algorithm converges to the minima.

Sometimes the cost function can be a non-convex function where you could settle at a local minima but for linear regression, it is always a convex function

![image](https://user-images.githubusercontent.com/22586467/121923472-9a088d80-cd58-11eb-8522-a1629a65c0ae.png)

You may be wondering how to use gradient descent to update a_0 and a_1. To update a_0 and a_1, we take gradients from the cost function. To find these gradients, we take partial derivatives with respect to a_0 and a_1. Now, to understand how the partial derivatives are found below you would require some calculus but if you don’t, it is alright. You can take it as it is.

![image](https://user-images.githubusercontent.com/22586467/121924060-2f0b8680-cd59-11eb-90b5-0802c6899801.png)

![image](https://user-images.githubusercontent.com/22586467/121924081-33d03a80-cd59-11eb-9a0d-bce30ebf7c6a.png)

The partial derivates are the gradients and they are used to update the values of a_0 and a_1. Alpha is the learning rate which is a hyperparameter that you must specify. A smaller learning rate could get you closer to the minima but takes more time to reach the minima, a larger learning rate converges sooner but there is a chance that you could overshoot the minima.


## Assumptions of Linear Regression
1. **Linearity:** It states that the dependent variable Y should be linearly related to independent variables. 

2. **Normality:** The X and Y variables should be normally distributed.

3. **Homoscedasticity:** The variance of the error terms should be constant i.e the spread of residuals should be constant for all values of X. 

4. **Independence/No Multicollinearity:** The variables should be independent of each other i.e no correlation should be there between the independent variables. 

5. The **error terms should be normally distributed**. 

6. **No Autocorrelation:** The error terms should be independent of each other.


## Performance Evaluation of Regression Model
The performance of the regression model can be evaluated by using various metrics like R square, adjusted R- square, Mean Squared Error (MSE), Root Mean Squared Error (RMSE)

**1. R-Square (R2):**
It determines how much of the total variation in Y (dependent variable) is explained by the variation in X (independent variable). Mathematically, it can be written as:
![image](https://user-images.githubusercontent.com/22586467/121925704-d0dfa300-cd5a-11eb-97a5-dd58b3fdcbf5.png)

The value of R-square is always between 0 and 1, where 0 means that the model does not explain any variability in the target variable (Y) and 1 meaning it explains full variability in the target variable.

**2. Adjusted R-square**
The only drawback of R2 is that if new predictors (X) are added to our model, R2 only increases or remains constant but it never decreases. We can not judge that by increasing complexity of our model, are we making it more accurate?

That is why, we use `Adjusted R-Square`.

The Adjusted R-Square is the modified form of R-Square that has been adjusted for the number of predictors in the model. It incorporates model’s degree of freedom. The adjusted R-Square only increases if the new term improves the model accuracy.
![image](https://user-images.githubusercontent.com/22586467/121925994-18fec580-cd5b-11eb-86a4-1daad620fc72.png)

Where,
- R2 = Sample R square
- p = Number of predictors
- N = total sample size

**3. Mean Squared Error (MSE):** Another Common metric for evaluation is Mean squared error which is the mean of the squared difference of actual vs predicted values.
![image](https://user-images.githubusercontent.com/22586467/121930145-a6dcaf80-cd5f-11eb-880b-ee673980a7c7.png)

**4. Root Mean Squared Error (RMSE):** It is the root of MSE i.e Root of the mean difference of Actual and Predicted values. RMSE penalizes the large errors whereas MSE doesn’t.
![image](https://user-images.githubusercontent.com/22586467/121930199-bb20ac80-cd5f-11eb-8950-718889b21e44.png)



## Underfitting and Overfitting
When we fit a model, we try to find the optimised, best-fit line, which can describe the impact of the change in the independent variable on the change in the dependent variable by keeping the error term minimum. While fitting the model, there can be 2 events which will lead to the bad performance of the model. These events are
- Underfitting 
- Overfitting

**Underfitting** 
Underfitting is the condition where the model could not fit the data well enough. The under-fitted model leads to low accuracy of the model. Therefore, the model is unable to capture the relationship, trend or pattern in the training data. Underfitting of the model could be avoided by using more data, or by optimising the parameters of the model.

**Overfitting**
Overfitting is the opposite case of underfitting, i.e., when the model predicts very well on training data and is not able to predict well on test data or validation data. The main reason for overfitting could be that the model is memorising the training data and is unable to generalise it on test/unseen dataset. Overfitting can be reduced by doing feature selection or by using regularisation techniques. 

![image](https://user-images.githubusercontent.com/22586467/121924519-9c1f1c00-cd59-11eb-9455-845460e89e3e.png)

## Bias and Variance
What does that bias and variance actually mean? Let us understand this by an example of archery targets.

![image](https://user-images.githubusercontent.com/22586467/121926138-464b7380-cd5b-11eb-9e8f-364753b0e5db.png)

Let’s say we have model which is very accurate, therefore the error of our model will be low, meaning a low bias and low variance as shown in first figure. All the data points fit within the bulls-eye. Similarly we can say that if the variance increases, the spread of our data point increases which results in less accurate prediction. And as the bias increases the error between our predicted value and the observed values increases.

Now how this bias and variance is balanced to have a perfect model? Take a look at the image below and try to understand.

![image](https://user-images.githubusercontent.com/22586467/121926196-55322600-cd5b-11eb-85dc-824bfc1144cf.png)

As we add more and more parameters to our model, its complexity increases, which results in increasing variance and decreasing bias, i.e., overfitting. So we need to find out one optimum point in our model where the decrease in bias is equal to increase in variance. In practice, there is no analytical way to find this point. So how to deal with high variance or high bias?

To overcome underfitting or high bias, we can basically add new parameters to our model so that the model complexity increases, and thus reducing high bias.

In Simple terms, `Bias is the error of training data` and `Variance is the error of test data`.

Now, how can we overcome Overfitting for a regression model?

Basically there are two methods to overcome overfitting,
- Reduce the model complexity
- Regularization

## Regularization
In regularization, what we do is normally we keep the same number of features, but reduce the magnitude of the coefficients.

These seek to both minimize the sum of the squared error of the model on the training data (using ordinary least squares) but also to reduce the complexity of the model (like the number or absolute size of the sum of all coefficients in the model).

Two popular examples of regularization procedures for linear regression are:

**Ridge Regression:** where Ordinary Least Squares is modified to also minimize the squared absolute sum of the coefficients (called L2 regularization).

Cost function for Ridge Regression: ![image](https://user-images.githubusercontent.com/22586467/121927263-76474680-cd5c-11eb-9b82-a3d779a8821f.png)

Here, we come across an extra term, which is known as the penalty term. λ given here, is actually denoted by alpha parameter in the ridge function. So by changing the values of alpha, we are basically controlling the penalty term. Higher the values of alpha, bigger is the penalty and therefore the magnitude of coefficients are reduced.
Here, 
- It shrinks the parameters, therefore it is mostly used to prevent multicollinearity.
- It reduces the model complexity by coefficient shrinkage.
- It uses L2 regularization technique.
- It Penalizes the Steeper slopes
- Slope will shrink but will never be 0


**Lasso Regression(Least Absolute Shrinkage Selector Operator):** where Ordinary Least Squares is modified to also minimize the absolute sum of the coefficients (called L1 regularization). Here, we penalize the steeper slopes. 

Cost function for Lasso Regression: ![image](https://user-images.githubusercontent.com/22586467/121927315-865f2600-cd5c-11eb-9f49-d63c02f87438.png)
 
- Lasso selects only some feature while reduces the coefficients of others to zero. This property is known as feature selection and which is absent in case of ridge.
- Mathematics behind lasso regression is quiet similar to that of ridge only difference being instead of adding squares of theta, we will add absolute value of Θ.
- It uses L1 regularization technique
- It is generally used when we have more number of features, because it automatically does feature selection.
- It helps in overfitting scenario
- It heps in feature selection
- It removes features where slope is very very low
- Slope will slowly move to 0 and finally will be 0 

These methods are effective to use when there is collinearity in your input values and ordinary least squares would overfit the training data.

## Implementation


Refer to the [Notebook](https://github.com/mittalsharad/ML_Algorithms/blob/main/Linear%20Regression/Linear_Regression.ipynb)


