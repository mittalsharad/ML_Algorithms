# Logistic Rgression

## Introduction

> “Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function” (Wikipedia)


Logistic Regression is a technique borrowed from Statistic field by machine learnng.
It's an extension of the linear regression model for classification problems.
The building block concepts of logistic regression can be helpful in deep learning while building the neural networks.
It is a class of Regression where the Independent variables are used to predict the dependent variables.

Based on Dependent variable, it can be classified as:
- **Binomial**: if the dependent variable has only 2 possible outcomes
- **Multinomial**: when the dependent variable has more than 2 possible outcomes.
- **Ordinal**: when the dependent variable category has to be ranked.

Let’s consider a small example, here is a plot on the x-axis Age of the persons, and the y-axis shows they have a smartphone. It is a classification problem where given the age of a person and we have to predict if he posses a smartphone or not.

![image](https://user-images.githubusercontent.com/22586467/122337818-7a41b700-cf5c-11eb-864a-58370b742c0e.png)

In such a classification problem,  can we use linear regression?

## Problem with Linear Regression:

To solve the above prediction problem, let’s first use a Linear model. On the plot, we can draw a line that separates the data points into two groups. with a threshold Age value. All the data points below that threshold will be classified as 0 i.e those who do not have smartphones. Similarly, all the observations above the threshold will be classified as 1 which means these people have smartphones as shown in the image below.

![image](https://user-images.githubusercontent.com/22586467/122337969-a78e6500-cf5c-11eb-816a-bbe3bd6092e0.png)

Don’t you think it is successfully working? let me discuss some scenarios.

**Case 1**

Suppose we got a new data point on the extreme right in the plot, suddenly you see the slope of the line changes. Now we have to inadvertently change the threshold of our model. Hence, this is the first issue we have with linear regression, our threshold of Age can not be changed in a predicting algorithm.

![image](https://user-images.githubusercontent.com/22586467/122338046-c42a9d00-cf5c-11eb-9b31-67b04e797fff.png)

**Case 2**
The other issue with Linear regression is when you extend this line it will give you values above 1 and below 0. In our classification problem, we do not know what the values greater than one and below 0 represents. so it is not the natural extension of the linear model. Further, it makes the model interpretation at extremes a challenge.

![image](https://user-images.githubusercontent.com/22586467/122338108-de647b00-cf5c-11eb-91cd-43e9eb8fb61d.png)


So,

1. By its nature, linear regression only looks at linear relationships between dependent and independent variables. That is, it assumes there is a straight-line relationship between them.

2. Linear regression looks at a relationship between the mean of the dependent variable and the independent variables. 

3. Linear Regression Is Sensitive to Outliers.

4. Linear regression assumes that the data are independent. That means that the scores of one subject (such as a person) have nothing to do with those of another. This is often, but not always, sensible. 

## From Linear to logistic regression

Here the Logistic regression comes in. let’s try and build a new model known as Logistic regression. Suppose the equation of this linear line is ![image](https://user-images.githubusercontent.com/22586467/122338720-b32e5b80-cf5d-11eb-9116-bb0b73a47283.png)

Now we want a function Q( Z) that transforms the values between 0 and 1 as shown in the following image. This is the time when a `sigmoid function or logit function` comes in handy.
![image](https://user-images.githubusercontent.com/22586467/122338771-c6d9c200-cf5d-11eb-9a2b-397c2ec8ff69.png)

This sigmoid function transforms the linear line into a curve. This will constraint the values between 0 and 1. Now it doesn’t matter how many new points I add to each extreme it will not affect my model.

The other important aspect is, for each observation model will give a continuous value between 0 and 1. This continuous value is the prediction probability of that data point. If the prediction probability is near 1 then the data point will be classified as 1 else 0.

### What is Softmax Function (Sigmoid function):

Softmax function is the popular function to calculate the probabilities of the events. The other mathematical advantages of using the softmax function are the output range.  Softmax function output values are always in the range of (0, 1). The sum of the output values will always equal to the 1. The Softmax is also known as the normalized exponential function.

![image](https://user-images.githubusercontent.com/22586467/122339199-6434f600-cf5e-11eb-8c05-0ccee58de3dd.png)

The above is the softmax formula. Which takes each value (Logits) and find the probability. The numerator the e-power values of the Logit and the denominator calculates the sum of the e-power values of all the Logits.

Softmax function used in:
- Naive Bayes Classifier
- Multinomial Logistic Classifier
- Deep Learning (While building Neural networks)


**The special cases of softmax function input**
The two special cases we need to consider about the Softmax function output, If we do the below modifications to the Softmax function inputs.

1.Multiplying the Softmax function inputs (Multiplying the Logits with any value)
2. Dividing the Softmax function inputs (Dividing the Logits with any value)

**Multiplying the Softmax function inputs:**
If we multiply the Softmax function inputs, the inputs values will become large. So the logistic regression will be more confident (High Probability value) about the predicted target class.

**Dividing the Softmax function inputs:**
If we divide the Softmax function inputs, the inputs values will become small. So the Logistic regression model will be not confident (Less Probability value) of the predicted target class.

## The cost function for Logistic regression

For linear regression, the cost function is mostly we use Mean squared error represented as the difference y_predicted and y_actual iterated overall data points, and then you do a square and take the average. It is a convex function as shown below. This cost function can be optimized easily using gradient descent.

![image](https://user-images.githubusercontent.com/22586467/122339382-a8c09180-cf5e-11eb-98e5-4423792b55de.png)

Whereas, If we use the same cost function for the Logistic regression is a non-linear function, it will have a non-convex plot. It will create unnecessary complications if use gradient descent for model optimization.

![image](https://user-images.githubusercontent.com/22586467/122339433-b83fda80-cf5e-11eb-805d-2f9810f1165b.png)

Hence, we need a different cost function for our new model. Here comes the log loss in the picture.  As you can see, we have replaced the probability in the log loss equation with y_hat

![image](https://user-images.githubusercontent.com/22586467/122339473-c3930600-cf5e-11eb-8032-1988a1247ad1.png)

In the first case when the class is 1 and the probability is close to 1, the left side of the equation becomes active and the right part vanishes. You will notice in the plot below as the predicted probability moves towards 0 the cost increases sharply.

![image](https://user-images.githubusercontent.com/22586467/122339808-24bad980-cf5f-11eb-9da6-8a0ee83d302a.png)

Similarly, when the actual class is 0 and the predicted probability is 0, the right side becomes active and the left side vanishes. Increasing the cost of the wrong predictions. Later, these two parts will be added.

![image](https://user-images.githubusercontent.com/22586467/122339935-47e58900-cf5f-11eb-923f-dc6589ca2d55.png)

## Optimize the model
Once we have our model and the appropriate cost function handy, we can use “The Gradient Descent Algorithm” to optimize our model parameters. As we do in the case of linear regression.

##  Advantages and Disadvantages
Many of the pros and cons of the linear regression model also apply to the logistic regression model. Logistic regression has been widely used by many different people, but it struggles with its restrictive expressiveness (e.g. interactions must be added manually) and other models may have better predictive performance.

Another disadvantage of the logistic regression model is that the interpretation is more difficult because the interpretation of the weights is multiplicative and not additive.

Logistic regression can suffer from complete separation. If there is a feature that would perfectly separate the two classes, the logistic regression model can no longer be trained. This is because the weight for that feature would not converge, because the optimal weight would be infinite. This is really a bit unfortunate, because such a feature is really useful. But you do not need machine learning if you have a simple rule that separates both classes. The problem of complete separation can be solved by introducing penalization of the weights or defining a prior probability distribution of weights.

On the good side, the logistic regression model is not only a classification model, but also gives you probabilities. This is a big advantage over models that can only provide the final classification. Knowing that an instance has a 99% probability for a class compared to 51% makes a big difference.
