import numpy as np
import matplotlib.pyplot as plt
eps = 0.00000001

# here we'll familiarize with some mathematical concepts relevant to machine learning

# We're going to start by familiarizing with the "Cost function"
"""
# The "Cost function" is denoted by J

 J(theta0,thetha1) = 1 / (2*m) * sum( h_theta(x_i) - y_i )**2 [ for i from 1 to m ]

 theta0: y_intercept of model
 theta1: x_coefficient of model
 m: size, or # of data points
 x_i, y_i are observed data points from our data set

    We can re-write J as:
    J(theta0) = 1/(2*m) * sum( theta0 * x_i - y_i)**2  [ for i from 1 to m ]

    how do we take the derivate of this function with respect to theta0?

"""

# The cost function is the difference between the output of our regression model and the 
# true values observed in the sample. Our goal is to minimize the cost function.

# Our regression model is based on our hypothesis function. In linear regression
# the hypothesis function is parameterized by two parameters:
#
#  theta0: y-intercept
#  theta1: x-coefficient
#
# To begin with, let's focus on cases where theta0 is 0 (i.e. data passes through (0,0) )

def generate_x(start, end, size):
    return np.linspace(start,end,size)

def generate_y(x_values, theta0, theta1, size, scaling_factor=0):
    # scaling factor:
    # scales the generated random number 
    # to give us "observation" that we can then try to model

    y = theta1*(x_values+scaling_factor*np.random.rand(1,size)[0]-1)+theta0
    return y

def calculate_cost(y_model, y_observed, size):
    l = [(y_model[i]-y_observed[i])**2 for i in range(0,size)]
    ssum = sum(l)
    cost = 1/(2*size)*ssum
    return cost

def model_cost_function(y_model, y_observed, size):
    x = [i for i in range(0,size)]
    y = [(y_model[i] - y_observed[i])**2 for i in range(0,size)]
    return x,y

def d_theta0(x, theta0, theta1, y_observed, size):
    d = [theta0 + theta1*x[i] - y_observed[i] for i in range(0,size)]
    return sum(d)

def d_theta1(x, theta0, theta1, y_observed, size):
    d = [(theta0 + theta1*x[i] - y_observed[i])*x[i] for i in range(0,size)]
    return sum(d)

def gradient_descent(theta0, theta1, x, y_observed, size,  a):
    reached_theta0 = False
    reached_theta1 = False
    while (not (reached_theta0 and reached_theta1)):
        reached_theta0_last = reached_theta0
        reached_theta1_last = reached_theta1
        theta0_last = theta0
        theta1_last = theta1
        d0 = d_theta0(x, theta0, theta1, y_observed, size)
        d1 = d_theta1(x, theta0, theta1, y_observed, size)
        #print("Iteration starts with: Theta0 = {0}, Theta1 = {1}, DTheta0 = {2}, DTheta1 = {3}".format(theta0, theta1, d0, d1))
        theta0 = theta0_last - a * (1/size) * d0
        theta1 = theta1_last - a * (1/size) * d1
        reached_theta0 = abs(theta0 - theta0_last) <= eps
        reached_theta1 = abs(theta1 - theta1_last) <= eps
        if (reached_theta0 and reached_theta0 != reached_theta0_last):
            print("Found Theta0: {0}".format(theta0))
        if (reached_theta1 and reached_theta1 != reached_theta1_last):
            print("Found Theta1: {0}".format(theta1))
    print("Found parameters. Theta0: {0}, Theta1: {1}".format(theta0, theta1))
    return theta0, theta1


def main():
    print("Graphing the cost function...")

    start=0
    end=20
    size=51
    x = generate_x(start,end,size)

    theta0=0
    theta1=2.5

    y_observed = generate_y(x,theta0,theta1,size,3)
    y_guessed_model = generate_y(x,theta0,theta1,size,0)

    theta0_start = theta0+1
    theta1_start = theta1
    a = 0.01
    ("Will try to find thetas from our observations. Let's start with theta0 = {0}, theta1 = {1}. We shall use a = {2}".format(theta0_start, theta1_start, a))
    found_theta0, found_theta1 = gradient_descent(theta0_start, theta1_start, x, y_observed, size, a)
    # let's graph the two functions and the points to compare
    y_found_model = generate_y(x,found_theta0,found_theta1,size,0)

    plt.scatter(x, y_observed, label="Observed")
    plt.plot(x, y_guessed_model, label="Guess")
    plt.plot(x, y_found_model, label="Gradient Descent")
    plt.legend(loc="upper left", frameon=False, ncol=3)
    plt.show()


    """
    # Graphing the cost function with only a single parameter (giving us a parabola on the X,Y plane)
    # this is useless other than for my own visualization purposes of the values of theta1 vs. cost
    y_cost_function_points = [calculate_cost(generate_y(x,0,i,size,0), y_observed, size) for i in np.linspace(-10,20,201)]
    plt.plot(np.linspace(-10,20,201),y_cost_function_points)
    plt.show()
    """
if __name__=='__main__':
    main()

