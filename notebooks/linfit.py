import numpy as np

def LineFitWt(x, y, dy):
    """Evaluates a weighted linear regression and calculates the errors in
    the deviation of a and b

    Parameters:
    x: array (any shape)
    y: array (any shape)
    dy: deviation of y (array, any shape)

    Returns:
    slope: scalar (float)
    yint: scalar (float)
    sigma_b_error: scalar (float)
    sigma_a_error: scalar (float)

    x=np.array([2,9,1,4,5])
    y=np.array([6,4,2,1,2])
    dy=np.array([0.5,0.3,0.6,0.7,0.4])
    print(linfit.LineFitWt(x,y,dy))
    (0.066112348651233666, 3.0027585519020397, 0.063751727558763752, 0.41468364136714186)
    """

    if type(x) != np.ndarray:
        raise ValueError(
            "expected x type of array got {}",
            format(type(x)))

    if type(y) != np.ndarray:
        raise ValueError(
            "expected y type of array got {}",
            format(type(y)))

    if type(dy) != np.ndarray:
        raise ValueError(
            "expected dy type of array got {}",
            format(type(dy)))

    x.flatten()
    y.flatten()
    dy.flatten()

    if len(x) != len(y) or len(x) != len(dy):
        raise ValueError("length of arrays must be equal")

    dy2 = dy**2.
    denom= np.sum(1/(dy**2.))  # Pine 7.14
    xnumerator = np.sum(x/dy2)
    ynumerator = np.sum(y/dy2)
    xhat = xnumerator/denom
    yhat = ynumerator/denom
    bnum = np.sum((x - xhat)*y/dy2)
    bdenom = np.sum((x - xhat)*x/dy2)
    slope = bnum/bdenom
    yint = yhat - slope*xhat
    sigma_b_error = np.sqrt(1/np.sum(((x - xhat)*x)/dy2))
    sigma_a_error = np.sqrt(sigma_b_error**2*((np.sum(x**2/dy2))/denom))
    return slope,yint,sigma_b_error,sigma_a_error
