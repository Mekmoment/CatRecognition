from propagate import propagate

def optimize(w, b, X, Y, num_iterations, learning_rate, print_cost = False):
    """Optimize by gradient descent"""
    costs = []

    for i in range(num_iterations):
        # Initialize grads, cost
        grads, cost = propagate(w, b, X, Y)

        dw = grads["dw"]
        db = grads["db"]

        # update gradient
        w = w - learning_rate*db
        b = b - learning_rate*dw

        # Record the cost every 100 iteration
        if i % 100 == 0:
            costs.append(cost)

    params = {"w": w,
              "b": b}
    
    grads = {"dw": dw,
             "db": db}
    
    return params, grads, costs

