import math

class nag:
    def __init__(self):
        self.lambda_prev = 0
        self.lambda_curr = 1
        self.gamma = 1

    def iter(self, var_and, var_or, step_size=0.1):
        or_prev = var_or.clone()
        and_prev = var_and.clone()

        or_curr = var_or - step_size * var_or.grad
        and_curr = var_and + step_size * var_and.grad
        var_or = (1 - self.gamma) * or_curr + self.gamma * or_prev
        var_and = (1 - self.gamma) * and_curr + self.gamma * and_prev

        lambda_tmp = self.lambda_curr
        self.lambda_curr = (1 + math.sqrt(1 + 4 * self.lambda_prev * self.lambda_prev)) / 2
        self.lambda_prev = lambda_tmp
        self.gamma = (1 - self.lambda_prev) / self.lambda_curr
        return var_or, var_and


