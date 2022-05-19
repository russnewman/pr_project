from abc import ABC
import torch


class Function(ABC):
    def __init__(self, x):
        self.x = x
        self.params = torch.rand(4)

    def value(self):
        pass

    def jacobian(self):
        pass

    def calc(self):
        pass

    def update_params(self, d):
        self.params = self.params + d


class GradLM:
    def __init__(self, y, func, lamda_min=0.1, lambda_max=1, D=1, sigma=1e-5, tol=1e-8, max_iter=100):
        self.y = y
        self.func = func
        # damping function
        self.lamda_min = lamda_min
        self.lambda_max = lambda_max
        self.D = D
        self.sigma = sigma

        self.tol = tol
        self.max_iter = max_iter

    def _qLambda(self, r0, r1):
        return self.lamda_min + (self.lambda_max - self.lamda_min) / (1 + self.D * torch.exp(-self.sigma * (r1 - r0)))

    def _qX(self, dx, r0, r1):
        return dx / (1 + torch.exp(r0 - r1))

    def step(self, lmda):
        J = self.func.jacobian()
        r = self.y - self.func.value()
        return torch.inverse(J.T@J + lmda * torch.eye(J.shape[-1]).type_as(J))@(J.T@r)

    def optimize(self):
        lmda = self.lambda_max
        r = self.y - self.func.value()
        r0 = r.T@r
        r1 = r0.clone()

        for _ in range(self.max_iter):
            dx = self.step(lmda)
            d = self._qX(dx, r0, r1)
            self.func.update_params(d)

            if dx.norm() < self.tol:
                return self.func

            r0 = r1.clone()
            r = self.y - self.func.value()
            r1 = r.T@r
            lmda = self._qLambda(r0, r1)

        return self.func
