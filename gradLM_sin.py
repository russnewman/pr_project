import torch
import matplotlib.pyplot as plt
from GradLM import Function, GradLM


class Sin(Function):
    def __init__(self, x, init_params=None):
        super().__init__(x)
        if init_params is not None:
            self.init_params = init_params

    def value(self):
        return self.params[0] * torch.sin(self.params[1] * self.x + self.params[2]) + self.params[3]

    def jacobian(self):
        return torch.stack([torch.sin(self.params[1]*self.x + self.params[2]),
                            self.params[0]*self.x *
                            torch.cos(self.params[1]*self.x + self.params[2]),
                            self.params[0]*torch.cos(self.params[1]
                                                     * self.x + self.params[2]),
                            torch.ones_like(self.x)], dim=1)

    def calc(self, x):
        return self.params[0] * torch.sin(self.params[1] * x + self.params[2]) + self.params[3]


# data prep
x = torch.linspace(0, 1, 30)
true_params = torch.Tensor([5, 1, 0, 10])
y = true_params[0] * torch.sin(true_params[1] * x +
                               true_params[2]) + true_params[3] + torch.rand_like(x)
y.requires_grad = True

# solve with GradLM
s = Sin(x, init_params=torch.ones(4))
solver = GradLM(y=y, func=s, lamda_min=0.1)
f = solver.optimize()

# plot results
x_test = torch.linspace(0, 1, 1000)
y_test = f.calc(x_test)
plt.plot(x, y.detach(), '.', label='Input data with noise')
plt.plot(x_test.detach(), y_test.detach(), label='Fitted curve')
plt.legend()
plt.show()
