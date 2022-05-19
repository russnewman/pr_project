import torch
from GradLM import Function


class PoseEstimation(Function):
    EPS = 1e-10

    def __init__(self, x, y, init_params=None):
        '''
        x - point cloud (torch.Tensor [N, 3])
        y - point cloud (torch.Tensor [N, 3])
        init_params (torch.Tensor [6])
        '''
        super().__init__(x)
        self.y = y
        if init_params is None:
            self.params = torch.rand().to(x)
        else:
            self.params = init_params.to(x)

    def value(self):
        return self.calc(self.x)

    def _rotation_matrix(self, angle):
        N = angle.size(0)
        angle2 = (angle * angle).sum(1).view(N, 1, 1)
        angle_norm = torch.sqrt(angle2)

        vector = angle.contiguous().view(N, 3)
        K = vector.new().resize_(N, 3, 3).fill_(0)
        K[:, 0, 1] = -vector[:, 2]
        K[:, 1, 0] = vector[:, 2]
        K[:, 0, 2] = vector[:, 1]
        K[:, 2, 0] = -vector[:, 1]
        K[:, 1, 2] = -vector[:, 0]
        K[:, 2, 1] = vector[:, 0]
        K = K.type_as(angle)
        K2 = torch.bmm(K, K)

        s = torch.sin(angle_norm) / angle_norm
        s.masked_fill_(angle2.lt(self.EPS), 1)
        c = (1 - torch.cos(angle_norm)) / angle2
        c.masked_fill_(angle2.lt(self.EPS), 0)

        rotation_matrix = torch.eye(3).view(
            1, 3, 3).repeat(N, 1, 1).type_as(angle)
        rotation_matrix += K * s.expand(N, 3, 3)
        rotation_matrix += K2 * c.expand(N, 3, 3)
        return rotation_matrix

    def jacobian(self):
        transl, angle = self.params.split([3, 3], dim=-1)
        rotation_matrix = self._rotation_matrix(angle.unsqueeze(0)).squeeze(0)
        x_transform = torch.bmm(self.x.unsqueeze(1), rotation_matrix.T.unsqueeze(
            0).repeat(self.x.size(0), 1, 1)) + transl
        x_transform.squeeze_(1)
        N = x_transform.size(0)
        vec = x_transform.contiguous().view(N, 3)
        m = vec.new().resize_(N, 3, 3).fill_(0)
        m[:, 0, 1] = -vec[:, 2]
        m[:, 1, 0] = vec[:, 2]
        m[:, 0, 2] = vec[:, 1]
        m[:, 2, 0] = -vec[:, 1]
        m[:, 1, 2] = -vec[:, 0]
        m[:, 2, 1] = vec[:, 0]
        m = m.type_as(x_transform)
        J = torch.cat((torch.eye(m.size(1)).type_as(
            self.x).unsqueeze(0).repeat(m.size(0), 1, 1), -m), dim=-1)
        return J.flatten(start_dim=0, end_dim=1)

    def calc(self, x):
        transl, angle = self.params.split([3, 3], dim=-1)
        rotation_matrix = self._rotation_matrix(angle.unsqueeze(0)).squeeze(0)
        x_transform = torch.bmm(x.unsqueeze(1), rotation_matrix.T.unsqueeze(
            0).repeat(x.size(0), 1, 1)) + transl
        x_transform.squeeze_(1)
        return x_transform.flatten()
