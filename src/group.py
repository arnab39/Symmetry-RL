import torch
from functools import reduce


class SOnRep:
    def __init__(self, batch_size=128, n=3):
        self.dim = n
        self.num_params = n * (n - 1) // 2
        self.batch_size = batch_size
        self.thetas = None
        self.__matrix = None

    def set_group_params(self, thetas):
        self.thetas = thetas
        self.clear_matrix()
        self.device = thetas.get_device()
        if self.device < 0:
            self.device = torch.device('cpu')

    def set_matrix(self, mat):
        self.__matrix = mat
        self.clear_thetas()

    def get_group_params(self):
        return self.get_thetas()

    def get_thetas(self):
        if self.thetas is None:
            if self.dim == 3:
                theta_23 = torch.atan2(
                    -self.__matrix[:, 2, 1], self.__matrix[:, 2, 2]
                ).unsqueeze(1)
                theta_13 = torch.atan2(
                    -self.__matrix[:, 2, 0], torch.sqrt(
                        self.__matrix[:, 2, 1]**2 + self.__matrix[:, 2, 2]**2
                    )
                ).unsqueeze(1)
                theta_12 = torch.atan2(
                    -self.__matrix[:, 1, 0], self.__matrix[:, 0, 0]
                ).unsqueeze(1)
                thetas = torch.cat(
                    (theta_12, theta_13, theta_23), dim=1
                ).to(self.device)
                self.set_group_params(thetas)
            elif self.dim == 2:
                theta = torch.acos(
                    self.__matrix[:, 0, 0]
                ).unsqueeze(1).to(self.device)
                self.set_group_params(theta)
            else:
                raise Exception('Not implemented')
        return self.thetas

    def clear_matrix(self):
        self.__matrix = None

    def clear_thetas(self):
        self.thetas = None

    def get_matrix(self):
        if self.__matrix is None:
            n_th = 0
            rotation_mats = []
            for i in range(self.dim - 1):
                for j in range(self.dim - 1 - i):
                    theta_ij = self.thetas[:, n_th]
                    n_th += 1
                    cos, sin = torch.cos(theta_ij), torch.sin(theta_ij)
                    rot_mat = torch.eye(self.dim, device=self.device)
                    rot_mat = rot_mat.unsqueeze(0).repeat(self.batch_size, 1, 1)
                    rot_mat[:, i, i] = cos
                    rot_mat[:, i, i + j + 1] = sin
                    rot_mat[:, j + i + 1, i] = -sin
                    rot_mat[:, j + i + 1, j + i + 1] = cos
                    rotation_mats.append(rot_mat)
            self.__matrix = reduce(lambda x, y: x @ y, rotation_mats)
        return self.__matrix


class EnRep:
    def __init__(self, batch_size=128, n=3):
        self.batch_size = batch_size
        self.dim = n
        self.num_son_params = n * (n - 1) // 2
        self.num_euclidean_params = n + self.num_son_params
        self.thetas = None
        self.translation_v = None
        self.__matrix = None

    def set_group_params(self, params):
        self.thetas = params[:, :self.num_son_params]
        self.translation_v = params[:, self.num_son_params:]
        self.clear_matrix()
        self.device = params.get_device()
        if self.device < 0:
            self.device = torch.device('cpu')

    def set_matrix(self, mat):
        self.__matrix = mat
        self.clear_group_parameters()

    def get_group_params(self):
        return torch.cat((self.get_thetas(), self.get_translation_vector()), dim=-1)

    def get_thetas(self):
        if self.thetas is None:
            rotation_matrix_rep = SOnRep(self.batch_size, self.dim)
            rotation_matrix_rep.set_matrix(self.__matrix[:, :self.dim, :self.dim])
            self.thetas = rotation_matrix_rep.get_thetas()
        return self.thetas()

    def get_translation_vector(self):
        if self.translation_v is None:
            self.translation_v = self.__matrix[:, :self.dim, self.dim]
        return self.translation_v

    def clear_matrix(self):
        self.__matrix = None

    def clear_group_parameters(self):
        self.thetas = None
        self.translation_v = None

    def get_matrix(self):
        if self.__matrix is None:
            rotation_matrix__rep = SOnRep(self.batch_size, self.dim)
            rotation_matrix__rep.set_group_params(self.thetas)
            self.__matrix = torch.zeros(
                (self.batch_size, self.dim + 1, self.dim + 1)
                , device=self.device
            )
            self.__matrix[:, :self.dim, :self.dim] = rotation_matrix__rep.get_matrix()
            self.__matrix[:, :self.dim, self.dim] = self.translation_v
            self.__matrix[:, self.dim, self.dim] = 1
        return self.__matrix


class Group:
    def __init__(self, group_type, group_dim, num_blocks):
        """
        :param group_type: Type of group SOn or En
        :param group_dim: dimension of the group
        :param num_blocks: number of independent group blocks
        false for representations
        """
        self.group_type = group_type
        self.group_dim = group_dim
        if group_type == 'SOn':
            self.group_param_dim = group_dim * (group_dim - 1) // 2
            self.rep_dim = group_dim
        elif group_type == 'En':
            self.group_param_dim = group_dim + group_dim * (group_dim - 1) // 2
            self.rep_dim = group_dim + 1
        elif group_type in ('Real', 'none'):
            self.group_param_dim = group_dim
            self.rep_dim = group_dim
        elif group_type == 'GLn':
            self.group_param_dim = group_dim * group_dim
            self.rep_dim = group_dim
        self.num_blocks = num_blocks

    def compose(self, grouprep_1, grouprep_2):
        """
        :param grouprep_1: batch_size * num_blocks * rep_dim * rep_dim
        :param grouprep_2: batch_size * num_blocks * rep_dim * rep_dim
        :return:
            composed representation
        """
        if self.group_type in ('SOn', 'En', 'GLn'):
            #grouprep_out = torch.einsum('bnij,bnjk->bnik', grouprep_1, grouprep_2)
            grouprep_out = torch.matmul(grouprep_1, grouprep_2)
        elif self.group_type == 'Real':
            grouprep_out = grouprep_1 + grouprep_2
        return grouprep_out

    def inverse(self, grouprep):
        """
        :param grouprep: batch_size * num_blocks * rep_dim * rep_dim
        :return:
            inverse representation
        """
        if self.group_type == 'Real':
            return -1 * grouprep
        grouprep_out = []
        for i in range(self.num_blocks):
            grouprep_out.append(
                torch.linalg.inv(grouprep[:, i, :, :]).unsqueeze(1)
            )
        return torch.cat(grouprep_out, dim=1)



class EulerParameterization(Group):
    def __init__(self, group_type, group_dim, num_blocks):
        super().__init__(group_type, group_dim, num_blocks)

    def get_group_rep(self, params):
        """
        :param params: batch_size * num_blocks * param_dim
        :return:
            representation: batch_size * num_blocks * rep_dim * rep_dim
        """
        batch_size = params.shape[0]
        if self.group_type == 'SOn':
            group_type = SOnRep(
                batch_size, n=self.group_dim
            )
        elif self.group_type == 'En':
            group_type = EnRep(
                batch_size, n=self.group_dim
            )
        elif self.group_type == 'GLn':
            return params.reshape(
                params.shape[0], self.num_blocks, self.rep_dim, self.rep_dim
            )
        elif self.group_type in ('Real', 'none'):
            return params
        rep_out = []
        for j in range(self.num_blocks):
            group_type.set_group_params(params[:, j, :])
            rep_out.append(
                group_type.get_matrix().unsqueeze(1)
            )
        return torch.cat(rep_out, dim=1)



class LieParameterization(Group):
    def __init__(self, group_type, group_dim, num_blocks):
        super().__init__(group_type, group_dim, num_blocks)
    """
    :param params: batch_size * num_blocks * param_dim
    :return:
        representation: batch_size * num_codes * rep_dim * rep_dim
    """

    def get_son_bases(self):
        """
        :return:
            son basis: basis of the Lie group of SOn:  num_poarams * group_dim * group_dim
        """
        num_son_bases = self.group_dim * (self.group_dim - 1) // 2
        son_bases = torch.zeros((num_son_bases, self.group_dim, self.group_dim))
        counter = 0
        for i in range(0, self.group_dim):
            for j in range(i + 1, self.group_dim):
                son_bases[counter, i, j] = 1
                son_bases[counter, j, i] = -1
                counter += 1
        return son_bases

    def get_son_rep(self, params):
        son_bases = self.get_son_bases()
        son_bases = son_bases.to(self.device)
        A = torch.einsum('nbs,sij->nbij', params, son_bases)
        rho = torch.matrix_exp(A)
        return rho

    def get_group_rep(self, params):
        """
        :param params: batch_size * num_blocks * param_dim
        :return:
            representation: batch_size * num_codes * rep_dim * rep_dim
        """
        self.device = params.get_device()
        if self.group_type == 'SOn':
            rho = self.get_son_rep(params)
        elif self.group_type == 'En':
            son_param_dim = self.group_param_dim - self.group_dim
            thetas = params[:, :, :son_param_dim]
            translations = params[:, :, son_param_dim:]
            rho_thetas = self.get_son_rep(thetas)
            rho = torch.zeros(
                (params.shape[0], self.num_blocks, self.rep_dim, self.rep_dim)
                , device=self.device
            )
            rho[:, :, :self.group_dim, :self.group_dim] = rho_thetas
            rho[:, :, :self.group_dim, self.group_dim] = translations
            rho[:, :, self.group_dim, self.group_dim] = 1
        elif self.group_type == 'GLn':
            rho = params.reshape(
                params.shape[0], self.num_blocks, self.rep_dim, self.rep_dim
            )
        elif self.group_type in ('Real', 'none'):
            rho = params

        return rho
