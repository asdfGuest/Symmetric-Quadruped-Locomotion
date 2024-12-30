
import omni.isaac.lab.utils.math as math_utils

import torch as th
from typing import List, Tuple, Literal


@th.jit.script
def sq_norm(x:th.Tensor):
    return (x**2).sum(dim=-1)

@th.jit.script
def unit(x:th.Tensor):
    return x / (th.norm(x,dim=-1,keepdim=True) + 1e-8)

@th.jit.script
def vec_ang(x:th.Tensor, y:th.Tensor):
    cos = (x*y).sum(dim=-1) / (th.norm(x,dim=-1) * th.norm(y,dim=-1) + 1e-8)
    return cos.clip(-1.+1e-8,1.-1e-8).acos()

@th.jit.script
def dot(x:th.Tensor, y:th.Tensor, keepdim:bool=False):
    return (x*y).sum(dim=-1, keepdim=keepdim)

@th.jit.script
def get_relative_values(
        pos_a:th.Tensor,
        quat_a:th.Tensor,
        linvel_a:th.Tensor,
        angvel_a:th.Tensor,
        pos_b:th.Tensor,
        quat_b:th.Tensor,
        linvel_b:th.Tensor,
        angvel_b:th.Tensor
    ) ->Tuple[th.Tensor,th.Tensor,th.Tensor,th.Tensor]:
    '''
    compute relative quantities of b respect to a

    Args:
        pos_a:      (n_batch, n_body, 3)
        quat_a:     (n_batch, n_body, 4)
        linvel_a:   (n_batch, n_body, 3)
        angvel_a:   (n_batch, n_body, 3)
        pos_b:      (n_batch, n_body, 3)
        quat_b:     (n_batch, n_body, 4)
        linvel_b:   (n_batch, n_body, 3)
        angvel_b:   (n_batch, n_body, 3)

    Returns:
        - relative position
        - relative quaternion
        - relative linear velocity
        - relative angular velocity
    '''
    pos_a,pos_b = th.broadcast_tensors(pos_a,pos_b)
    quat_a,quat_b = th.broadcast_tensors(quat_a,quat_b)
    linvel_a,linvel_b = th.broadcast_tensors(linvel_a,linvel_b)
    angvel_a,angvel_b = th.broadcast_tensors(angvel_a,angvel_b)

    quat_a_inv = math_utils.quat_inv(quat_a)

    rel_pos = math_utils.quat_rotate(quat_a_inv, pos_b-pos_a)
    rel_quat = math_utils.quat_mul(quat_a_inv, quat_b)

    rel_linvel = math_utils.quat_rotate(quat_a_inv, linvel_b-linvel_a) - th.cross(angvel_a, rel_pos, dim=-1)
    rel_angvel = math_utils.quat_rotate(quat_a_inv, angvel_b-angvel_a)

    return rel_pos, rel_quat, rel_linvel, rel_angvel


cuda = th.device('cuda')


@th.jit.script
def th_zeros_f(size:List[int], device:th.device=cuda) :
    return th.zeros(size=size, dtype=th.float32, device=device)

@th.jit.script
def th_zeros_i(size:List[int], device:th.device=cuda) :
    return th.zeros(size=size, dtype=th.int32, device=device)

@th.jit.script
def th_zeros_b(size:List[int], device:th.device=cuda) :
    return th.zeros(size=size, dtype=th.bool, device=device)

@th.jit.script
def th_rand_f(size:List[int], device:th.device=cuda) :
    return th.rand(size=size, dtype=th.float32, device=device)


worldvec_f = th.tensor([1.0, 0.0, 0.0], dtype=th.float32, device=cuda)
worldvec_r = th.tensor([0.0, 1.0, 0.0], dtype=th.float32, device=cuda)
worldvec_u = th.tensor([0.0, 0.0, 1.0], dtype=th.float32, device=cuda)


@th.jit.script
def apply_tensor_list(x:List[th.Tensor], oper:str) :
    a = th.zeros_like(x[0])
    for b in x :
        if oper == 'add' :
            a += b
        elif oper == 'or' :
            a |= b
        elif oper == 'and' :
            a &= b
    return a


@th.jit.script
def sum_tensor_list(x:List[th.Tensor]) :
    a = th.zeros_like(x[0])
    for b in x :
        a += b
    return a
