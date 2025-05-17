import copy

import numpy as np
from dataclasses import dataclass, field
import src.utils.rnn_functions as rf

@dataclass
class ModelConfig:
    name       : str   = "model_default"
    state_size : int   = 32
    C_dims     : str   = "default"
    trainable  : str   = "ABC"
    eta0       : str   = "norm"
    eta_coeff  : str   = "default"
    epsilon    : float = 1e-5
    tau_theta  : int   = 64
    decay      : float = 0.001
    loss_func  : str   = 'CE'
    activation : str   = "relu"
    layernorm  : bool  = True
    tracing    : bool  = False

    def __post_init__(self):
        if self.activation not in {"relu", "tanh"}:
            raise ValueError("Invalid activation")

    @classmethod
    def from_kwargs(cls, **kwargs):
        valid_keys = set(f.name for f in cls.__dataclass_fields__.values())
        extra = set(kwargs) - valid_keys
        if extra:
            raise TypeError(f"Unexpected keyword arguments: {extra}")
        return cls(**kwargs)

class StateSpaceModel:
    def __init__(
            self, 
            config: ModelConfig, 
            in_shape: tuple, 
            out_shape: tuple, 
            init_key: int = 0,
            pert_key: int = 0,
            ):
        self.__dict__.update(config.__dict__)
        self.in_shape  = in_shape
        self.out_shape = out_shape
        self.init_key  = init_key
        self.pert_key  = pert_key

        self.init_parameters()
        self.assign_functions()
        self.assign_eta()

    def init_parameters(self):
        self.L = self.in_shape[1]
        A, B = rf.make_discrete_HiPPO_nojit(self.state_size,self.L)
        if self.C_dims == "default":
            self.C_dims = [self.state_size,self.state_size,self.out_shape[0]]
        C = rf.init_mlp_params(self.init_key,self.C_dims)
        self.theta = {"A":A, "B":B, "C":C}
        self.theta0 = copy.deepcopy(self.theta)
        self.gradient = rf.init_grad(self.theta)

    def assign_functions(self):
        (
            self.state_fn, 
            self.readout_fn, 
            self.loss_fn, 
            self.logit_fn, 
            self.input_to_cost_fn,
            self.test_fn,
            self.grad_estimate_fn
            ) = rf.assign_functions(self.loss_func,self.activation,self.layernorm,self.out_shape)

    def print_functions(self):
        print(f"State function   : {self.state_fn.__name__}")
        print(f"Readout function : {self.readout_fn.__name__}")
        print(f"Loss function    : {self.loss_fn.__name__}")
        print(f"Logit function   : {self.logit_fn.__name__}")
        print(f"Full function    : {self.input_to_cost_fn.__name__}")
        print(f"Test function    : {self.test_fn.__name__}")

    def assign_eta(self):
        if self.eta0=="norm":
            self.normalize_eta()

    def eta_norm(self,eps,K,tau,classes,c): 
        return 1 / (eps**2 
                    * c
                    * np.sqrt(K) 
                    * np.sqrt(1+(K-1)/tau)
                    )

    def normalize_eta(self):
        self.K = 0
        if "C" in self.trainable:
            self.K += np.prod(self.C_dims)
            self.K += np.sum(self.C_dims[1:])

        if "A" in self.trainable: 
            self.K += self.L * (self.state_size)**2 

        if "B" in self.trainable:
            self.K += self.L * (self.state_size)

        if self.eta_coeff=="default":
            c = np.max([5,self.C_dims[-1]])
        else:
            c = self.eta_coeff

        self.eta0 = self.eta_norm(
            self.epsilon,self.K,self.tau_theta,self.C_dims[-1],c
            )



