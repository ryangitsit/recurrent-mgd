import jax.numpy as jnp
import numpy as np

import data_loader as dat
from model import ModelConfig, StateSpaceModel
import src.utils.helper_functions as hf
import src.utils.rnn_functions as rf
import visualize as viz


def main():
    np.random.seed(10)

    data     = dat.get_data("ECG")
    U_train  = data["U_train"]
    y0_train = data["y0_train"]
    U_test   = data["U_test"]
    y0_test  = data["y0_test"]

    model_config = ModelConfig(
        state_size=32,
        tau_theta=64,
        decay = 0.01,
        )

    model = StateSpaceModel(
        config    = model_config,
        in_shape  = (1,140),
        out_shape = (1,),
        )

    # model.print_functions()
    # hf.print_dict(model)

    model.accs=[]
    model.costs=[]
    model.s0      = jnp.zeros((model.tau_theta,model.state_size))
    model.s0_test = jnp.zeros((y0_test.shape[0],model.state_size))
    batch_size = model.tau_theta
    epochs = 250

    hf.print_header()

    # iterate over training epochs
    for epoch in range(epochs):
        epoch_costs = []
        eta = rf.decay_rate(model.eta0,model.decay,epoch)

        #iterate over batchs
        for i in range(0, len(U_train)-batch_size, batch_size):
            
            # specify batched data
            U_batch  = U_train[i:i+batch_size]
            y0_batch = y0_train[i:i+batch_size]

            # forwared pass, perturbed forward pass, gradient estimation
            model.gradient, batch_cost, model.pert_key = model.grad_estimate_fn(
                model.theta,
                model.epsilon,
                eta,
                model.pert_key,
                model.gradient,
                model.s0,
                U_batch,
                y0_batch[:,np.newaxis],
                )
            
            # update parameters
            model.theta = rf.MGD_update(model.theta,model.gradient)

            # reset gradient estimations
            model.gradient = rf.init_grad(model.theta)

            # record batched training cost
            epoch_costs.append(batch_cost)

        # recored avg training cost
        model.costs.append(np.mean(epoch_costs))

        # test model accuracy on unseen data
        acc = model.test_fn(model.theta,model.s0_test,U_test,y0_test)
        model.accs.append(acc)
        hf.print_progress(epoch,model,'--',False,)

    print("\n")

    viz.performance_plot(model)

if __name__=='__main__':
    main()