import jax
import jax.numpy as jnp
import numpy as np
import optax


@jax.jit
def scan_ssm_batched(A, B, s0, u):
    """
    Computes the state-space model (SSM) in a batched manner using JAX's `lax.scan`.

    Args:
        A (jax.numpy.ndarray): State transition matrix.
        B (jax.numpy.ndarray): Input matrix.
        s0 (jax.numpy.ndarray): Initial state vector.
        u (jax.numpy.ndarray): Input sequence.

    Returns:
        jax.numpy.ndarray: Final state after processing the input sequence.
    """
    def step(x, u_t):
        x_new = jnp.dot(x, A.T) + jnp.expand_dims(u_t, -1) * B 
        return x_new, x_new
    x_final, _ = jax.lax.scan(step, s0, u)
    return x_final 

def make_HiPPO_nojit(N):
    """
    Constructs the HiPPO transition and projection matrices.

    Args:
        N (int): Size of the HiPPO matrix.

    Returns:
        tuple: A tuple containing:
            - A (jax.numpy.ndarray): Transition matrix.
            - B (jax.numpy.ndarray): Projection matrix.
    """
    P = jnp.sqrt(1 + 2 * jnp.arange(N))
    A = P[:, jnp.newaxis] * P[jnp.newaxis, :]
    A = jnp.tril(A) - jnp.diag(jnp.arange(N))
    B = jnp.sqrt(2 * jnp.arange(N) + 1.0)
    # B = B.reshape(len(B),1)
    return -A, B

def discretize_nojit(A, B, step):
    """
    Discretizes a continuous-time state-space model using bilinear transformation.

    Args:
        A (jax.numpy.ndarray): Continuous-time state transition matrix.
        B (jax.numpy.ndarray): Continuous-time input matrix.
        step (float): Discretization step size.

    Returns:
        tuple: A tuple containing:
            - Ab (jax.numpy.ndarray): Discretized state transition matrix.
            - Bb (jax.numpy.ndarray): Discretized projection matrix.
    """
    I  = jnp.eye(A.shape[0])
    BL = jax.numpy.linalg.inv(I - (step / 2.0) * A)
    Ab = BL @ (I + (step / 2.0) * A)
    Bb = (BL * step) @ B
    return Ab, Bb

def make_discrete_HiPPO_nojit(N,L):
    """
    Constructs a discretized HiPPO matrix.

    Args:
        N (int): Size of the HiPPO matrix.
        L (int): Discretization factor.

    Returns:
        tuple: A tuple containing:
            - Ab (jax.numpy.ndarray): Discretized HiPPO matrix.
            - Bb (jax.numpy.ndarray): Discretized projection matrix.
    """
    A, B = make_HiPPO_nojit(N)
    return discretize_nojit(A, B, step=1.0 / L)

def init_mlp_params(key, sizes):
    """
    Initializes parameters for a multi-layer perceptron (MLP).

    Args:
        key (int): Random seed for parameter initialization.
        sizes (list): List of integers representing the sizes of each layer.

    Returns:
        list: A list of dictionaries containing weights ('W') and biases ('b') for each layer.
    """
    key  = jax.random.PRNGKey(key)
    keys = jax.random.split(key, len(sizes) - 1)
    params = []
    for k, (in_dim, out_dim) in zip(keys, zip(sizes[:-1], sizes[1:])):
        weight_key, bias_key = jax.random.split(k)
        W = jax.random.normal(weight_key, (in_dim, out_dim)) * jnp.sqrt(2 / in_dim)
        b = jnp.zeros((out_dim,))
        params.append({'W': W, 'b': b})
    return params

@jax.jit
def layernorm(batch):
    """
    Applies layer normalization to a batch of data.

    Args:
        batch (jax.numpy.ndarray): Input batch of data.

    Returns:
        jax.numpy.ndarray: Normalized batch.
    """
    mean = jnp.mean(batch, axis=-1, keepdims=True)
    std  = jnp.std(batch, axis=-1, keepdims=True) 
    return (batch - mean) / (std + 1e-5)

@jax.jit
def forward_readout_relu(X, readout):
    """
    Performs a forward pass through a readout MLP with ReLU activation.

    Args:
        X (jax.numpy.ndarray): Input data.
        readout (list): List of layer parameters (weights and biases).

    Returns:
        jax.numpy.ndarray: Output of the readout MLP.
    """
    *hidden, last = readout
    for layer in hidden:
        X = jax.nn.relu(X @ layer['W'] + layer['b'])
    return X @ last['W'] + last['b']

@jax.jit
def forward_readout_relu_norm(X, readout):
    """
    Performs a forward pass through a readout MLP with ReLU activation and layer normalization.

    Args:
        X (jax.numpy.ndarray): Input data.
        readout (list): List of layer parameters (weights and biases).

    Returns:
        jax.numpy.ndarray: Output of the readout MLP.
    """
    *hidden, last = readout
    for layer in hidden:
        X = jax.nn.relu(layernorm(X @ layer['W'] + layer['b']))
    return X @ last['W'] + last['b']

@jax.jit
def forward_readout_tanh(X, readout):
    """
    Performs a forward pass through a readout MLP with Tanh activation.

    Args:
        X (jax.numpy.ndarray): Input data.
        readout (list): List of layer parameters (weights and biases).

    Returns:
        jax.numpy.ndarray: Output of the readout MLP.
    """
    *hidden, last = readout
    for layer in hidden:
        X = jax.nn.tanh(X @ layer['W'] + layer['b'])
    return X @ last['W'] + last['b']

@jax.jit
def forward_readout_tanh_norm(X, readout):
    """
    Performs a forward pass through a readout MLP with Tanh activation and layer normalization.

    Args:
        X (jax.numpy.ndarray): Input data.
        readout (list): List of layer parameters (weights and biases).

    Returns:
        jax.numpy.ndarray: Output of the readout MLP.
    """
    *hidden, last = readout
    for layer in hidden:
        X = jax.nn.tanh(layernorm(X @ layer['W'] + layer['b']))
    return X @ last['W'] + last['b']

@jax.jit
def init_grad(params):
    """
    Initializes gradients for a parameter tree with zeros.

    Args:
        params (dict): Parameter tree.

    Returns:
        dict: Gradient tree initialized with zeros.
    """
    return jax.tree.map(lambda p: jnp.zeros(shape=(p.shape)), params)


@jax.jit
def get_perturbations(theta,epsilon,pert_key):
    """
    Generates random perturbations for parameters.

    Args:
        theta (dict): Parameter tree.
        epsilon (float): Perturbation magnitude.
        pert_key (int): Random seed for perturbation generation.

    Returns:
        dict: Perturbation tree.
    """
    key = jax.random.PRNGKey(pert_key, impl=None)
    return jax.tree.map(
        lambda p: jax.random.choice(key, jnp.array([-1,1])*epsilon, shape=(p.shape)), theta
    )

@jax.jit
def collect_grad(perts,delta_c,eta,G):
    """
    Updates the gradient estimate using perturbations and cost differences.

    Args:
        perts (dict): Perturbation tree.
        delta_c (float): Cost difference.
        eta (float): Learning rate.
        G (dict): Current gradient estimate.

    Returns:
        dict: Updated gradient estimate.
    """
    return jax.tree.map(
        lambda G, p: G + p*delta_c*eta, G, perts
    )

@jax.jit
def apply_perturbations(theta,perturbations):
    """
    Applies perturbations to parameters.

    Args:
        theta (dict): Parameter tree.
        perturbations (dict): Perturbation tree.

    Returns:
        dict: Updated parameter tree with applied perturbations.
    """
    return jax.tree.map(lambda param, pert: param+pert, theta, perturbations)

@jax.jit
def MGD_update(params,G):
    """
    Updates parameters using the gradient estimate.

    Args:
        params (dict): Parameter tree.
        G (dict): Gradient estimate.

    Returns:
        dict: Updated parameter tree.
    """
    return jax.tree.map(
        lambda p, G: p - G, params, G
    )

# @jax.jit
# def estimate_gradient(theta,epsilon,eta,pert_key,grad_est,s0,U,y,full_forward_fn):

#     perturbations = get_perturbations(theta,epsilon,pert_key)
#     theta_pert    = apply_perturbations(theta,perturbations)

#     c_0           = full_forward_fn(theta["A"], theta["B"], theta["C"], s0, U, y)
#     c_pert        = full_forward_fn(theta_pert["A"], theta_pert["B"], theta_pert["C"], s0, U, y)

#     delta_c       = c_pert - c_0

#     grad_est      = collect_grad(perturbations,delta_c,eta,grad_est)

#     return grad_est, c_0, pert_key+1

# @jax.jit
# def test_fn(theta,U,y0,x0):
#     X = scan_ssm_batched(theta["A"], theta["B"], U.T, x0)
#     y = jax.nn.sigmoid(forward_readout_relu_norm(X,theta["readout"]))
#     pred_class = (y[:,0] > 0.5).astype(jnp.int32)
#     true_class = y0.astype(jnp.int32)
#     return jnp.mean(pred_class == true_class)

@jax.jit
def loss_BCE(logits,labels):
    """
    Computes the binary cross-entropy loss.

    Args:
        logits (jax.numpy.ndarray): Predicted logits.
        labels (jax.numpy.ndarray): Ground truth labels.

    Returns:
        float: Binary cross-entropy loss.
    """
    return optax.sigmoid_binary_cross_entropy(logits,labels).mean()

@jax.jit
def loss_CE(logits, labels):
    """
    Computes the categorical cross-entropy loss.

    Args:
        logits (jax.numpy.ndarray): Predicted logits.
        labels (jax.numpy.ndarray): Ground truth labels.

    Returns:
        float: Categorical cross-entropy loss.
    """
    return optax.softmax_cross_entropy(logits=logits, labels=labels).mean()

@jax.jit
def loss_MSE(logits, labels):
    """
    Computes the mean squared error (MSE) loss.

    Args:
        logits (jax.numpy.ndarray): Predicted logits.
        labels (jax.numpy.ndarray): Ground truth labels.

    Returns:
        float: MSE loss.
    """
    return optax.losses.l2_loss(logits,labels).mean()

@jax.jit
def compute_accuracy_binary(logits,labels):
    """
    Computes binary classification accuracy.

    Args:
        logits (jax.numpy.ndarray): Predicted logits.
        labels (jax.numpy.ndarray): Ground truth labels.

    Returns:
        float: Binary classification accuracy.
    """
    pred_class = (logits[:,0] > 0.5).astype(jnp.int32)
    true_class = labels.astype(jnp.int32)
    return jnp.mean(pred_class == true_class)

@jax.jit
def compute_accuracy_multi(logits,labels):
    """
    Computes multi-class classification accuracy.

    Args:
        logits (jax.numpy.ndarray): Predicted logits.
        labels (jax.numpy.ndarray): Ground truth labels.

    Returns:
        float: Multi-class classification accuracy.
    """
    pred_class = jnp.argmax(logits, axis=-1)
    true_class = jnp.argmax(labels, axis=-1)
    return jnp.mean(pred_class == true_class)

@jax.jit
def decay_rate(lr0,decay,epoch):
    """
    Computes the learning rate decay.

    Args:
        lr0 (float): Initial learning rate.
        decay (float): Decay factor.
        epoch (int): Current epoch.

    Returns:
        float: Decayed learning rate.
    """
    return (lr0/(1+decay*epoch))    

def assign_functions(
        loss_func  = 'CE',
        activation = 'relu',
        layernorm  = 'layernorm',
        out_shape  = (1,),
        ):
    """
    Assigns functions for loss, activation, and readout based on configuration.

    Args:
        loss_func (str): Loss function type ('CE', 'BCE', or 'MSE').
        activation (str): Activation function type ('relu' or 'tanh').
        layernorm (str): Whether to use layer normalization ('layernorm' or None).
        out_shape (tuple): Output shape.

    Returns:
        tuple: Assigned functions for state, readout, loss, logits, cost, testing, and gradient.
    """
    if loss_func=='CE':
        if out_shape[0]==1:
            loss_fn = loss_BCE
        else:
            loss_fn = loss_CE
    else:
        loss_fn = loss_MSE
    
    if activation=='relu':
        if layernorm==True:
            readout_fn = forward_readout_relu_norm
        else:
            readout_fn = forward_readout_relu

    elif activation=='tanh':
        if layernorm==True:
            readout_fn = forward_readout_tanh_norm
        else:
            readout_fn = forward_readout_tanh

    state_fn = scan_ssm_batched

    @jax.jit
    def logit_fn(A,B,C,s0,U):
        X      = state_fn(A, B, s0, U.T)
        logits = readout_fn(X,C)
        return logits

    @jax.jit
    def input_to_cost_fn(A,B,C,s0,U,y):
        logits = logit_fn(A,B,C,s0,U)
        loss   = loss_fn(logits,y)
        return loss
    
    if out_shape[0]==1:
        accuracy_fn = compute_accuracy_binary
    else:
        accuracy_fn = compute_accuracy_multi
    
    @jax.jit
    def test_fn(theta,s0,U,labels):
        logits   = logit_fn(theta["A"],theta["B"],theta["C"],s0,U)
        accuracy = accuracy_fn(logits,labels)
        return accuracy
    
    @jax.jit
    def grad_estimate_fn(theta,epsilon,eta,pert_key,grad_est,s0,U,y):

        perturbations = get_perturbations(theta,epsilon,pert_key)
        theta_pert    = apply_perturbations(theta,perturbations)

        c_0      = input_to_cost_fn(theta["A"], theta["B"], theta["C"], s0, U, y)
        c_pert   = input_to_cost_fn(theta_pert["A"], theta_pert["B"], theta_pert["C"], s0, U, y)
        delta_c  = c_pert - c_0
        
        grad_est = collect_grad(perturbations,delta_c,eta,grad_est)

        return grad_est, c_0, pert_key+1

    return state_fn, readout_fn, loss_fn, logit_fn, input_to_cost_fn, test_fn, grad_estimate_fn