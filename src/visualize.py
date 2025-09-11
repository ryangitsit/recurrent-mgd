import matplotlib.pyplot as plt
import jax
import jax.numpy as jnp
import numpy as np

def performance_plot(model):
    plt.style.use('seaborn-v0_8-muted')
    plt.figure(figsize=(8,3))
    plt.title(f"Performance: Max Accuracy = {np.max(model.accs):.3}")
    plt.xlabel("Epochs")
    plt.ylabel("Accuracy")
    plt.plot(model.accs,linewidth=4)
    plt.plot(model.costs/np.max(model.costs),linewidth=4)
    plt.tight_layout()
    plt.show()