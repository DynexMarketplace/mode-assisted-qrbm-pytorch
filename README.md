# Mode-assisted unsupervised learning of restricted Boltzmann machines
Restricted Boltzmann machines (RBMs) are a powerful class of generative models, but their training requires computing a gradient that, unlike supervised backpropagation on typical loss functions, is notoriously difficult even to approximate. Here, we show that properly combining standard gradient updates with an off-gradient direction, constructed from samples of the RBM ground state (mode), improves training dramatically over traditional gradient methods. This approach, which we call ‘mode-assisted training’, promotes faster training and stability, in addition to lower converged relative entropy (KL divergence). We demonstrate its efficacy on synthetic datasets where we can compute KL divergences exactly, as well as on a larger machine learning standard (MNIST). The proposed mode-assisted training can be applied in conjunction with any given gradient method, and is easily extended to more general energy- based neural network structures such as deep, convolutional and unrestricted Boltzmann machines.

The following chart displayes the superiority of mode-assisted training vs. CD-1:

![OUTOUT](https://github.com/dynexcoin/mode_assisted_unsupervised_learning/blob/main/output.png)

# Mode-Assisted QRBM as Python Class (PyTorch based)

Can be universally used. Here's an example using the MNIST data-set, comparison of CD-1 vs. Mode-Assisted. While traditional model training (CD) fails to improve at one point (blue line), is the new PyTorch mode-assisted Quantum Boltzmann Machine (QRBM) breaking through that boundary and proving superior capability in pattern reproduction and detailing.

To run the code:

```
python3 main.py
```


![Figure_8](https://github.com/dynexcoin/mode_assisted_unsupervised_learning/blob/main/Figure_8.jpg)

![Figure_1](https://github.com/dynexcoin/mode_assisted_unsupervised_learning/blob/main/Figure_1.png)

### References
[Mode-assisted unsupervised learning of restricted Boltzmann machines](https://arxiv.org/pdf/2001.05559.pdf), Communications Physics volume 3, Article number:105 (2020)
