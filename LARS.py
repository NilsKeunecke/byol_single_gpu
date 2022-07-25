'''
This implementation is taken from the tensorflow.contrib github page:
Original Implementation: https://github.com/tensorflow/tpu/blob/da262fcba1d0598321d4eb9aa1954fcbf84d1807/models/official/efficientnet/lars_optimizer.py#L24
Paper: https://arxiv.org/abs/1708.03888
'''
import tensorflow as tf
import math


class LARSOptimizer(tf.compat.v1.train.Optimizer):
    """Layer-wise Adaptive Rate Scaling for large batch training.
  Introduced by "Large Batch Training of Convolutional Networks" by Y. You,
  I. Gitman, and B. Ginsburg. (https://arxiv.org/abs/1708.03888)
  Implements the LARS learning rate scheme presented in the paper above. This
  optimizer is useful when scaling the batch size to up to 32K without
  significant performance degradation. It is recommended to use the optimizer
  in conjunction with:
      - Gradual learning rate warm-up
      - Linear learning rate scaling
      - Poly rule learning rate decay
  Note, LARS scaling is currently only enabled for dense tensors. Sparse tensors
  use the default momentum optimizer.
  """

    def __init__(
            self,
            learning_rate,
            current_epoch=0,
            momentum=0.9,
            weight_decay=3 * 10 ** -8,
            # The LARS coefficient is a hyperparameter
            eeta=0.001,
            epsilon=10 ** -6,
            name="LARSOptimizer",
            # Enable skipping variables from LARS scaling.
            # TODO(sameerkm): Enable a direct mechanism to pass a
            # subset of variables to the optimizer.
            skip_list=None,
            use_nesterov=False,
            num_epochs=150,
            warm_up=10,
            batch_size=1000):
        """Construct a new LARS Optimizer.
    Args:
      learning_rate: A `Tensor` or floating point value. The base learning rate.
      momentum: A floating point value. Momentum hyperparameter.
      weight_decay: A floating point value. Weight decay hyperparameter.
      eeta: LARS coefficient as used in the paper. Dfault set to LARS
        coefficient from the paper. (eeta / weight_decay) determines the highest
        scaling factor in LARS.
      epsilon: Optional epsilon parameter to be set in models that have very
        small gradients. Default set to 0.0.
      name: Optional name prefix for variables and ops created by LARSOptimizer.
      skip_list: List of strings to enable skipping variables from LARS scaling.
        If any of the strings in skip_list is a subset of var.name, variable
        'var' is skipped from LARS scaling. For a typical classification model
        with batch normalization, the skip_list is ['batch_normalization',
        'bias']
      use_nesterov: when set to True, nesterov momentum will be enabled
    Raises:
      ValueError: If a hyperparameter is set to a non-sensical value.
    """
        if momentum < 0.0:
            raise ValueError("momentum should be positive: %s" % momentum)
        if weight_decay < 0.0:
            raise ValueError("weight_decay should be positive: %s" % weight_decay)
        super(LARSOptimizer, self).__init__(use_locking=False, name=name)

        self._learning_rate = learning_rate
        self._momentum = momentum
        self._weight_decay = weight_decay
        self._eeta = eeta
        self._epsilon = epsilon
        self._name = name
        self._skip_list = skip_list
        self._use_nesterov = use_nesterov
        self.num_epochs = num_epochs
        self.warm_up = warm_up
        self.batch_size = batch_size
        self.current_epoch = current_epoch

    def update_current_epoch(self, epoch):
        self.current_epoch = epoch

    def _create_slots(self, var_list):
        for v in var_list:
            self._zeros_slot(v, "momentum", self._name)

    def compute_lr(self, grad, var):
        # Compute Cosine Decay learning rate
        cosine_decay = 0.5 * (1 + math.cos(math.pi * (self.current_epoch + 1) / self.num_epochs))
        decayed = (1 - 0.001) * cosine_decay + 0.001
        adapted_lr = (self._learning_rate * self.batch_size / 256) * decayed

        scaled_lr = adapted_lr

        # Perform warm-up if neccesary
        if (self.current_epoch + 1) <= self.warm_up:
            scaled_lr = scaled_lr * (self.current_epoch + 1) / self.warm_up

        if self._skip_list is None or not any(v in var.name
                                              for v in self._skip_list):
            w_norm = tf.norm(var, ord=2)
            g_norm = tf.norm(grad, ord=2)
            trust_ratio = tf.where(
                tf.math.greater(w_norm, 0),
                tf.where(
                    tf.math.greater(g_norm, 0),
                    (self._eeta * w_norm /
                     (g_norm + self._weight_decay * w_norm + self._epsilon)), 1.0),
                1.0)
            scaled_lr = self._learning_rate * trust_ratio
            # Add the weight regularization gradient
            grad = grad + self._weight_decay * var
        return scaled_lr, grad

    def _apply_dense(self, grad, var):
        scaled_lr, grad = self.compute_lr(grad, var)
        mom = self.get_slot(var, "momentum")
        return tf.raw_ops.ApplyMomentum(
            var,
            mom,
            tf.cast(1.0, var.dtype.base_dtype),
            grad * scaled_lr,
            self._momentum,
            use_locking=False,
            use_nesterov=self._use_nesterov)

    def _resource_apply_dense(self, grad, var):
        scaled_lr, grad = self.compute_lr(grad, var)
        mom = self.get_slot(var, "momentum")
        return tf.raw_ops.ResourceApplyMomentum(
            var=var.handle,
            accum=mom.handle,
            lr=tf.cast(1.0, var.dtype.base_dtype),
            grad=grad * scaled_lr,
            momentum=self._momentum,
            use_locking=False,
            use_nesterov=self._use_nesterov)

    # Fallback to momentum optimizer for sparse tensors
    def _apply_sparse(self, grad, var):
        mom = self.get_slot(var, "momentum")
        return tf.raw_ops.SparseApplyMomentum(
            var,
            mom,
            tf.cast(self._learning_rate_tensor, var.dtype.base_dtype),
            grad.values,
            grad.indices,
            tf.cast(self._momentum_tensor, var.dtype.base_dtype),
            use_locking=self._use_locking,
            use_nesterov=self._use_nesterov).op

    def _resource_apply_sparse(self, grad, var, indices):
        mom = self.get_slot(var, "momentum")
        return tf.raw_ops.ResourceSparseApplyMomentum(
            var.handle,
            mom.handle,
            tf.cast(self._learning_rate_tensor, grad.dtype),
            grad,
            indices,
            tf.cast(self._momentum_tensor, grad.dtype),
            use_locking=self._use_locking,
            use_nesterov=self._use_nesterov)

    def _prepare(self):
        learning_rate = self._learning_rate
        if callable(learning_rate):
            learning_rate = learning_rate()
        self._learning_rate_tensor = tf.convert_to_tensor(
            learning_rate, name="learning_rate")
        momentum = self._momentum
        if callable(momentum):
            momentum = momentum()
        self._momentum_tensor = tf.convert_to_tensor(momentum, name="momentum")
