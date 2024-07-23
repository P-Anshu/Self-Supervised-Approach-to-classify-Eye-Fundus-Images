import torch
import torch.nn as nn
import torch.nn.functional as F


def label_to_levels(label, num_classes, dtype=torch.float32):
    """Converts integer class label to extended binary label vector
    Parameters
    ----------
    label : int
        Class label to be converted into a extended
        binary vector. Should be smaller than num_classes-1.
    num_classes : int
        The number of class clabels in the dataset. Assumes
        class labels start at 0. Determines the size of the
        output vector.
    dtype : torch data type (default=torch.float32)
        Data type of the torch output vector for the
        extended binary labels.
    Returns
    ----------
    levels : torch.tensor, shape=(num_classes-1,)
        Extended binary label vector. Type is determined
        by the `dtype` parameter.
    """
    if not label <= num_classes - 1:
        raise ValueError(
            "Class label must be smaller or "
            "equal to %d (num_classes-1). Got %d." % (num_classes - 1, label)
        )
    if isinstance(label, torch.Tensor):
        int_label = label.item()
    else:
        int_label = label

    levels = [1] * int_label + [0] * (num_classes - 1 - int_label)
    levels = torch.tensor(levels, dtype=dtype)
    return levels


def levels_from_labelbatch(labels, num_classes, dtype=torch.float32):
    """
    Converts a list of integer class label to extended binary label vectors
    Parameters
    ----------
    labels : list or 1D orch.tensor, shape=(num_labels,)
        A list or 1D torch.tensor with integer class labels
        to be converted into extended binary label vectors.
    num_classes : int
        The number of class clabels in the dataset. Assumes
        class labels start at 0. Determines the size of the
        output vector.
    dtype : torch data type (default=torch.float32)
        Data type of the torch output vector for the
        extended binary labels.
    Returns
    ----------
    levels : torch.tensor, shape=(num_labels, num_classes-1)
    Examples
    ----------
    >>> levels_from_labelbatch(labels=[2, 1, 4], num_classes=5)
    tensor([[1., 1., 0., 0.],
            [1., 0., 0., 0.],
            [1., 1., 1., 1.]])
    """
    levels = []
    for label in labels:
        levels_from_label = label_to_levels(
            label=label, num_classes=num_classes, dtype=dtype
        )
        levels.append(levels_from_label)

    levels = torch.stack(levels)
    return levels


def proba_to_label(probas):
    """
    Converts predicted probabilities from extended binary format
    to integer class labels
    Parameters
    ----------
    probas : torch.tensor, shape(n_examples, n_labels)
        Torch tensor consisting of probabilities returned by CORAL model.
    Examples
    ----------
    >>> # 3 training examples, 6 classes
    >>> probas = torch.tensor([[0.934, 0.861, 0.323, 0.492, 0.295],
    ...                        [0.496, 0.485, 0.267, 0.124, 0.058],
    ...                        [0.985, 0.967, 0.920, 0.819, 0.506]])
    >>> proba_to_label(probas)
    tensor([2, 0, 5])
    """
    predict_levels = probas > 0.5
    predicted_labels = torch.sum(predict_levels, dim=1)
    return predicted_labels


def coral_loss(logits, levels, importance_weights=None, reduction="mean"):
    """Computes the CORAL loss described
    Parameters
    ----------
    logits : torch.tensor, shape(num_examples, num_classes-1)
        Outputs of the CORAL layer.
    levels : torch.tensor, shape(num_examples, num_classes-1)
        True labels represented as extended binary vectors
        (via `coral_pytorch.dataset.levels_from_labelbatch`).
    importance_weights : torch.tensor, shape=(num_classes-1,) (default=None)
        Optional weights for the different labels in levels.
        A tensor of ones, i.e.,
        `torch.ones(num_classes-1, dtype=torch.float32)`
        will result in uniform weights that have the same effect as None.
    reduction : str or None (default='mean')
        If 'mean' or 'sum', returns the averaged or summed loss value across
        all data points (rows) in logits. If None, returns a vector of
        shape (num_examples,)
    Returns
    ----------
    loss : torch.tensor
        A torch.tensor containing a single loss value (if `reduction='mean'` or '`sum'`)
        or a loss value for each data record (if `reduction=None`).
    """
    if not logits.shape == levels.shape:
        raise ValueError(
            "Please ensure that logits (%s) has the same shape as levels (%s). "
            % (logits.shape, levels.shape)
        )

    term1 = F.logsigmoid(logits) * levels + (F.logsigmoid(logits) - logits) * (
        1 - levels
    )

    if importance_weights is not None:
        term1 *= importance_weights

    val = -torch.sum(term1, dim=1)

    if reduction == "mean":
        loss = torch.mean(val)
    elif reduction == "sum":
        loss = torch.sum(val)
    elif reduction is None:
        loss = val
    else:
        s = (
            'Invalid value for `reduction`. Should be "mean", '
            '"sum", or None. Got %s' % reduction
        )
        raise ValueError(s)

    return loss


def corn_loss(logits, y, num_classes):
    """Computes the CORN loss described in our forthcoming
    Parameters
    ----------
    logits : torch.tensor, shape=(num_examples, num_classes-1)
        Outputs of the CORN layer.
    y_train : torch.tensor, shape=(num_examples)
        Torch tensor containing the class labels.
    num_classes : int
        Number of unique class labels (class labels should start at 0).
    Returns
    ----------
    loss : torch.tensor
        A torch.tensor containing a single loss value.
    """
    sets = []
    for i in range(num_classes - 1):
        label_mask = y > i - 1
        label_tensor = (y[label_mask] > i).to(torch.int64)
        sets.append((label_mask, label_tensor))

    num_examples = 0
    losses = 0.0
    for task_index, s in enumerate(sets):
        train_examples = s[0]
        train_labels = s[1]

        if len(train_labels) < 1:
            continue

        num_examples += len(train_labels)
        pred = logits[train_examples, task_index]

        loss = -torch.sum(
            F.logsigmoid(pred) * train_labels
            + (F.logsigmoid(pred) - pred) * (1 - train_labels)
        )
        losses += loss

    return losses / num_classes


def corn_labels_from_logits(logits):
    """
    Returns the predicted rank label from logits for a
    network trained via the CORN loss.
    Parameters
    ----------
    logits : torch.tensor, shape=(n_examples, n_classes)
        Torch tensor consisting of logits returned by the
        neural net.
    Returns
    ----------
    labels : torch.tensor, shape=(n_examples)
        Integer tensor containing the predicted rank (class) labels
    Examples
    ----------
    >>> # 2 training examples, 5 classes
    >>> logits = torch.tensor([[14.152, -6.1942, 0.47710, 0.96850],
    ...                        [65.667, 0.303, 11.500, -4.524]])
    >>> corn_label_from_logits(logits)
    >>> tensor([1, 3])
    """
    probas = torch.sigmoid(logits)
    probas = torch.cumprod(probas, dim=1)
    predict = probas > 0.5
    preds = torch.sum(predict, dim=1)
    return preds


class CoralLayer(nn.Module):
    """Implements CORAL layer
    Parameters
    -----------
    size_in : int
        Number of input features for the inputs to the forward method, which
        are expected to have shape=(num_examples, num_features).
    num_classes : int
        Number of classes in the dataset.
    preinit_bias : bool (default=True)
        If true, it will pre-initialize the biases to descending values in
        [0, 1] range instead of initializing it to all zeros. This pre-
        initialization scheme results in faster learning and better
        generalization performance in practice.
    """

    def __init__(self, size_in, num_classes, preinit_bias=True):
        super().__init__()
        self.size_in, self.size_out = size_in, 1

        self.coral_weights = torch.nn.Linear(self.size_in, 1, bias=False)
        if preinit_bias:
            self.coral_bias = torch.nn.Parameter(
                torch.arange(num_classes - 1, 0, -1).float() / (num_classes - 1)
            )
        else:
            self.coral_bias = torch.nn.Parameter(torch.zeros(num_classes - 1).float())

    def forward(self, x):
        """
        Computes forward pass.
        Parameters
        -----------
        x : torch.tensor, shape=(num_examples, num_features)
            Input features.
        Returns
        -----------
        logits : torch.tensor, shape=(num_examples, num_classes-1)
        """
        return self.coral_weights(x) + self.coral_bias
