import numpy as np
import torch


def is_list_or_tuple(x):
    """
    Check if the input is a list or a tuple.
    Parameters
    ----------
    x : any type
        The input to check.
    Returns
    -------
    bool
        True if the input is a list or a tuple, False otherwise.
    """
    return isinstance(x, list) or isinstance(x, tuple)


def assert_numpy(x, arr_type=None):
    """
    Convert the input to a NumPy array if it is not already one.

    Parameters
    ----------
    x : array-like or torch.Tensor
        The input to be converted to a NumPy array.
    arr_type : data-type, optional
        The desired data-type for the output array. If None, the data-type is not changed.

    Returns
    -------
    numpy.ndarray
        The converted NumPy array.

    Raises
    ------
    AssertionError
        If the input cannot be converted to a NumPy array.
    """
    if isinstance(x, torch.Tensor):
        if x.is_cuda:
            x = x.cpu()
        x = x.detach().numpy()
    if is_list_or_tuple(x):
        x = np.array(x)
    assert isinstance(x, np.ndarray)
    if arr_type is not None:
        x = x.astype(arr_type)
    return x


def try_gpu(i=0):
    """
    Returns a GPU device if available, otherwise returns a CPU device.

    Parameters
    ----------
    i : int, optional
        The index of the GPU device to use (default is 0).

    Returns
    -------
    torch.device
        The specified GPU device if available, otherwise a CPU device.
    """
    if torch.cuda.device_count() >= i + 1:
        return torch.device(f"cuda:{i}")
    return torch.device("cpu")


def assert_tensor(x, arr_type=None, device=try_gpu()):
    """
    Convert input to a PyTorch tensor and ensure it is of the specified type and device.

    Parameters
    ----------
    x : np.ndarray, list, tuple, or torch.Tensor
        The input data to be converted to a PyTorch tensor.
    arr_type : torch.dtype, optional
        The desired data type of the tensor. If None, the tensor retains its original type.
    device : torch.device, optional
        The device on which to place the tensor. Defaults to the result of `try_gpu()`.

    Returns
    -------
    torch.Tensor
        The input data converted to a PyTorch tensor, with the specified type and device.

    Raises
    ------
    AssertionError
        If the input data cannot be converted to a PyTorch tensor.
    """
    if isinstance(x, np.ndarray):
        x = torch.tensor(x, device=device)
    if is_list_or_tuple(x):
        x = np.array(x)
        x = torch.tensor(x, device=device)
    assert isinstance(x, torch.Tensor)
    if arr_type is not None:
        x = x.to(arr_type)
    return x


def assert_list(a, length, dtype=int):
    """
    Ensures that the input `a` is a list of a specified length and type.

    Parameters
    ----------
    a : int, list
        The input to be converted to a list. If `a` is of type `dtype`, it will be converted to a list of length `length` with all elements being `a`.
        If `a` is already a list, its length will be checked against `length`.
    length : int
        The desired length of the list.
    dtype : type, optional
        The expected type of the elements in the list. Default is `int`.

    Returns
    -------
    list
        A list of length `length` with elements of type `dtype`.

    Raises
    ------
    AssertionError
        If `a` is a list and its length is not equal to `length`.
    """
    if isinstance(a, dtype):
        a = [a] * length
    elif isinstance(a, list):
        assert len(a) == length
    return a
