# Standard Libraries
import importlib
import os
import sys
from typing import List, Tuple, Union
from Net import *
# Thirs Party Imports
import numpy as np
import tensorflow as tf



def is_prime(n):
    """Check if n is a prime number. """
    if n <= 1:
        return False
    if n <= 3:
        return True
    if n % 2 == 0 or n % 3 == 0:
        return False
    i = 5
    while i * i <= n:
        if n % i == 0 or n % (i + 2) == 0:
            return False
        i += 6
    return True

def next_prime(n):
    """Find the smallest prime number larger than n. """
    if n <= 1:
        return 2
    prime = n
    found = False
    while not found:
        prime += 1
        if is_prime(prime):
            found = True
    return prime

def get_flat_weights(model) -> Tuple[List[int], List[int]]:
    """ Get model weights, convert and flatten nested tensor to a list, original shape is saved. """
    params = get_weights(model)
    original_shapes = [tensor.shape for tensor in params]
    flattened_weights = []
    for weight_tensor in params:
        # Convert the tensor to a numpy array
        weight_array = weight_tensor.numpy()
        #print("WEIGHT ARRAY",weight_array)
        # Flatten the numpy array and add it to the flattened_weights list
        flattened_weights.extend(weight_array.flatten())
    return flattened_weights, original_shapes

def get_flat_bias(model) -> Tuple[List[int], List[int]]:
    """ Get model weights, convert and flatten nested tensor to a list, original shape is saved. """
    params = get_bias(model)
    original_shapes = [tensor.shape for tensor in params]
    flattened_bias = []
    for bias_tensor in params:
        # Convert the tensor to a numpy array
        bias_array = bias_tensor.numpy()#前面直接处理了的化取原本大小的命令也要改
        #print("WEIGHT ARRAY",weight_array)
        # Flatten the numpy array and add it to the flattened_weights list
        #好像bias本身就是一维的？？？
        flattened_bias.extend(bias_array.flatten())
    return flattened_bias, original_shapes


def unflatten_weights(flattened_weights, original_shapes):
    """Convert flat list into the model's original nested tensors. """
    unflattened_weights = []
    current_index = 0
    for shape in original_shapes:
        # Calculate the number of elements in the current shape
        num_elements = np.prod(shape)
        
        # Slice the flattened_weights list to get the elements for the current shape
        current_elements = flattened_weights[current_index : current_index + num_elements]
        #print("In unflatten ",current_index,num_elements,len(current_elements))
        # Reshape the elements to the original shape and append them to the unflattened_weights list
        reshaped_elements = np.reshape(current_elements, shape)
        unflattened_weights.append(torch.tensor(reshaped_elements, dtype=torch.float))
        # Update the index for the next iteration
        current_index += num_elements
    return unflattened_weights


def pad_to_power_of_2(flat_params, target_length=2**13, weight_decimals=7):
    """
    - Client Side
    - Before encryption of local weights:
    - Pad flat weights to nearest 2^n, original length is saved. 
    """
    pad_length = target_length - len(flat_params)
    if pad_length < 0:
        raise ValueError("The given target_length is smaller than the current parameter list length.")
    # Let the padding be random numbers within the min and max values of the weights
    random_padding = np.random.randint(-10**weight_decimals, 10**weight_decimals + 1, pad_length).tolist()
    padded_params = flat_params + random_padding
    return padded_params, len(flat_params)


def remove_padding(padded_params, original_length):
    """
    - Client Side
    - After receiving decrypted model updates:
    - Remove the padding that was added during encryption 
    """
    return padded_params[:original_length]
