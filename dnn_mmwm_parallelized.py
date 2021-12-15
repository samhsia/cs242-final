import os
import copy
import torch
import numpy as np
import multiprocessing

from lib.utils import *
from lib.data_generator import *
from lib.approximation import Approximation
from lib.compute import Compute
from lib.experiments import Experiments

from torch import nn

from tqdm import tqdm



# Approximation Hyperparameters
MAX_VAL  = 1 # Weights and Data values are always < 1
LUT_BITS = 8
QUANTIZE_FACE = True
SCHEMES_PER_LAYER = 5
dtype=np.float16

# Other Hyperparameters
BATCHSIZE = 64
NUM_BATCHES = 156
SEED=42

if __name__ == "__main__":
    os.system("clear")
    print("################################################################")
    multiprocessing.set_start_method('fork')

    fix_seed(SEED)

    test_loader = create_testloader(BATCHSIZE)
    print('Loaded in test set')

    weights = torch.load('./notebooks/saved_models/weights.pt')
    print('Loaded in saved model weights')

    intermediate_results = []
    final_results = {}

    for i in range(SCHEMES_PER_LAYER):
        final_results[i] = {}
        for j in range(SCHEMES_PER_LAYER):
            final_results[i][j] = {}
            final_results[i][j]['predictions'] = []
    
    for layer_num, layer in enumerate(weights):

        # Note for a DNN, at each layer, we need to re-initialize the approx object since each layer's LUTs are different.
        approx = Approximation(MAX_VAL, dtype)
        comp   = Compute(dtype)
        exp    = Experiments(approx, comp, dtype)

        layer      = layer.T # need to transpose first to have dimensions of (input dim, output dim)
        DIMENSIONS = layer.shape # dimensions are layer-dependent

        initialize_layer(LUT_BITS, QUANTIZE_FACE, approx, layer, layer_num)

        print('Processing Layer {} {}'.format(layer_num, DIMENSIONS))

        # Process First Layer
        if layer_num == 0:

            for batch, (images, labels) in enumerate(tqdm(test_loader)):
                # Limit number of batches otherwise experiments may take too long
                if batch + 1 == NUM_BATCHES:
                    break
                mbs    = images.shape[0] # batch size may be different for last batch.
                images = images.view(mbs, -1).numpy()
                DATA_POINTS = mbs 
                images_conv = initialize_input(approx, images, layer_num, batch)

                # run a single experiment for MMM!
                # Note: adjust the 16 to accomodate dimension sizes
                results = exp.sweep_approximation_pool(images, images_conv, layer, STEP=int(DIMENSIONS[0]/(SCHEMES_PER_LAYER-1)), compute_method="blas") # compute_method either "outer" or "blas"
                final_results['layer_0_perf'] = results

                results = np.array(results['results'])
                results = np.maximum(results, 0 ) # ReLU
                intermediate_results.append(results)
        
        # Process second Layer
        elif layer_num == 1:

            for batch, intermediate_result in enumerate(tqdm(intermediate_results)):
                # print('*****', np.sum(np.array(intermediate_result))) # Unique ID to make sure that intermediate results are different based on approx scheme.
                for layer_0_scheme, i_result in enumerate(intermediate_result):
                    print('Handling Layer 0 - Scheme {}'.format(layer_0_scheme))
                    mbs           = i_result.shape[0]
                    i_result_conv = initialize_input(approx, i_result, layer_num, batch)

                    results = exp.sweep_approximation_pool(i_result, i_result_conv, layer, STEP=int(DIMENSIONS[0]/(SCHEMES_PER_LAYER-1)), compute_method="blas") # compute_method either "outer" or "blas"
                    final_results[layer_0_scheme]['layer_1_perf'] = results

                    results = np.array(results['results'])
                    for layer_1_scheme, f_result in enumerate(results):
                        outputs = nn.functional.softmax(torch.Tensor(f_result), dim = 1)
                        predictions = torch.argmax(outputs, dim=1)
                        final_results[layer_0_scheme][layer_1_scheme]['predictions'].append(predictions)

    find_accuracies(final_results, test_loader, SCHEMES_PER_LAYER, SEED)

    print("################################################################")

    torch.save(final_results, './results/final_results.pt')