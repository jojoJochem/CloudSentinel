import os
import time
import torch
import math
import warnings
import base64
import json
import matplotlib
import logging
import traceback
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import networkx as nx
import torch.optim as optim
import torch.nn.functional as F

from torch.autograd import Variable
from torch.optim import lr_scheduler
from io import BytesIO
from sknetwork.ranking import PageRank

from utils import (matrix_poly, get_triu_offdiag_indices, get_tril_offdiag_indices,
                   nll_gaussian, kl_gaussian_sem, A_connect_loss, A_positive_loss)
from modules import MLPEncoder, MLPDecoder, SEMEncoder, SEMDecoder
from config import get_config

# Set matplotlib backend to 'Agg' for non-interactive plotting
matplotlib.use("Agg")

# Ignore warnings
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def run_crca(crca_data, crca_info, task_id, parallel=False):
    """
    Run the CRCA (Causal Relation Extraction) algorithm.

    Args:
    - crca_data (pd.DataFrame): Data for CRCA algorithm.
    - crca_info (dict): Information related to CRCA.
    - task_id (str): Identifier for the current task.
    - parallel (bool, optional): Whether to use parallel processing. Defaults to False.

    Returns:
    - dict: Response data containing containers, metrics, ranking, and graph image.
    """
    try:
        logger.info("Starting CRCA run for task %s", task_id)
        # Get number of iterations from configuration
        iterations_arg = get_config()['arg_iterations']

        # Perform CRCA sequentially
        results = crca_sequential(crca_data, iterations_arg)

        # Concatenate results and calculate average
        df_concat = pd.concat([result[0] for result in results])
        image_base64 = [result[1] for result in results]
        cumulative_df = calculate_average(df_concat)

        # Match column names and prepare response data
        name_matched_df = match_names(cumulative_df, crca_info)
        response_data = return_json_object(name_matched_df, image_base64, crca_info, task_id)

        logger.info("CRCA run completed for task %s", task_id)
        return response_data
    except Exception as e:
        logger.error("Error in run_crca: %s", traceback.format_exc())
        raise e


def return_json_object(df, image_base64, crca_info, task_id):
    """
    Create a JSON object from DataFrame and image base64 data.

    Args:
    - df (pd.DataFrame): DataFrame containing ranking information.
    - image_base64 (str): Base64-encoded image data.
    - crca_info (dict): Information related to CRCA.
    - task_id (str): Identifier for the current task.

    Saves the JSON object to a file.
    """
    try:
        logger.info("Creating JSON object for task %s", task_id)
        # Prepare response data
        csv_payload = df.to_csv(index=False)
        response_data = {
            'containers': crca_info['data']['crca_pods'],
            'metrics': crca_info['data']['metrics'],
            'ranking': csv_payload,
            'graph_image': image_base64
        }

        # Save response data to a JSON file
        path = 'results/' + task_id
        if not os.path.isdir(path):
            os.mkdir(path)
        with open(path + '/crca_results.json', 'w') as f:
            json.dump(response_data, f, indent=2)
        logger.info("JSON object created and saved for task %s", task_id)
    except Exception as e:
        logger.error("Error in return_json_object: %s", traceback.format_exc())
        raise e


def match_names(df_concat, crca_info):
    """
    Match column names based on CRCA information.

    Args:
    - df_concat (pd.DataFrame): DataFrame containing concatenated results.
    - crca_info (dict): Information related to CRCA.

    Returns:
    - pd.DataFrame: DataFrame with matched column names.
    """
    try:
        logger.info("Matching column names")
        # Add index column and match container and metric names
        df_concat['index'] = df_concat['column_nr'].astype(int)
        selected_containers = crca_info['data']['crca_pods']
        selected_metrics = crca_info['data']['metrics']
        column_names = [f"{container}_{metric}" for container in selected_containers for metric in selected_metrics]
        name_matched_df = df_concat.copy()
        name_matched_df['metric'] = name_matched_df['column_nr'].apply(lambda x: column_names[x])
        name_matched_df = name_matched_df[['metric', 'index', 'score']]
        logger.info("Matched column names: %s", name_matched_df)
        return name_matched_df
    except Exception as e:
        logger.error("Error in match_names: %s", traceback.format_exc())
        raise e


def crca_sequential(data_arg, iterations_arg):
    """
    Perform CRCA algorithm sequentially for multiple iterations.

    Args:
    - data_arg (pd.DataFrame): Data for CRCA algorithm.
    - iterations_arg (int): Number of iterations to run.

    Returns:
    - list: List of results from each iteration.
    """
    try:
        results = []
        for i in range(iterations_arg):
            logger.info("Running iteration %d of %d", i + 1, iterations_arg)
            result = crca(data_arg)
            results.append(result)
        return results
    except Exception as e:
        logger.error("Error in crca_sequential: %s", traceback.format_exc())
        raise e


def calculate_average(df):
    """
    Calculate the average score from the DataFrame.

    Args:
    - df (pd.DataFrame): DataFrame containing scores.

    Returns:
    - pd.DataFrame: DataFrame with average scores.
    """
    try:
        logger.info("Calculating average scores")
        cumulative_df = df.groupby('column_nr')['score'].mean().reset_index().sort_values(by='score', ascending=False)
        return cumulative_df
    except Exception as e:
        logger.error("Error in calculate_average: %s", traceback.format_exc())
        raise e


def crca(data_arg):
    """
    CRCA algorithm implementation.

    Args:
    - data_arg (pd.DataFrame): Data for CRCA algorithm.

    Returns:
    - pd.DataFrame: DataFrame with sorted scores.
    - str: Base64-encoded image data.
    """
    try:
        logger.info("Starting CRCA algorithm")
        config = get_config()
        gamma_arg = config['arg_gamma']
        eta_arg = config['arg_eta']
        data = data_arg
        data_sample_size = data.shape[0]
        data_variable_size = data.shape[1]

        config['cuda'] = torch.cuda.is_available()
        config['factor'] = not config['no_factor']

        train_data = data

        # Generate off-diagonal interaction graph
        num_nodes = data_variable_size
        adj_A = np.zeros((num_nodes, num_nodes))

        if config['encoder'] == 'mlp':
            encoder = MLPEncoder(data_variable_size * config['x_dims'],
                                 config['x_dims'],
                                 config['encoder_hidden'],
                                 int(config['z_dims']),
                                 adj_A,
                                 batch_size=config['batch_size'],
                                 do_prob=config['encoder_dropout'],
                                 factor=config['factor']).double()
        elif config['encoder'] == 'sem':
            encoder = SEMEncoder(data_variable_size * config['x_dims'],
                                 config['encoder_hidden'],
                                 int(config['z_dims']),
                                 adj_A,
                                 batch_size=config['batch_size'],
                                 do_prob=config['encoder_dropout'],
                                 factor=config['factor']).double()

        if config['decoder'] == 'mlp':
            decoder = MLPDecoder(data_variable_size * config['x_dims'],
                                 config['z_dims'],
                                 config['x_dims'],
                                 encoder,
                                 data_variable_size=data_variable_size,
                                 batch_size=config['batch_size'],
                                 n_hid=config['decoder_hidden'],
                                 do_prob=config['decoder_dropout']).double()
        elif config['decoder'] == 'sem':
            decoder = SEMDecoder(data_variable_size * config['x_dims'],
                                 config['z_dims'],
                                 2,
                                 encoder,
                                 data_variable_size=data_variable_size,
                                 batch_size=config['batch_size'],
                                 n_hid=config['decoder_hidden'],
                                 do_prob=config['decoder_dropout']).double()

        if config['optimizer'] == 'Adam':
            optimizer = optim.Adam(list(encoder.parameters()) + list(decoder.parameters()), lr=config['lr'])
        elif config['optimizer'] == 'LBFGS':
            optimizer = optim.LBFGS(list(encoder.parameters()) + list(decoder.parameters()), lr=config['lr'])
        elif config['optimizer'] == 'SGD':
            optimizer = optim.SGD(list(encoder.parameters()) + list(decoder.parameters()), lr=config['lr'])

        scheduler = lr_scheduler.StepLR(optimizer, step_size=config['lr_decay'], gamma=config['gamma'])

        triu_indices = get_triu_offdiag_indices(data_variable_size)
        tril_indices = get_tril_offdiag_indices(data_variable_size)

        if config['prior']:
            prior = np.array([0.91, 0.03, 0.03, 0.03])  # hard coded for now
            log_prior = torch.DoubleTensor(np.log(prior))
            log_prior = torch.unsqueeze(log_prior, 0)
            log_prior = torch.unsqueeze(log_prior, 0)
            log_prior = Variable(log_prior)

            if config['cuda']:
                log_prior = log_prior.cuda()

        if config['cuda']:
            encoder.cuda()
            decoder.cuda()
            triu_indices = triu_indices.cuda()
            tril_indices = tril_indices.cuda()

        def _h_A(A, m):
            expm_A = matrix_poly(A * A, m)
            h_A = torch.trace(expm_A) - m
            return h_A

        prox_plus = torch.nn.Threshold(0., 0.)

        def stau(w, tau):
            w1 = prox_plus(torch.abs(w) - tau)
            return torch.sign(w) * w1

        def update_optimizer(optimizer, original_lr, c_A):
            '''related LR to c_A, whenever c_A gets big, reduce LR proportionally'''
            MAX_LR = 1e-2
            MIN_LR = 1e-4

            estimated_lr = original_lr / (math.log10(c_A) + 1e-10)
            if estimated_lr > MAX_LR:
                lr = MAX_LR
            elif estimated_lr < MIN_LR:
                lr = MIN_LR
            else:
                lr = estimated_lr

            for parame_group in optimizer.param_groups:
                parame_group['lr'] = lr

            return optimizer, lr

        def train(epoch, best_val_loss, lambda_A, c_A, optimizer):
            nll_train = []
            kl_train = []
            mse_train = []

            encoder.train()
            decoder.train()
            scheduler.step()

            optimizer, lr = update_optimizer(optimizer, config['lr'], c_A)

            for i in range(1):
                data = train_data[i * data_sample_size:(i + 1) * data_sample_size]
                data = torch.tensor(data.to_numpy().reshape(data_sample_size, data_variable_size, 1))
                if config['cuda']:
                    data = data.cuda()
                data = Variable(data).double()

                optimizer.zero_grad()

                enc_x, logits, origin_A, adj_A_tilt_encoder, z_gap, z_positive, myA, Wa = encoder(data)
                edges = logits
                dec_x, output, adj_A_tilt_decoder = decoder(data, edges, data_variable_size * config['x_dims'],
                                                            origin_A, adj_A_tilt_encoder, Wa)

                if torch.sum(output != output):
                    logger.error('NaN error encountered during training')
                    raise ValueError('NaN error encountered during training')

                target = data
                preds = output
                variance = 0.

                loss_nll = nll_gaussian(preds, target, variance)
                loss_kl = kl_gaussian_sem(logits)
                loss = loss_kl + loss_nll

                one_adj_A = origin_A
                sparse_loss = config['tau_A'] * torch.sum(torch.abs(one_adj_A))

                if config['use_A_connect_loss']:
                    connect_gap = A_connect_loss(one_adj_A, config['graph_threshold, z_gap'])
                    loss += lambda_A * connect_gap + 0.5 * c_A * connect_gap * connect_gap

                if config['use_A_positiver_loss']:
                    positive_gap = A_positive_loss(one_adj_A, z_positive)
                    loss += .1 * (lambda_A * positive_gap + 0.5 * c_A * positive_gap * positive_gap)

                h_A = _h_A(origin_A, data_variable_size)
                loss += lambda_A * h_A + 0.5 * c_A * h_A * h_A + 100. * torch.trace(origin_A * origin_A) + sparse_loss

                loss.backward()
                optimizer.step()

                myA.data = stau(myA.data, config['tau_A'] * lr)

                if torch.sum(origin_A != origin_A):
                    logger.error('NaN error encountered during optimizer step')
                    raise ValueError('NaN error encountered during optimizer step')

                graph = origin_A.data.clone().cpu().numpy()
                graph[np.abs(graph) < config['graph_threshold']] = 0

                mse_train.append(F.mse_loss(preds, target).item())
                nll_train.append(loss_nll.item())
                kl_train.append(loss_kl.item())

            return np.mean(np.mean(kl_train) + np.mean(nll_train)), np.mean(nll_train), np.mean(mse_train), graph, origin_A

        gamma = gamma_arg
        eta = eta_arg

        best_ELBO_loss = np.inf
        best_NLL_loss = np.inf
        best_MSE_loss = np.inf
        best_epoch = 0
        c_A = config['c_A']
        lambda_A = config['lambda_A']
        h_A_new = torch.tensor(1.)
        h_tol = config['h_tol']
        k_max_iter = int(config['k_max_iter'])
        h_A_old = np.inf

        E_loss = []
        N_loss = []
        M_loss = []
        start_time = time.time()
        try:
            for step_k in range(k_max_iter):
                logger.info("Running step %d of %d", step_k, k_max_iter)
                while c_A < 1e+20:
                    for epoch in range(config['epochs']):
                        ELBO_loss, NLL_loss, MSE_loss, graph, origin_A = train(epoch, best_ELBO_loss,
                                                                               lambda_A, c_A, optimizer)
                        E_loss.append(ELBO_loss)
                        N_loss.append(NLL_loss)
                        M_loss.append(MSE_loss)
                        if ELBO_loss < best_ELBO_loss:
                            best_ELBO_loss = ELBO_loss
                            best_epoch = epoch

                        if NLL_loss < best_NLL_loss:
                            best_NLL_loss = NLL_loss
                            best_epoch = epoch

                        if MSE_loss < best_MSE_loss:
                            best_MSE_loss = MSE_loss
                            best_epoch = epoch

                    if ELBO_loss > 2 * best_ELBO_loss:
                        break

                    A_new = origin_A.data.clone()
                    h_A_new = _h_A(A_new, data_variable_size)
                    if h_A_new.item() > gamma * h_A_old:
                        c_A *= eta
                    else:
                        break

                h_A_old = h_A_new.item()
                lambda_A += c_A * h_A_new.item()

                if h_A_new.item() <= h_tol:
                    break

            logger.info("Best Epoch: %d", best_epoch)

            graph = origin_A.data.clone().cpu().numpy()
            graph[np.abs(graph) < 0.1] = 0
            graph[np.abs(graph) < 0.2] = 0
            graph[np.abs(graph) < 0.3] = 0

        except KeyboardInterrupt:
            logger.info('Training interrupted by user')

        end_time = time.time()
        logger.info("Time spent: %f seconds", end_time - start_time)

        adj = graph
        org_G = nx.from_numpy_matrix(adj, parallel_edges=True, create_using=nx.DiGraph)
        pos = nx.circular_layout(org_G)
        plt.figure(figsize=(8, 6))
        nx.draw(org_G, pos=pos, with_labels=True, node_color='skyblue', edge_color='k')

        img_data = BytesIO()
        plt.savefig(img_data, format='png')
        img_data.seek(0)

        image_base64 = base64.b64encode(img_data.getvalue()).decode('utf-8')

        pagerank = PageRank()
        scores = pagerank.fit_transform(np.abs(adj.T))

        score_dict = {i: s for i, s in enumerate(scores)}
        sorted_dict = sorted(score_dict.items(), key=lambda item: item[1], reverse=True)
        for i in range(0, len(sorted_dict), 3):
            if i < len(sorted_dict):
                logger.info(f"{i + 1}: {sorted_dict[i]}")
            if i + 1 < len(sorted_dict):
                logger.info(f"{i + 2}: {sorted_dict[i + 1]}")
            if i + 2 < len(sorted_dict):
                logger.info(f"{i + 3}: {sorted_dict[i + 2]}")

        df = pd.DataFrame(sorted_dict)
        df.columns = ['column_nr', 'score']

        logger.info("CRCA algorithm completed successfully")
        return df, image_base64
    except Exception as e:
        logger.error("Error in crca: %s", traceback.format_exc())
        raise e
