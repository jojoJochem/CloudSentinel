import numpy as np
import pandas as pd
from tqdm import tqdm
import torch

from utils import SlidingWindowDataset, adjust_anomaly_scores


class Predictor:
    """MTAD-GAT predictor class.

    :param model: MTAD-GAT model (pre-trained) used to forecast and reconstruct
    :param window_size: Length of the input sequence
    :param n_features: Number of input features
    :param pred_args: params for thresholding and predicting anomalies

    """

    def __init__(self, model, window_size, n_features, pred_args, summary_file_name="model_evaluation"):
        self.model = model
        self.window_size = window_size
        self.n_features = n_features
        self.dataset = pred_args["dataset"]
        self.target_dims = pred_args["target_dims"]
        self.scale_scores = pred_args["scale_scores"]
        self.q = pred_args["q"]
        self.level = pred_args["level"]
        self.dynamic_pot = pred_args["dynamic_pot"]
        self.use_mov_av = pred_args["use_mov_av"]
        self.gamma = pred_args["gamma"]
        self.reg_level = pred_args["reg_level"]
        self.save_path = pred_args["save_path"]
        self.batch_size = 256
        self.use_cuda = True  #
        self.pred_args = pred_args
        self.summary_file_name = summary_file_name

    def get_score(self, values):
        """Method that calculates anomaly score using given model and data
        :param values: 2D array of multivariate time series data, shape (N, k)
        :return np array of anomaly scores + dataframe with prediction for each channel and global anomalies
        """

        print("Predicting and calculating anomaly scores..")
        data = SlidingWindowDataset(values, self.window_size, self.target_dims)
        loader = torch.utils.data.DataLoader(data, batch_size=self.batch_size, shuffle=False)
        device = "cuda" if self.use_cuda and torch.cuda.is_available() else "cpu"

        self.model.eval()
        preds = []
        with torch.no_grad():
            for x, y in tqdm(loader):
                x = x.to(device)
                y = y.to(device)

                y_hat = self.model(x)

                preds.append(y_hat.detach().cpu().numpy())

        preds = np.concatenate(preds, axis=0)
        actual = values.detach().cpu().numpy()[self.window_size:]

        if self.target_dims is not None:
            actual = actual[:, self.target_dims]

        anomaly_scores = np.zeros_like(actual)
        df = pd.DataFrame()
        for i in range(preds.shape[1]):
            df[f"Forecast_{i}"] = preds[:, i]
            df[f"True_{i}"] = actual[:, i]
            a_score = np.sqrt((preds[:, i] - actual[:, i]) ** 2)

            if self.scale_scores:
                q75, q25 = np.percentile(a_score, [75, 25])
                iqr = q75 - q25
                median = np.median(a_score)
                a_score = (a_score - median) / (1+iqr)

            anomaly_scores[:, i] = a_score
            df[f"A_Score_{i}"] = a_score
        # print(anomaly_scores.shape)
        score = anomaly_scores
        anomaly_scores = np.mean(anomaly_scores, 1)
        df['A_Score_Global'] = anomaly_scores

        return df, score

    def predict_anomalies_without_labels(self, test, global_epsilon, save_output=True):
        """ Predicts anomalies

        :param test: 2D array of test multivariate time series data
        :param load_scores: Whether to load anomaly scores instead of calculating them
        :param save_output: Whether to save output dataframe
        """

        test_pred_df, test_score = self.get_score(test)
        test_anomaly_scores = test_pred_df['A_Score_Global'].values
        # print("voor adjust", test_anomaly_scores)
        # test_anomaly_scores = adjust_anomaly_scores(test_anomaly_scores, self.dataset, True, self.window_size)
        # print("na adjust", test_anomaly_scores)
        test_pred_df['A_Score_Global'] = test_anomaly_scores

        if self.use_mov_av:
            smoothing_window = int(self.batch_size * self.window_size * 0.05)
            test_anomaly_scores = pd.DataFrame(test_anomaly_scores).ewm(span=smoothing_window).mean().values.flatten()

        test_preds_global = (test_anomaly_scores >= global_epsilon).astype(int)
        test_pred_df["A_Pred_Global"] = test_preds_global
        print(test_pred_df)
        # Save anomaly predictions made using epsilon method (could be changed to pot or bf-method)
        # if save_output:
        #     print(f"Saving output to {self.save_path}/<train/test>_output.pkl")
        #     test_pred_df.to_pickle(f"{self.save_path}/test_output.pkl")

        print("-- Done.")
        return test_pred_df
