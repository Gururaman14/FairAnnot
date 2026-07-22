# cspell:disable
import numpy as np
import pandas as pd


class DawidSkene:
    def __init__(self, n_iter=50, tol=1e-6, verbose=True):
        self.n_iter = n_iter
        self.tol = tol
        self.verbose = verbose

    def fit(self, annotations_df):
        self._setup(annotations_df)
        self.T_ = self._init_majority_vote()

        prev_ll = -np.inf
        for iteration in range(self.n_iter):
            self.pi_, self.p_ = self._m_step()
            self.T_ = self._e_step()

            ll = self._log_likelihood()
            delta = abs(ll - prev_ll)

            if self.verbose:
                print(f"  Iter {iteration + 1:3d} | log-likelihood: {ll:>14.4f} | delta: {delta:.2e}")

            if delta < self.tol:
                if self.verbose:
                    print(f"  Converged after {iteration + 1} iterations.")
                break
            prev_ll = ll

        return self

    def get_soft_labels(self):
        cols = {f'p_{c}': self.T_[:, k] for k, c in enumerate(self.classes_)}
        df = pd.DataFrame({'item_id': self.items_} | cols)
        df['predicted_label'] = [self.classes_[k] for k in self.T_.argmax(axis=1)]
        df['confidence'] = self.T_.max(axis=1)
        return df

    def get_annotator_weights(self):
        rows = []
        for a_idx, a_id in enumerate(self.annotators_):
            diag = np.diag(self.pi_[a_idx])
            n_ann = int((self.ann_idxs_ == a_idx).sum())
            rows.append({
                'annotator_id': a_id,
                'reliability_score': float(diag.mean()),
                'num_annotations': n_ann,
            })
        return pd.DataFrame(rows)

    def get_confusion_matrices(self):
        return {
            ann_id: pd.DataFrame(
                self.pi_[a_idx],
                index=[f'true_{c}' for c in self.classes_],
                columns=[f'pred_{c}' for c in self.classes_],
            )
            for a_idx, ann_id in enumerate(self.annotators_)
        }

    def _setup(self, annotations_df):
        self.items_ = sorted(annotations_df['item_id'].unique())
        self.annotators_ = sorted(annotations_df['annotator_id'].unique())
        self.classes_ = sorted(annotations_df['label'].unique())

        self.n_items_ = len(self.items_)
        self.n_annotators_ = len(self.annotators_)
        self.n_classes_ = len(self.classes_)

        item_idx = {v: i for i, v in enumerate(self.items_)}
        ann_idx = {v: i for i, v in enumerate(self.annotators_)}
        class_idx = {v: i for i, v in enumerate(self.classes_)}

        self.item_idxs_ = annotations_df['item_id'].map(item_idx).to_numpy()
        self.ann_idxs_ = annotations_df['annotator_id'].map(ann_idx).to_numpy()
        self.label_idxs_ = annotations_df['label'].map(class_idx).to_numpy()

        if self.verbose:
            print(f"  Items: {self.n_items_:,} | Annotators: {self.n_annotators_:,} "
                  f"| Classes: {self.n_classes_} {self.classes_} "
                  f"| Annotations: {len(annotations_df):,}")

    def _init_majority_vote(self):
        T = np.zeros((self.n_items_, self.n_classes_))
        np.add.at(T, (self.item_idxs_, self.label_idxs_), 1.0)
        row_sums = T.sum(axis=1, keepdims=True)
        row_sums[row_sums == 0] = 1
        T /= row_sums
        return T

    def _m_step(self):
        eps = 1e-10

        p = self.T_.mean(axis=0) + eps
        p /= p.sum()

        pi = np.zeros((self.n_annotators_, self.n_classes_, self.n_classes_)) + eps
        T_at_anns = self.T_[self.item_idxs_] 

        for k in range(self.n_classes_):
            temp = np.zeros((self.n_annotators_, self.n_classes_))
            np.add.at(temp, (self.ann_idxs_, self.label_idxs_), T_at_anns[:, k])
            pi[:, k, :] += temp

        pi /= pi.sum(axis=2, keepdims=True)
        return pi, p

    def _e_step(self):
        eps = 1e-10

        log_T = np.tile(np.log(self.p_ + eps), (self.n_items_, 1)) 

        log_pi_contrib = np.log(self.pi_[self.ann_idxs_, :, self.label_idxs_] + eps)
        np.add.at(log_T, self.item_idxs_, log_pi_contrib)

        log_T -= log_T.max(axis=1, keepdims=True)
        T = np.exp(log_T)
        T /= T.sum(axis=1, keepdims=True)
        return T

    def _log_likelihood(self):
        eps = 1e-10
        log_item = np.tile(np.log(self.p_ + eps), (self.n_items_, 1))
        log_pi_contrib = np.log(self.pi_[self.ann_idxs_, :, self.label_idxs_] + eps)
        np.add.at(log_item, self.item_idxs_, log_pi_contrib)

        shift = log_item.max(axis=1, keepdims=True)
        ll = (np.log(np.exp(log_item - shift).sum(axis=1)) + shift.squeeze()).sum()
        return float(ll)
