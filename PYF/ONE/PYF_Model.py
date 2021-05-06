import numpy as np
import joblib

QUANTILES = ['50', '75', '90']
TARGET_VARS = {
    'POE': ['POE'],
    'WOP': ['0.05', '0.1', '0.25', '0.5', '0.75', '0.9', '0.95']
}

LABELS = ['LOW', 'MID', 'HIGH']
class LWModel:
    def __init__(self, WOPS_CLFS, POE_CLF):
        self.WOPS_CLFS = WOPS_CLFS
        self.POE_CLF = POE_CLF
    
    def predict(self, x):
        x[2] /= 100
        x[3] /= 100
        x[4] /= 100
        # x.append(x[1]*x[2])

        poe_preds = self.POE_CLF.predict(np.array([x]))
        poe_probs = self.POE_CLF.predict_proba(np.array([x]))
        
        poe_result = (poe_preds[0], np.max(poe_probs))

        wop_results = {}
        for target in TARGET_VARS['WOP']:
            t_results = {}
            for qnt in QUANTILES:
                target_name = 'WOP_{}_{}'.format(target, qnt)
                target_preds = self.WOPS_CLFS[target_name].predict(np.array([x]))
                target_probs = self.WOPS_CLFS[target_name].predict_proba(np.array([x]))

                res = (target_preds[0], np.max(target_probs))
                t_results[qnt] = res
            wop_results[target] = t_results

        return {
            'POE': poe_result,
            'WOP': wop_results
        }

    def save(self, file_path):
        joblib.dump(self, file_path)