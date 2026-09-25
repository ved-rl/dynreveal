import numpy as np


def score_model(model, true_terms, tol=1e-3):
    feature_names = model.get_feature_names()
    coefs = model.coefficients() 
    eq_names = [f"dx{i+1}" for i in range(coefs.shape[0])]

    n_true, n_pred, n_correct = 0, 0, 0
    coef_errors = []

    for k, eqname in enumerate(eq_names):
        true_eq = true_terms.get(eqname, {})
        recovered = {
            feature_names[i]: coefs[k, i]
            for i in range(len(feature_names))
            if abs(coefs[k, i]) > tol
        }
        n_true += len(true_eq)
        n_pred += len(recovered)
        for term, true_val in true_eq.items():
            if term in recovered:
                n_correct += 1
                coef_errors.append(
                    abs(recovered[term] - true_val) / (abs(true_val) + 1e-8)
                )

    precision = n_correct / n_pred if n_pred else 0.0
    recall = n_correct / n_true if n_true else 0.0
    coef_err = float(np.mean(coef_errors)) if coef_errors else np.nan
    return dict(precision=precision, recall=recall, coef_err=coef_err)
