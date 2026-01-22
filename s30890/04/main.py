import numpy as np
import pandas as pd

from sklearn.datasets import load_diabetes
from sklearn.impute import SimpleImputer, KNNImputer
from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer

from sklearn.linear_model import LinearRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor

from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset


def make_missing(X, missing_rate, seed=42):
    rng = np.random.RandomState(seed)
    X2 = X.copy()
    mask = rng.rand(*X2.shape) < missing_rate
    X2[mask] = np.nan
    return X2


class MLP(nn.Module):
    def __init__(self, d):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(d, 64), nn.ReLU(), nn.Linear(64, 1))

    def forward(self, x):
        return self.net(x).squeeze(1)


def r2_sklearn(Xm, y, imputer, model):
    Xtr, Xte, ytr, yte = train_test_split(Xm, y, test_size=0.2, random_state=42)
    Xtr = imputer.fit_transform(Xtr)
    Xte = imputer.transform(Xte)
    model.fit(Xtr, ytr)
    pred = model.predict(Xte)
    return r2_score(yte, pred)


def r2_dnn(Xm, y, imputer, epochs=120, lr=1e-3, batch=64):
    Xtr, Xte, ytr, yte = train_test_split(Xm, y, test_size=0.2, random_state=42)
    Xtr = imputer.fit_transform(Xtr)
    Xte = imputer.transform(Xte)

    Xtr = torch.tensor(Xtr, dtype=torch.float32)
    ytr = torch.tensor(ytr, dtype=torch.float32)
    Xte = torch.tensor(Xte, dtype=torch.float32)
    yte = torch.tensor(yte, dtype=torch.float32)

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = MLP(Xtr.shape[1]).to(device)
    opt = torch.optim.Adam(model.parameters(), lr=lr)
    loss_fn = nn.MSELoss()

    dl = DataLoader(TensorDataset(Xtr, ytr), batch_size=batch, shuffle=True)
    model.train()
    for _ in range(epochs):
        for xb, yb in dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            loss = loss_fn(model(xb), yb)
            loss.backward()
            opt.step()

    model.eval()
    with torch.no_grad():
        pred = model(Xte.to(device)).cpu().numpy()
    return r2_score(yte.numpy(), pred)


data = load_diabetes()
X, y = data.data, data.target

imputers = {
    "Mean": SimpleImputer(strategy="mean"),
    "KNN": KNNImputer(n_neighbors=5),
    "MICE": IterativeImputer(random_state=42, max_iter=20),
}

models = {
    "Linear": LinearRegression(),
    "Tree": DecisionTreeRegressor(random_state=42),
    "RF": RandomForestRegressor(n_estimators=100, random_state=42),
}

missing_rates = [0.05, 0.10, 0.20]

rows = []
for mr in missing_rates:
    Xm = make_missing(X, mr, seed=42)

    for imp_name, imp in imputers.items():
        for model_name, model in models.items():
            rows.append({
                "Missing%": int(mr * 100),
                "Imputer": imp_name,
                "Model": model_name,
                "R2": r2_sklearn(Xm, y, imp, model),
            })

    for imp_name, imp in imputers.items():
        rows.append({
            "Missing%": int(mr * 100),
            "Imputer": imp_name,
            "Model": "DNN",
            "R2": r2_dnn(Xm, y, imp),
        })

df = pd.DataFrame(rows)
df.to_csv("wyniki.csv", index=False)
print(df)

best = df.sort_values("R2", ascending=False).groupby("Missing%", as_index=False).head(1).sort_values("Missing%")

with open("wnioski.md", "w", encoding="utf-8") as f:
    f.write("# Ćwiczenia 4 — brakujące dane\n\n")
    f.write("Zbiór: diabetes (regresja). Metryka: R2.\n\n")
    f.write("Najlepsza kombinacja dla poziomu braków:\n\n")
    for _, r in best.iterrows():
        f.write(f"- {int(r['Missing%'])}%: {r['Model']} + {r['Imputer']} -> R2={r['R2']:.4f}\n")
