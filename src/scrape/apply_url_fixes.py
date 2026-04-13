import pandas as pd

MAIN = "output/researchers_clean.csv"
FIXES = "output/url_validation_results.csv"
OUT = "output/researchers_clean_fixed.csv"

df = pd.read_csv(MAIN)
fx = pd.read_csv(FIXES)

# Keep only actually usable URLs (2xx/3xx)
fx = fx[fx["best_status"].fillna(0).astype(int).between(200, 399)].copy()

# We merge by name (good enough for now; later use an ID if you have one)
fx = fx[["name", "best_final_or_error"]].rename(columns={"name": "Full Name", "best_final_or_error": "Personal Website (fixed)"})

out = df.merge(fx, on="Full Name", how="left")

# If we have a fixed url, replace Personal Website
out["Personal Website"] = out["Personal Website (fixed)"].fillna(out["Personal Website"])
out = out.drop(columns=["Personal Website (fixed)"])

out.to_csv(OUT, index=False)
print("Wrote:", OUT)
