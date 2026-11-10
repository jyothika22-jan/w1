import argparse
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd

SEED = 42
np.random.seed(SEED)

MANUAL_MAP: dict = {}

ALIASES = {
    "invoice_id": ["invoice_id", "invoice_no", "invoice_number", "invoice", "inv_id", "id"],
    "customer": ["customer", "customer_id", "customer_name", "client", "client_id",
                 "buyer", "buyer_id", "account", "company"],
    "invoice_date": ["invoice_date", "issue_date", "issued_date", "billing_date",
                     "invoice_issue_date", "date", "created"],
    "due_date": ["due_date", "payment_due_date", "duedate", "due"],
    "payment_date": ["payment_date", "paid_date", "date_paid", "settlement_date",
                     "clear_date", "payment_received_date", "paid_on"],
    "payment_status": ["payment_status", "invoice_status", "status", "paid_status", "is_paid"],
    "amount": ["amount", "invoice_amount", "total_amount", "amount_due", "invoice_total",
               "total", "billing_amount", "net_amount", "value"],
    "currency": ["currency", "currency_code", "curr", "ccy"],
    "country": ["country", "country_code", "customer_country"],
}

FIELD_MEANING = {
    "invoice_id": "Unique invoice identifier",
    "customer": "Customer / buyer identifier or name",
    "invoice_date": "Date the invoice was issued",
    "due_date": "Date payment is contractually due",
    "payment_date": "Date the payment was received (blank = not paid yet)",
    "payment_status": "Payment status as recorded in the source",
    "amount": "Invoice amount in the invoice currency",
    "currency": "ISO currency code of the amount",
    "country": "Customer country",
}

COUNTRY_FIX = {
    "USA": "United States", "US": "United States", "U.S.": "United States",
    "UNITED STATES OF AMERICA": "United States", "UK": "United Kingdom",
    "U.K.": "United Kingdom", "GREAT BRITAIN": "United Kingdom",
    "UAE": "United Arab Emirates", "SOUTH KOREA": "South Korea", "KOREA": "South Korea",
}

DROP_NONPOSITIVE_AMOUNT = True   
DROP_IMPOSSIBLE_DATES = True     
CAP_PERCENTILE = 0.995           
MIN_GROUP_FOR_CAP = 50           


def norm_col(c: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(c).strip().lower()).strip("_")


class CleaningLog:
    def __init__(self):
        self.rows = []

    def add(self, issue, action, reason, n):
        self.rows.append({"issue": issue, "action": action, "reason": reason,
                          "rows_affected": int(n)})
        print(f"[LOG] {issue:<45} -> {action:<28} ({int(n)} rows)")

    def to_df(self):
        return pd.DataFrame(self.rows)


def read_any(path: str) -> pd.DataFrame:
    p = Path(path)
    if p.suffix.lower() in (".parquet", ".pq"):
        df = pd.read_parquet(p)
    elif p.suffix.lower() in (".xlsx", ".xls"):
        df = pd.read_excel(p)
    else:
        df = pd.read_csv(p, low_memory=False)
    df.columns = [norm_col(c) for c in df.columns]
    return df


def build_mapping(cols):
    mapping, used = {}, set()
    for field, aliases in ALIASES.items():
        if field in MANUAL_MAP:
            mapping[field] = norm_col(MANUAL_MAP[field])
            used.add(mapping[field])
            continue
        mapping[field] = None
        for a in aliases:
            if a in cols and a not in used:
                mapping[field] = a
                used.add(a)
                break
    return mapping


def parse_dates(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", format="mixed")
    except (TypeError, ValueError):  # older pandas
        return pd.to_datetime(s, errors="coerce")


def parse_amount(s: pd.Series) -> pd.Series:
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)
    t = s.astype("string").str.strip()
    neg = t.str.match(r"^\(.*\)$", na=False)          # (123.45) -> negative
    t = t.str.replace(",", "", regex=False).str.replace(r"[^\d.\-]", "", regex=True)
    out = pd.to_numeric(t, errors="coerce")
    return out.where(~neg, -out.abs())


def clean_text(s: pd.Series) -> pd.Series:
    return (s.astype("string").str.strip().str.replace(r"\s+", " ", regex=True)
            .replace({"": pd.NA, "nan": pd.NA, "NaN": pd.NA, "None": pd.NA,
                      "NULL": pd.NA, "null": pd.NA, "N/A": pd.NA, "n/a": pd.NA}))


def status_flags(s: pd.Series):
    """Fallback when there is no exact payment date. CONFIRM THIS WITH YOUR MENTOR."""
    t = s.astype("string").str.lower().str.strip()
    unpaid = t.str.contains(r"unpaid|not paid|open|pending|outstanding|due|overdue", na=False)
    paid = t.str.contains(r"paid|settled|cleared|closed", na=False) & ~t.str.contains(
        r"unpaid|not paid", na=False)
    late = t.str.contains(r"late|overdue|delay", na=False)
    return paid, late


def build_data_dictionary(raw: pd.DataFrame, mapping: dict) -> pd.DataFrame:
    rev = {v: k for k, v in mapping.items() if v}
    rows = []
    for c in raw.columns:
        if c == "split":
            continue
        s = raw[c]
        field = rev.get(c, "")
        ex = s.dropna().iloc[0] if s.notna().any() else ""
        rows.append({
            "column": c, "dtype": str(s.dtype), "non_null": int(s.notna().sum()),
            "missing": int(s.isna().sum()), "n_unique": int(s.nunique(dropna=True)),
            "example": ex, "project_field": field,
            "meaning": FIELD_MEANING.get(field, "TODO: write one-line meaning"),
        })
    dd = pd.DataFrame(rows)
    missing_fields = [f for f, c in mapping.items() if c is None]
    for f in missing_fields:
        dd.loc[len(dd)] = ["(not in dataset)", "", 0, 0, 0, "", f,
                           "UNAVAILABLE - discuss alternative with mentor"]
    return dd


DERIVED_DICT = [
    ("payment_terms_days", "due_date - invoice_date (days)"),
    ("days_to_pay", "payment_date - invoice_date (days); NaN while unpaid"),
    ("delay_days", "payment_date - due_date (days); >0 = late, <0 = early; NaN while unpaid"),
    ("paid_late_flag", "1 if delay_days > 0, 0 if paid on time/early, NaN if unpaid/unknown"),
    ("is_paid", "1 = paid, 0 = open, NaN = unknown (outcome not available)"),
    ("amount_capped", "amount winsorised at train per-currency percentile"),
    ("amount_outlier_flag", "1 if amount was above the cap"),
    ("due_date_imputed", "1 if due_date was imputed from median train terms"),
]


def clean(train_path, test_path):
    log = CleaningLog()

    tr, te = read_any(train_path), read_any(test_path)
    tr["split"], te["split"] = "train", "test"
    df = pd.concat([tr, te], ignore_index=True, sort=False)
    print(f"Raw shapes -> train {tr.shape}, test {te.shape}, combined {df.shape}")

    mapping = build_mapping(set(df.columns))
    print("Column mapping:")
    for k, v in mapping.items():
        print(f"   {k:<15} <- {v}")
    dictionary = build_data_dictionary(df, mapping)

    for field, col in mapping.items():
        if col and field in df.columns and col != field:
            df = df.rename(columns={field: f"{field}_orig"})
    df = df.rename(columns={v: k for k, v in mapping.items() if v})

    for req in ("invoice_id", "invoice_date", "amount"):
        if req not in df.columns:
            raise SystemExit(f"Required field '{req}' not mapped. Set MANUAL_MAP.")

    for c in ("invoice_id", "customer", "currency", "country", "payment_status"):
        if c in df.columns:
            before = df[c].copy()
            df[c] = clean_text(df[c])
            if c in ("currency", "customer", "invoice_id"):
                df[c] = df[c].str.upper()
            elif c == "country":
                df[c] = df[c].str.upper().replace(COUNTRY_FIX).str.title()
            elif c == "payment_status":
                df[c] = df[c].str.title()
            log.add(f"Inconsistent whitespace/case in '{c}'", "strip + standardise case",
                    "Same entity must have one spelling for grouping/joins",
                    (before.fillna("§") != df[c].fillna("§")).sum())

    for c in ("invoice_date", "due_date", "payment_date"):
        if c in df.columns:
            raw_nonnull = df[c].notna()
            df[c] = parse_dates(df[c])
            bad = raw_nonnull & df[c].isna()
            log.add(f"Unparseable values in '{c}'", "coerced to NaT (kept row)",
                    "Cannot be trusted; handled by missing-value rules", bad.sum())

    raw_amt_nonnull = df["amount"].notna()
    df["amount"] = parse_amount(df["amount"])
    log.add("Non-numeric amount strings", "parsed to float / NaN",
            "Strip symbols & thousands separators", (raw_amt_nonnull & df["amount"].isna()).sum())

    n = df.duplicated().sum()
    df = df.drop_duplicates()
    log.add("Exact duplicate rows (within split)", "dropped", "Identical records add no information", n)

    df["_nn"] = df.notna().sum(axis=1)
    df = df.sort_values("_nn", ascending=False, kind="stable")
    dup_id = df.duplicated(subset=["split", "invoice_id"], keep="first") & df["invoice_id"].notna()
    log.add("Duplicate invoice_id with different content", "kept most complete row",
            "One row per invoice is required", dup_id.sum())
    df = df[~dup_id].drop(columns="_nn").sort_index().reset_index(drop=True)

    overlap = set(df.loc[df.split == "train", "invoice_id"].dropna()) & \
              set(df.loc[df.split == "test", "invoice_id"].dropna())
    log.add("invoice_id present in BOTH train and test", "flagged only (check split)",
            "Possible leakage between splits", len(overlap))
    df["id_in_both_splits"] = df["invoice_id"].isin(overlap).astype(int)

    train = df["split"] == "train"

    crit = df["invoice_id"].isna() | df["invoice_date"].isna() | df["amount"].isna()
    log.add("Missing invoice_id / invoice_date / amount", "dropped",
            "Row unusable for any downstream feature or target", crit.sum())
    df = df[~crit].reset_index(drop=True)
    train = df["split"] == "train"

    for c in ("customer", "currency", "country"):
        if c in df.columns:
            m = df[c].isna()
            df[c] = df[c].fillna("UNKNOWN")
            log.add(f"Missing '{c}'", "imputed 'UNKNOWN'", "Keeps row; no guessing of categories", m.sum())

    if "due_date" in df.columns:
        terms = (df["due_date"] - df["invoice_date"]).dt.days
        median_terms = terms[train & (terms >= 0)].median()   # TRAIN ONLY
        miss = df["due_date"].isna()
        df["due_date_imputed"] = miss.astype(int)
        df.loc[miss, "due_date"] = df.loc[miss, "invoice_date"] + pd.to_timedelta(median_terms, unit="D")
        log.add("Missing due_date", f"imputed invoice_date + {median_terms:.0f}d (train median) & flagged",
                "Terms are fairly standard; flag preserves information", miss.sum())
    else:
        print("WARNING: no due_date column -> discuss alternative (e.g. terms column) with mentor.")

    bad_amt = df["amount"] <= 0
    log.add("Zero / negative amounts", "dropped" if DROP_NONPOSITIVE_AMOUNT else "flagged",
            "Credit notes are not payable invoices", bad_amt.sum())
    if DROP_NONPOSITIVE_AMOUNT:
        df = df[~bad_amt]

    bad = pd.Series(False, index=df.index)
    if "due_date" in df.columns:
        b1 = df["due_date"] < df["invoice_date"]
        log.add("due_date before invoice_date", "dropped" if DROP_IMPOSSIBLE_DATES else "flagged",
                "Impossible chronology", b1.sum())
        bad |= b1
    if "payment_date" in df.columns:
        b2 = df["payment_date"] < df["invoice_date"]
        log.add("payment_date before invoice_date", "dropped" if DROP_IMPOSSIBLE_DATES else "flagged",
                "Impossible chronology", b2.sum())
        bad |= b2
    if DROP_IMPOSSIBLE_DATES:
        df = df[~bad]
    df = df.reset_index(drop=True)
    train = df["split"] == "train"

    gcap = df.loc[train, "amount"].quantile(CAP_PERCENTILE)
    caps = {}
    if "currency" in df.columns:
        for cur, g in df[train].groupby("currency"):
            caps[cur] = g["amount"].quantile(CAP_PERCENTILE) if len(g) >= MIN_GROUP_FOR_CAP else gcap
        cap_series = df["currency"].map(caps).fillna(gcap)
    else:
        cap_series = pd.Series(gcap, index=df.index)
    df["amount_capped"] = np.minimum(df["amount"], cap_series)
    df["amount_outlier_flag"] = (df["amount"] > cap_series).astype(int)
    log.add(f"Extreme amounts > p{CAP_PERCENTILE*100:.1f} (per currency, train)", "capped + flagged",
            "Limit influence on models; original `amount` kept", df["amount_outlier_flag"].sum())

    has_pay_date = "payment_date" in df.columns
    has_status = "payment_status" in df.columns
    if not has_pay_date:
        print("WARNING: no payment_date -> using payment_status fallback. Agree target definition with mentor.")

    paid = pd.Series(np.nan, index=df.index)
    if has_pay_date:
        paid = df["payment_date"].notna().astype(float)
    if has_status:
        s_paid, s_late = status_flags(df["payment_status"])
        status_known = df["payment_status"].notna()
        if has_pay_date:
            conflict = df["payment_date"].notna() & status_known & ~s_paid
            log.add("payment_date present but status says unpaid", "flagged only",
                    "Source inconsistency to discuss with mentor", conflict.sum())
            paid = paid.where(paid == 1, s_paid.astype(float).where(status_known, paid))
        else:
            paid = s_paid.astype(float).where(status_known)
    df["is_paid"] = paid

    for sp in df["split"].unique():
        m = df["split"] == sp
        no_pay = (not has_pay_date) or df.loc[m, "payment_date"].isna().all()
        no_stat = (not has_status) or df.loc[m, "payment_status"].isna().all()
        if no_pay and no_stat:
            df.loc[m, "is_paid"] = np.nan
            print(f"NOTE: split '{sp}' has no payment outcome -> is_paid set to NaN (unknown)")

    if "due_date" in df.columns:
        df["payment_terms_days"] = (df["due_date"] - df["invoice_date"]).dt.days
    if has_pay_date:
        df["days_to_pay"] = (df["payment_date"] - df["invoice_date"]).dt.days
        if "due_date" in df.columns:
            df["delay_days"] = (df["payment_date"] - df["due_date"]).dt.days
            df["paid_late_flag"] = np.where(df["delay_days"].notna(),
                                            (df["delay_days"] > 0).astype(float), np.nan)
    elif has_status:
        df["paid_late_flag"] = np.where(df["is_paid"] == 1, s_late.astype(float), np.nan)
    log.add("Derived: days_to_pay, delay_days, paid_late_flag, is_paid", "created",
            "Targets/base columns; NaN (not 'today') for unpaid to avoid leakage", len(df))

    assert not df.duplicated(subset=["split", "invoice_id"]).any(), "duplicate invoice ids remain"
    if "days_to_pay" in df.columns:
        assert (df["days_to_pay"].dropna() >= 0).all(), "negative days_to_pay remain"
    print(f"\nFinal shape: {df.shape}")

    dictionary = pd.concat([
        dictionary,
        pd.DataFrame([{"column": c, "dtype": str(df[c].dtype) if c in df else "", "non_null": "",
                       "missing": "", "n_unique": "", "example": "", "project_field": "DERIVED",
                       "meaning": m} for c, m in DERIVED_DICT if c in df.columns] +
                     [{"column": c, "dtype": str(df[c].dtype), "non_null": "", "missing": "",
                       "n_unique": "", "example": "", "project_field": "DERIVED",
                       "meaning": m} for c, m in [
                         ("split", "train / test origin of the row"),
                         ("id_in_both_splits", "1 if invoice_id appears in both train and test")]]),
    ], ignore_index=True)
    return df, dictionary, mapping, log.to_df()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--test", required=True)
    ap.add_argument("--out", default="outputs")
    args = ap.parse_args()

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    df, dictionary, mapping, log = clean(args.train, args.test)

    df.to_csv(out / "invoices_clean.csv", index=False)
    try:
        df.to_parquet(out / "invoices_clean.parquet", index=False)
    except Exception as e:  # pyarrow missing
        print("Parquet skipped:", e)
    dictionary.to_csv(out / "data_dictionary.csv", index=False)
    log.to_csv(out / "cleaning_log.csv", index=False)
    (out / "column_mapping.json").write_text(json.dumps(mapping, indent=2))
    print(f"\nSaved deliverables to {out.resolve()}")


if __name__ == "__main__":
    main()
