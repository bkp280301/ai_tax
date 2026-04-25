"""
Parses bank/financial CSV exports into structured transaction records.
Auto-categorizes transactions and flags potential tax deductions.
"""

import re
import pandas as pd
from pathlib import Path
from io import StringIO

# ── Category keyword maps ──────────────────────────────────────────────────────
INCOME_KEYWORDS = [
    "payroll", "direct dep", "salary", "wages", "ach deposit", "zelle in",
    "venmo in", "transfer in", "refund", "reimbursement", "commission",
    "freelance", "consulting", "payment received", "invoice",
]

DEDUCTIBLE_BUSINESS = {
    "Office Supplies": ["staples", "office depot", "amazon business", "officemax", "paper", "printer"],
    "Software/SaaS": ["adobe", "github", "aws", "google cloud", "microsoft", "zoom", "slack",
                      "notion", "dropbox", "quickbooks", "figma", "canva"],
    "Phone/Internet": ["at&t", "verizon", "t-mobile", "comcast", "spectrum", "xfinity", "cox"],
    "Travel (Business)": ["uber", "lyft", "airline", "delta", "united", "american air",
                          "southwest", "marriott", "hilton", "hyatt", "airbnb", "hotel"],
    "Meals (50% deductible)": ["restaurant", "grubhub", "doordash", "ubereats", "seamless",
                                "chipotle", "starbucks", "coffee", "lunch", "dinner", "cafe"],
    "Professional Services": ["attorney", "accountant", "cpa", "legal", "consulting fee"],
    "Education/Training": ["udemy", "coursera", "linkedin learning", "pluralsight", "training",
                           "conference", "seminar", "workshop", "book", "amazon kindle"],
    "Advertising/Marketing": ["google ads", "facebook ads", "meta ads", "mailchimp",
                               "hubspot", "advertising", "marketing"],
    "Shipping": ["fedex", "ups", "usps", "shipping", "postage"],
    "Equipment": ["apple store", "best buy", "b&h photo", "newegg", "dell", "lenovo", "equipment"],
    "Rent/Utilities (Office)": ["office rent", "coworking", "wework", "regus"],
}

DEDUCTIBLE_PERSONAL = {
    "Charitable Donations": ["goodwill", "salvation army", "red cross", "charity", "donation",
                              "church", "tithe", "nonprofit", "npo"],
    "Medical/Dental": ["cvs", "walgreens", "rite aid", "pharmacy", "doctor", "hospital",
                        "dental", "vision", "medical", "clinic", "urgent care", "lab corp"],
    "Mortgage Interest": ["mortgage", "home loan", "escrow payment"],
    "Student Loan": ["sallie mae", "navient", "fed loan", "student loan", "mohela"],
    "Childcare": ["daycare", "child care", "babysitter", "preschool", "after school"],
    "Education (529)": ["529 plan", "education savings", "college savings"],
}

TRANSFER_KEYWORDS = ["transfer", "zelle", "venmo", "cashapp", "cash app", "payment to",
                      "credit card payment", "loan payment", "bill payment"]


def _detect_columns(df: pd.DataFrame) -> dict:
    """Try to map common bank CSV column names to standard ones."""
    cols = {c.lower().strip(): c for c in df.columns}
    mapping = {}

    for key, candidates in {
        "date": ["date", "transaction date", "trans date", "posted date", "posting date"],
        "description": ["description", "memo", "payee", "transaction description", "details", "name"],
        "amount": ["amount", "transaction amount", "debit", "credit"],
        "debit": ["debit", "withdrawal", "withdrawals", "debit amount"],
        "credit": ["credit", "deposit", "deposits", "credit amount"],
    }.items():
        for c in candidates:
            if c in cols:
                mapping[key] = cols[c]
                break

    return mapping


def _categorize(desc: str) -> tuple[str, str, bool]:
    """
    Returns (category, type, is_deductible).
    type: 'income' | 'business_expense' | 'personal_expense' | 'transfer' | 'other'
    """
    d = desc.lower()

    for kw in TRANSFER_KEYWORDS:
        if kw in d:
            return "Transfer / Payment", "transfer", False

    for kw in INCOME_KEYWORDS:
        if kw in d:
            return "Income / Deposit", "income", False

    for cat, keywords in DEDUCTIBLE_BUSINESS.items():
        for kw in keywords:
            if kw in d:
                return cat, "business_expense", True

    for cat, keywords in DEDUCTIBLE_PERSONAL.items():
        for kw in keywords:
            if kw in d:
                return cat, "personal_expense", True

    return "Personal / Other", "other", False


def parse_transactions(file_path: str) -> pd.DataFrame:
    """
    Parse a bank CSV/Excel file into a clean transactions DataFrame.
    Columns: date, description, amount, type, category, is_deductible
    """
    path = Path(file_path)
    if path.suffix.lower() in (".xlsx", ".xls"):
        raw = pd.read_excel(file_path)
    else:
        # Try different encodings / skip metadata rows
        for enc in ("utf-8", "latin-1", "cp1252"):
            try:
                raw = pd.read_csv(file_path, encoding=enc, skip_blank_lines=True)
                if len(raw.columns) >= 2:
                    break
            except Exception:
                continue

    mapping = _detect_columns(raw)

    rows = []
    for _, row in raw.iterrows():
        # Date
        date_val = str(row.get(mapping.get("date", ""), "")).strip()

        # Description
        desc = str(row.get(mapping.get("description", ""), "")).strip()
        if not desc or desc.lower() in ("nan", ""):
            continue

        # Amount: handle separate debit/credit columns or single amount column
        amount = 0.0
        if "debit" in mapping and "credit" in mapping:
            debit = row.get(mapping["debit"], 0)
            credit = row.get(mapping["credit"], 0)
            try:
                debit = float(str(debit).replace(",", "").replace("$", "") or 0)
            except Exception:
                debit = 0.0
            try:
                credit = float(str(credit).replace(",", "").replace("$", "") or 0)
            except Exception:
                credit = 0.0
            amount = credit - debit  # positive = money in, negative = money out
        elif "amount" in mapping:
            try:
                amount = float(str(row[mapping["amount"]]).replace(",", "").replace("$", "").strip())
            except Exception:
                amount = 0.0

        category, txn_type, is_deductible = _categorize(desc)

        # Override type for obvious positive amounts in non-income categories
        if amount > 0 and txn_type not in ("income", "transfer"):
            txn_type = "income"
            category = "Income / Deposit"
            is_deductible = False

        rows.append({
            "date": date_val,
            "description": desc,
            "amount": amount,
            "type": txn_type,
            "category": category,
            "is_deductible": is_deductible,
        })

    return pd.DataFrame(rows)


def summarize_transactions(df: pd.DataFrame) -> dict:
    """Return a summary dict for reporting and RAG ingestion."""
    total_income = df[df["type"] == "income"]["amount"].sum()
    total_expenses = df[df["type"].isin(["business_expense", "personal_expense"])]["amount"].abs().sum()
    business_expenses = df[df["type"] == "business_expense"]["amount"].abs().sum()
    deductible = df[df["is_deductible"] == True]["amount"].abs().sum()

    by_category = (
        df[df["is_deductible"] == True]
        .groupby("category")["amount"]
        .apply(lambda x: x.abs().sum())
        .sort_values(ascending=False)
        .to_dict()
    )

    return {
        "total_transactions": len(df),
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "business_expenses": round(business_expenses, 2),
        "total_deductible": round(deductible, 2),
        "deductible_by_category": {k: round(v, 2) for k, v in by_category.items()},
        "transaction_types": df["type"].value_counts().to_dict(),
    }


def transactions_to_text(df: pd.DataFrame, summary: dict) -> str:
    """Convert parsed transactions into a text block suitable for RAG ingestion."""
    lines = [
        "=== BANK / FINANCIAL TRANSACTION ANALYSIS ===",
        f"Total Transactions: {summary['total_transactions']}",
        f"Total Income Deposits: ${summary['total_income']:,.2f}",
        f"Total Expenses: ${summary['total_expenses']:,.2f}",
        f"Business Expenses: ${summary['business_expenses']:,.2f}",
        f"Potentially Tax-Deductible Amount: ${summary['total_deductible']:,.2f}",
        "",
        "--- DEDUCTIBLE EXPENSES BY CATEGORY ---",
    ]
    for cat, amt in summary["deductible_by_category"].items():
        lines.append(f"  {cat}: ${amt:,.2f}")

    lines += ["", "--- TRANSACTION TYPE BREAKDOWN ---"]
    for txn_type, count in summary["transaction_types"].items():
        lines.append(f"  {txn_type}: {count} transactions")

    lines += ["", "--- INDIVIDUAL TRANSACTIONS (deductible flagged) ---"]
    for _, row in df.iterrows():
        flag = "[DEDUCTIBLE]" if row["is_deductible"] else ""
        lines.append(
            f"{row['date']} | {row['description'][:50]:<50} | "
            f"${abs(row['amount']):>10,.2f} | {row['category']} {flag}"
        )

    return "\n".join(lines)
