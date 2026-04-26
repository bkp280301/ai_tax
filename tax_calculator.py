"""
Real US federal + state tax computations via tenforty (wraps Open Tax Solver).
Gracefully disabled if tenforty is not installed (Windows dev machines).
"""

try:
    from tenforty import evaluate_return
    TENFORTY_AVAILABLE = True
except (ImportError, Exception):
    TENFORTY_AVAILABLE = False

FILING_STATUSES = ["Single", "Married/Joint", "Head_of_House", "Married/Sep"]
FILING_STATUS_LABELS = {
    "Single":        "Single",
    "Married/Joint": "Married Filing Jointly",
    "Head_of_House": "Head of Household",
    "Married/Sep":   "Married Filing Separately",
}

US_STATES = [
    "AL","AK","AZ","AR","CA","CO","CT","DE","FL","GA","HI","ID","IL","IN","IA",
    "KS","KY","LA","ME","MD","MA","MI","MN","MS","MO","MT","NE","NV","NH","NJ",
    "NM","NY","NC","ND","OH","OK","OR","PA","RI","SC","SD","TN","TX","UT","VT",
    "VA","WA","WV","WI","WY",
]


def _run(w2_income, se_income, filing_status, state, num_dependents,
         itemized_deductions=0.0):
    """Run a single tenforty evaluation. Returns model dict or None on error."""
    if not TENFORTY_AVAILABLE:
        return None
    try:
        result = evaluate_return(
            year=2024,
            w2_income=float(max(0, w2_income)),
            self_employment_income=float(max(0, se_income)),
            filing_status=filing_status,
            state=state,
            num_dependents=int(num_dependents),
            itemized_deductions=float(max(0, itemized_deductions)),
            standard_or_itemized="Itemized" if itemized_deductions > 0 else "Standard",
        )
        return result.model_dump()
    except Exception as exc:
        return {"_error": str(exc)}


def compute_scenarios(w2_income: float, se_income: float = 0,
                      filing_status: str = "Single", state: str = "TX",
                      num_dependents: int = 0, age: int = 40) -> list[dict] | None:
    """
    Run 6 what-if tax scenarios and return comparison list.
    Returns None if tenforty is unavailable.

    Each dict keys:
        scenario, pre_tax_deductions, federal_agi, taxable_income,
        federal_tax, se_tax, state_tax, total_tax,
        effective_rate, state_rate, bracket, savings_vs_baseline
    """
    if not TENFORTY_AVAILABLE:
        return None

    max_401k = 30_500 if age >= 50 else 23_000
    max_hsa  = 8_300  if num_dependents > 0 else 4_150
    max_ira  = 8_000  if age >= 50 else 7_000

    # (label, w2_reduction)  — all reduce w2 income (pre-tax or above-the-line)
    scenarios = [
        ("Current (No Changes)",                            0),
        (f"+ Max 401(k)  −${max_401k:,}",                  max_401k),
        (f"+ Max HSA  −${max_hsa:,}",                      max_hsa),
        (f"+ Max IRA  −${max_ira:,}",                      max_ira),
        (f"+ Max 401(k) + HSA  −${max_401k + max_hsa:,}",  max_401k + max_hsa),
        (f"+ Max Everything  −${max_401k+max_hsa+max_ira:,}", max_401k + max_hsa + max_ira),
    ]

    results = []
    baseline_total = None

    for label, reduction in scenarios:
        r = _run(
            w2_income=w2_income - reduction,
            se_income=se_income,
            filing_status=filing_status,
            state=state,
            num_dependents=num_dependents,
        )
        if r is None or "_error" in r:
            continue

        total = r.get("total_tax", 0) or 0
        if baseline_total is None:
            baseline_total = total

        results.append({
            "scenario":           label,
            "pre_tax_deductions": reduction,
            "federal_agi":        r.get("federal_adjusted_gross_income", 0) or 0,
            "taxable_income":     r.get("federal_taxable_income", 0) or 0,
            "federal_tax":        r.get("federal_income_tax", 0) or 0,
            "se_tax":             r.get("federal_se_tax", 0) or 0,
            "state_tax":          r.get("state_total_tax", 0) or 0,
            "total_tax":          total,
            "effective_rate":     r.get("federal_effective_tax_rate", 0) or 0,
            "state_rate":         r.get("state_effective_tax_rate", 0) or 0,
            "bracket":            r.get("federal_tax_bracket", 0) or 0,
            "savings_vs_baseline": round((baseline_total - total), 2),
        })

    return results
