"""
Tax savings recommendations engine.
Compares current-year data against prior-year filings and IRS limits
to generate specific, dollar-quantified money-saving actions.
"""

# 2024 IRS limits used for savings calculations
IRS_LIMITS_2024 = {
    "401k_limit":            23_000,
    "403b_limit":            23_000,
    "ira_limit":             7_000,
    "ira_catchup_50plus":    8_000,
    "hsa_self_limit":        4_150,
    "hsa_family_limit":      8_300,
    "hsa_catchup_55plus":      1_000,
    "fsa_dep_care_limit":    5_000,
    "se_tax_rate":           0.1413,   # 15.3% * 92.35%
    "standard_single":       14_600,
    "standard_mfj":          29_200,
    "standard_hoh":          21_900,
    "salt_cap":              10_000,
    "child_tax_credit":      2_000,
    "ctc_phaseout_mfj":      400_000,
    "ctc_phaseout_single":   200_000,
    "qbi_deduction_rate":    0.20,     # 20% of qualified business income
    "meal_deduction_rate":   0.50,
    "mileage_rate":          0.67,
    "scorp_savings_rate":    0.1413,   # SE tax saved by electing S-Corp
}

BRACKETS_MFJ_2024 = [
    (23_200,  0.10),
    (94_300,  0.12),
    (201_050, 0.22),
    (383_900, 0.24),
    (487_450, 0.32),
    (731_200, 0.35),
    (float("inf"), 0.37),
]

BRACKETS_SINGLE_2024 = [
    (11_600,  0.10),
    (47_150,  0.12),
    (100_525, 0.22),
    (191_950, 0.24),
    (243_725, 0.32),
    (365_600, 0.35),
    (float("inf"), 0.37),
]


def marginal_rate(taxable_income: float, filing_status: str = "MFJ") -> float:
    brackets = BRACKETS_MFJ_2024 if filing_status == "MFJ" else BRACKETS_SINGLE_2024
    prev = 0
    for ceiling, rate in brackets:
        if taxable_income <= ceiling:
            return rate
        prev = ceiling
    return 0.37


def calc_savings(deduction_amount: float, taxable_income: float,
                 filing_status: str = "MFJ") -> float:
    """Estimate tax saved by increasing deductions by deduction_amount."""
    rate = marginal_rate(taxable_income, filing_status)
    return round(deduction_amount * rate, 0)


def build_savings_context(current: dict, prior: dict) -> str:
    """
    Build a rich text context for the AI to generate recommendations.
    current / prior are dicts with keys matching the tax profile.
    """
    lines = [
        "=== YEAR-OVER-YEAR TAX COMPARISON FOR SAVINGS RECOMMENDATIONS ===",
        "",
        "--- CURRENT YEAR (2024) ---",
        f"Filing Status:              {current.get('filing_status', 'MFJ')}",
        f"Total Gross Income:         ${current.get('gross_income', 0):,.2f}",
        f"W-2 Wages:                  ${current.get('w2_wages', 0):,.2f}",
        f"Self-Employment Income:     ${current.get('se_income', 0):,.2f}",
        f"Other Income:               ${current.get('other_income', 0):,.2f}",
        f"Estimated AGI:              ${current.get('agi', 0):,.2f}",
        f"Taxable Income:             ${current.get('taxable_income', 0):,.2f}",
        f"Federal Tax Owed:           ${current.get('federal_tax', 0):,.2f}",
        f"Effective Tax Rate:         {current.get('effective_rate', 0):.1f}%",
        f"401k Contributed:           ${current.get('retirement_401k', 0):,.2f}  (limit ${IRS_LIMITS_2024['401k_limit']:,})",
        f"HSA Contributed:            ${current.get('hsa_total', 0):,.2f}  (limit ${IRS_LIMITS_2024['hsa_family_limit']:,} family)",
        f"Charitable Donations:       ${current.get('charitable', 0):,.2f}",
        f"Deduction Taken:            {current.get('deduction_type', 'Standard')} (${current.get('deduction_amount', 0):,.2f})",
        "",
        "--- PRIOR YEAR (2023) ---",
        f"Total Gross Income:         ${prior.get('gross_income', 0):,.2f}",
        f"AGI:                        ${prior.get('agi', 0):,.2f}",
        f"Federal Tax Paid:           ${prior.get('federal_tax', 0):,.2f}",
        f"Effective Tax Rate:         {prior.get('effective_rate', 0):.1f}%",
        f"401k Contributed:           ${prior.get('retirement_401k', 0):,.2f}",
        f"Refund / Balance Due:       ${prior.get('refund', 0):,.2f}",
        "",
        "--- YEAR-OVER-YEAR CHANGES ---",
        f"Income Change:              ${current.get('gross_income',0) - prior.get('gross_income',0):+,.2f}",
        f"AGI Change:                 ${current.get('agi',0) - prior.get('agi',0):+,.2f}",
        f"Tax Change:                 ${current.get('federal_tax',0) - prior.get('federal_tax',0):+,.2f}",
        f"Effective Rate Change:      {current.get('effective_rate',0) - prior.get('effective_rate',0):+.1f}%",
        "",
        "--- SAVINGS OPPORTUNITIES IDENTIFIED ---",
    ]

    tips = []
    agi = current.get("agi", 0)
    taxable = current.get("taxable_income", 0)
    filing = current.get("filing_status", "MFJ")
    marginal = marginal_rate(taxable, filing)

    # 401k gap
    contrib_401k = current.get("retirement_401k", 0)
    gap_401k = IRS_LIMITS_2024["401k_limit"] - contrib_401k
    if gap_401k > 0:
        save = calc_savings(gap_401k, taxable, filing)
        tips.append(f"401k: Room to contribute ${gap_401k:,.0f} more → saves ~${save:,.0f} in federal tax")

    # HSA gap
    hsa_total = current.get("hsa_total", 0)
    hsa_limit = IRS_LIMITS_2024["hsa_family_limit"] if filing == "MFJ" else IRS_LIMITS_2024["hsa_self_limit"]
    gap_hsa = hsa_limit - hsa_total
    if gap_hsa > 0:
        save = calc_savings(gap_hsa, taxable, filing)
        tips.append(f"HSA: Room to contribute ${gap_hsa:,.0f} more → saves ~${save:,.0f} in federal tax")

    # S-Corp election for SE income
    se_income = current.get("se_income", 0)
    if se_income > 40_000:
        reasonable_salary = se_income * 0.60
        se_savings = (se_income - reasonable_salary) * IRS_LIMITS_2024["scorp_savings_rate"]
        tips.append(f"S-Corp election: Pay ${reasonable_salary:,.0f} salary, distribute rest → saves ~${se_savings:,.0f} in SE tax")

    # QBI deduction (20% of net SE income if under thresholds)
    qbi_threshold_mfj = 383_900
    qbi_threshold_single = 191_950
    threshold = qbi_threshold_mfj if filing == "MFJ" else qbi_threshold_single
    if se_income > 0 and agi < threshold:
        qbi_deduction = se_income * IRS_LIMITS_2024["qbi_deduction_rate"]
        save = calc_savings(qbi_deduction, taxable, filing)
        tips.append(f"QBI Deduction (Sec 199A): 20% of ${se_income:,.0f} SE income = ${qbi_deduction:,.0f} deduction → saves ~${save:,.0f}")

    # Charitable bunching
    charitable = current.get("charitable", 0)
    std_ded = IRS_LIMITS_2024["standard_mfj"] if filing == "MFJ" else IRS_LIMITS_2024["standard_single"]
    if charitable > 0 and charitable < std_ded * 0.3:
        bunched = charitable * 2
        itemized_approx = current.get("itemized_total", 0) + bunched - charitable
        if itemized_approx > std_ded:
            extra = itemized_approx - std_ded
            save = calc_savings(extra, taxable, filing)
            tips.append(f"Donate bunching: Double 2 years of donations in one year (${bunched:,.0f}) → itemize and save ~${save:,.0f}")

    # Bracket boundary check
    bracket_boundaries = [23_200, 94_300, 201_050] if filing == "MFJ" else [11_600, 47_150, 100_525]
    for boundary in bracket_boundaries:
        gap_to_boundary = taxable - boundary
        if 0 < gap_to_boundary < 15_000:
            save = calc_savings(gap_to_boundary, taxable, filing)
            tips.append(f"Bracket edge: ${gap_to_boundary:,.0f} above {int(marginal*100-2)}% bracket boundary — "
                        f"extra deductions could save ~${save:,.0f} by dropping a bracket")
            break

    # IRA deduction (if not covered by employer plan or phased out)
    tips.append(f"Traditional IRA: Contribute up to ${IRS_LIMITS_2024['ira_limit']:,} → saves ~"
                f"${calc_savings(IRS_LIMITS_2024['ira_limit'], taxable, filing):,.0f} (income limits apply)")

    for tip in tips:
        lines.append(f"  >> {tip}")

    lines += [
        "",
        f"Marginal Tax Rate: {int(marginal*100)}%  (every $1,000 deduction saves ${marginal*1000:.0f})",
        f"2024 IRS Limits Reference: 401k ${IRS_LIMITS_2024['401k_limit']:,}  |  HSA family ${IRS_LIMITS_2024['hsa_family_limit']:,}  |  IRA ${IRS_LIMITS_2024['ira_limit']:,}",
    ]

    return "\n".join(lines)
