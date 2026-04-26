"""
Claude-powered tax advisor agent with session memory and RAG retrieval.
Collection names are passed per-call so each user session is fully isolated.
"""

import base64
import json
import os
from anthropic import Anthropic
from rag import retrieve_and_format, retrieve_and_format_prior_year, retrieve_all_user_docs, retrieve_all_prior_docs

client = Anthropic(api_key=os.environ["ANTHROPIC_API_KEY"])
MODEL = "claude-haiku-4-5-20251001"

SAVINGS_PROMPT = """You are Aria, a senior US tax strategist and CPA with 25 years of experience helping individuals and small business owners legally minimize their tax burden.

Your ONLY job in this mode is to answer: "What specific actions should I take to pay less tax?"

CRITICAL INSTRUCTIONS — READ BEFORE RESPONDING:
1. The <current_year_data> block contains the user's ACTUAL financial documents — W-2s, 1099s, bank transactions, tax profile, deductions. USE EVERY NUMBER IN IT.
2. NEVER say data is "not provided" or "missing". If a figure appears anywhere in the context, it IS provided.
3. If something is truly absent, make a reasonable IRS-standard assumption and label it "(assumed)" — do NOT refuse to produce the report.
4. You MUST produce all sections of the SAVINGS RECOMMENDATIONS REPORT regardless of what is or isn't in context.
5. If the context contains W-2 wages, 1099 income, deductions, retirement contributions — cite the exact figures from it.

You have access to:
1. The user's CURRENT year financial data (transactions, W-2s, 1099s)
2. The user's PRIOR year tax filings for comparison
3. IRS regulations and 2024 tax limits
4. A pre-calculated savings opportunities analysis

## OUTPUT FORMAT — SAVINGS RECOMMENDATIONS REPORT:

**TAX SAVINGS SUMMARY**
- Current Year Estimated Tax: $X
- Prior Year Tax Paid: $X
- Year-over-Year Change: +/- $X (+/-X%)
- Potential Annual Savings Identified: $X

---

**RECOMMENDATION 1: [Action Title]**
- What to do: [Specific action]
- Why: [IRS rule / tax law basis]
- Estimated Annual Savings: $X
- How to implement: [Exact steps]
- Deadline: [When to act by]

**RECOMMENDATION 2: [Action Title]**
[same format...]

[Continue for all recommendations, ranked by dollar savings]

---

**FILING STRATEGY COMPARISON**
| Option | Taxable Income | Est. Tax | Savings vs Current |
|---|---|---|---|
| Current approach | $X | $X | baseline |
| [Alternative 1] | $X | $X | save $X |
| [Alternative 2] | $X | $X | save $X |

---

**YEAR-OVER-YEAR ANALYSIS**
- What changed and why your tax went up or down
- What you did right vs last year
- What you should do differently next year

---

**TOTAL POTENTIAL SAVINGS: $X/year**

## RULES:
- Every recommendation MUST have a specific dollar savings estimate
- Rank recommendations by highest dollar savings first
- Cite the IRS rule, publication, or tax code section for each recommendation
- Be specific: "Increase 401k by $8,000" not "consider contributing more"
- Include implementation steps and deadlines
- NEVER produce a "SITUATION ASSESSMENT" or "CRITICAL LIMITATION NOTICE" — go straight to the SAVINGS RECOMMENDATIONS REPORT
- NEVER say you need more information — use what is in <current_year_data> and produce the full report
- If you can see ANY financial figures in the context, use them. A single W-2 amount is enough to start."""

SYSTEM_PROMPT = """You are Aria, a senior US tax attorney and CPA with 25 years of experience in personal finance, small business tax compliance, and IRS audit defense.

You have access to two sources of information:
1. IRS regulations, publications, and tax law (retrieved from the knowledge base)
2. The user's actual financial data — bank transactions, W-2s, 1099s, and other documents

## YOUR CORE TASKS:

### When bank/financial transactions are present:
- Identify ALL income sources (deposits, transfers in, payroll, freelance payments)
- Find ALL potentially deductible expenses by category (business, medical, charitable, etc.)
- Flag transactions that may indicate unreported income or missing deductions
- Calculate estimated taxable income based on transaction patterns
- Identify compliance risks (large cash transactions, unclassified income, etc.)

### When tax documents are present:
- Verify W-2 withholding is adequate for the income level
- Check 1099 income matches transaction deposits
- Validate retirement/HSA contribution limits
- Cross-reference deductions claimed vs IRS limits

### Always:
- Cross-reference everything against the retrieved IRS regulations
- Apply actual IRS thresholds, limits, and rules from the knowledge base
- Never fabricate figures or rules not in the retrieved context

## FULL ANALYSIS OUTPUT FORMAT:

**COMPLIANCE SCORE: [0-100]/100**
[One sentence explanation]

**INCOME FOUND:**
- W-2 / Payroll: $X
- 1099 / Self-employment: $X
- Bank deposits (other income): $X
- Total Gross Income: $X

**DEDUCTIBLE EXPENSES FOUND:**
| Category | Amount | IRS Rule |
|---|---|---|
| [category] | $X | [cite rule] |
Total Deductions: $X

**ESTIMATED TAX POSITION:**
- Estimated AGI: $X
- Standard / Itemized Deduction: $X
- Estimated Taxable Income: $X
- Estimated Federal Tax Owed: $X
- Tax Already Withheld: $X
- Estimated Balance Due / Refund: $X

**KEY FINDINGS:**
- [Finding with dollar amount and IRS reference]

**COMPLIANCE RISKS & FLAGS:**
- [CRITICAL/HIGH/MEDIUM/LOW] [Risk description]

**RECOMMENDATIONS:**
1. [Highest priority — action + estimated tax savings]
2. [Next action]

## RULES:
- Always produce a COMPLETE analysis using whatever is in context — never say you need more documents
- If a specific figure is not explicitly stated, derive it from the transactions and documents available
- Make reasonable IRS-standard assumptions for anything not explicitly provided and state them clearly
- Meals are only 50% deductible per IRS rules
- SE tax is 15.3% on 92.35% of net self-employment income
- Always recommend a licensed CPA for final filing decisions
- NEVER ask the user to upload more documents — analyze what is there and produce the full report"""


def build_user_message(user_input: str, user_col: str, include_rag: bool = True) -> str:
    if not include_rag:
        return user_input
    context = retrieve_and_format(user_input, user_col=user_col)
    return f"""<retrieved_context>
{context}
</retrieved_context>

<user_question>
{user_input}
</user_question>"""


def chat(history: list[dict], user_input: str,
         user_col: str = "user_documents",
         include_rag: bool = True) -> tuple[str, list[dict]]:
    message_content = build_user_message(user_input, user_col=user_col, include_rag=include_rag)
    updated_history = history + [{"role": "user", "content": message_content}]
    response = client.messages.create(
        model=MODEL,
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=updated_history,
    )
    reply = response.content[0].text
    updated_history = updated_history + [{"role": "assistant", "content": reply}]
    return reply, updated_history


def analyze_documents(history: list[dict],
                      user_col: str = "user_documents") -> tuple[str, list[dict]]:
    full_context = retrieve_all_user_docs(user_col)
    prompt = f"""<uploaded_documents>
{full_context}
</uploaded_documents>

Every figure above is from my actual uploaded financial documents. Use all of it now.
Produce the COMPLETE tax compliance analysis:
1. COMPLIANCE SCORE (0-100) with one-line reason
2. INCOME FOUND — every source with dollar amount
3. DEDUCTIBLE EXPENSES — table: category | amount | IRS rule
4. ESTIMATED TAX POSITION — AGI, taxable income, tax owed, withheld, balance due/refund
5. KEY FINDINGS — specific dollar findings
6. COMPLIANCE RISKS — CRITICAL/HIGH/MEDIUM/LOW each
7. RECOMMENDATIONS — numbered, with estimated tax savings per action

Do not ask for more documents. Use all available data and produce the full report now."""
    return chat(history, prompt, user_col=user_col, include_rag=False)


def savings_recommendations(history: list[dict],
                            user_col: str = "user_documents",
                            prior_col: str = "prior_year_returns") -> tuple[str, list[dict]]:
    # Use comprehensive multi-query retrieval to get all user financial data
    current_context = retrieve_all_user_docs(user_col)
    prior_context   = retrieve_all_prior_docs(prior_col)

    prompt = f"""The following blocks contain the user's ACTUAL uploaded financial documents.
Every figure you see is real data — W-2 wages, 1099 income, deductions, contributions, etc.
USE ALL OF IT. Do not say anything is "not provided" — it IS in the data below.

<current_year_data>
{current_context}
</current_year_data>

<prior_year_data>
{prior_context}
</prior_year_data>

Using every figure in the data above, produce the complete TAX SAVINGS RECOMMENDATIONS REPORT now:
1. What specific dollar-amount actions should I take to reduce my tax bill?
2. What deductions or credits am I under-utilizing?
3. How does my current year compare to last year — more or less tax, and why?
4. What should I do differently this filing vs last year?
5. What changes for next year's planning?

For every recommendation give: exact dollar savings, IRS rule, implementation steps, deadline.
Start the report immediately — no preamble, no situation assessment, no "I cannot"."""

    updated_history = history + [{"role": "user", "content": prompt}]
    response = client.messages.create(
        model=MODEL,
        max_tokens=4000,
        system=SAVINGS_PROMPT,
        messages=updated_history,
    )
    reply = response.content[0].text
    updated_history = updated_history + [{"role": "assistant", "content": reply}]
    return reply, updated_history


def chat_stream(history: list[dict], user_input: str,
                user_col: str = "user_documents",
                include_rag: bool = True):
    """Streaming variant of chat(). Yields text tokens. Caller appends history."""
    message_content = build_user_message(user_input, user_col=user_col, include_rag=include_rag)
    updated_history = history + [{"role": "user", "content": message_content}]
    with client.messages.stream(
        model=MODEL,
        max_tokens=3000,
        system=SYSTEM_PROMPT,
        messages=updated_history,
    ) as stream:
        for text in stream.text_stream:
            yield text


def analyze_transactions(history: list[dict],
                         user_col: str = "user_documents") -> tuple[str, list[dict]]:
    prompt = (
        "Analyze ALL bank and financial transactions uploaded. "
        "Do NOT ask for more information — use what is here.\n\n"
        "Produce:\n"
        "1. COMPLIANCE SCORE\n"
        "2. Every income source found with amount\n"
        "3. Every deductible expense by category with amount and IRS rule\n"
        "4. Transactions flagged as unreported income or missed deductions\n"
        "5. Estimated tax impact of all findings\n"
        "6. Specific recommendations with dollar savings\n\n"
        "Give the complete analysis now."
    )
    return chat(history, prompt, user_col=user_col, include_rag=True)


def extract_receipt(image_bytes: bytes, media_type: str = "image/jpeg") -> dict:
    """
    Extract structured expense data from a receipt/invoice image using Claude Vision.
    Returns dict: {merchant, date, amount, category, is_deductible, notes}
    """
    b64 = base64.standard_b64encode(image_bytes).decode("utf-8")
    response = client.messages.create(
        model=MODEL,
        max_tokens=400,
        messages=[{
            "role": "user",
            "content": [
                {
                    "type": "image",
                    "source": {"type": "base64", "media_type": media_type, "data": b64},
                },
                {
                    "type": "text",
                    "text": (
                        "Extract data from this receipt or invoice. "
                        "Return ONLY valid JSON with these exact keys:\n"
                        '{"merchant":"business name","date":"YYYY-MM-DD or empty string",'
                        '"amount":0.00,"category":"meals|office|travel|utilities|software|medical|other",'
                        '"is_deductible":true,"notes":"one-line description"}\n\n'
                        "Rules: amount must be a float. "
                        "is_deductible=true for business or medical expenses, false for personal. "
                        "If a field is unclear, use an empty string or 0."
                    ),
                },
            ],
        }]
    )
    text = response.content[0].text.strip()
    if "```" in text:
        parts = text.split("```")
        text = parts[1] if len(parts) > 1 else text
        if text.lstrip().startswith("json"):
            text = text.lstrip()[4:].strip()
    try:
        data = json.loads(text)
        data["amount"] = float(data.get("amount", 0) or 0)
        data["is_deductible"] = bool(data.get("is_deductible", False))
        return data
    except Exception:
        return {
            "merchant": "Unknown", "date": "", "amount": 0.0,
            "category": "other", "is_deductible": False,
            "notes": text[:120],
        }
