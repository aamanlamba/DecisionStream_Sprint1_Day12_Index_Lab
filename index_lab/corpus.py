"""
corpus.py — twenty business policy documents, and the queries we test with.

WHY THE METADATA IS THE POINT
-----------------------------
These twenty documents contain SUPERSEDED VERSIONS and OTHER JURISDICTIONS
of the same policies. That is not an accident of the sample — it is what a
real policy library looks like. Nobody deletes the 2024 wording; they file it.

So the corpus contains, for the same subject:

    Total loss threshold, UK, 2024   -> 70%
    Total loss threshold, UK, 2025   -> 65%
    Total loss threshold, UK, 2026   -> 60%      <- in force now
    Total loss threshold, IE, 2026   -> 55%      <- different country

All four are well written. All four are findable. A vector search asked
"what is the total loss threshold" will happily return whichever one it
finds most similar — and three of the four answers are wrong for a UK claim
decided today.

    No embedding model can fix this. It is not a similarity problem.
    It is a FILTER problem, and the filter comes from metadata.

That is today's lesson, and it is why metadata design happens at ingestion
time rather than being bolted on later.
"""

from __future__ import annotations

from dataclasses import dataclass, asdict


@dataclass
class PolicyDoc:
    id: str
    title: str
    policy_type: str        # motor | property | liability | conduct
    jurisdiction: str       # UK | IE | EU
    effective_from: str     # ISO date
    effective_to: str | None
    version: str
    superseded: bool
    text: str

    def meta(self) -> dict:
        d = asdict(self)
        d.pop("text")
        return d


def _d(**kw) -> PolicyDoc:
    return PolicyDoc(**kw)


DOCS: list[PolicyDoc] = [

# ---------- MOTOR · UK · three versions of the same clauses ----------
_d(id="POL-MOT-UK-2024", title="Motor Policy Wording (UK)", policy_type="motor",
   jurisdiction="UK", effective_from="2024-01-01", effective_to="2024-12-31",
   version="v1", superseded=True, text="""
CLAUSE 4.1 TOTAL LOSS
A vehicle is treated as a total loss where the estimated cost of repair
exceeds seventy per cent of its pre-accident market value.

CLAUSE 2.4 EXCESS
The policyholder pays the excess stated in the schedule. Where the driver is
under twenty-one years of age, an additional young driver excess of GBP 150
applies.

CLAUSE 3.2 REPAIR AUTHORISATION
Repairs above GBP 3,000 require authorisation from a claims handler before
work begins.
"""),

_d(id="POL-MOT-UK-2025", title="Motor Policy Wording (UK)", policy_type="motor",
   jurisdiction="UK", effective_from="2025-01-01", effective_to="2025-12-31",
   version="v2", superseded=True, text="""
CLAUSE 4.1 TOTAL LOSS
A vehicle is treated as a total loss where the estimated cost of repair
exceeds sixty-five per cent of its pre-accident market value.

CLAUSE 2.4 EXCESS
The policyholder pays the excess stated in the schedule. Where the driver is
under twenty-three years of age, an additional young driver excess of GBP 200
applies.

CLAUSE 3.2 REPAIR AUTHORISATION
Repairs above GBP 4,000 require authorisation from a claims handler before
work begins.
"""),

_d(id="POL-MOT-UK-2026", title="Motor Policy Wording (UK)", policy_type="motor",
   jurisdiction="UK", effective_from="2026-01-01", effective_to=None,
   version="v3", superseded=False, text="""
CLAUSE 4.1 TOTAL LOSS
A vehicle is treated as a total loss where the estimated cost of repair
exceeds sixty per cent of its pre-accident market value.

CLAUSE 2.4 EXCESS
The policyholder pays the excess stated in the schedule for each and every
claim. Where the driver at the time of the incident is under twenty-five
years of age, an additional young driver excess of GBP 250 applies.

CLAUSE 3.2 REPAIR AUTHORISATION
Repairs above GBP 5,000 require authorisation from a claims handler before
work begins. Repairs carried out without authorisation may be settled at the
amount the insurer would reasonably have paid an approved repairer.

CLAUSE 3.5 SUPPLEMENTARY ESTIMATES
Where additional damage is found after work has begun, a supplementary
estimate must be submitted. Supplementary work above twenty per cent of the
original estimate requires an engineer inspection before authorisation.

CLAUSE 6.2 CLAIM NOTIFICATION
A claim must be notified within thirty days of the incident.
"""),

# ---------- MOTOR · IE · same subject, different country ----------
_d(id="POL-MOT-IE-2026", title="Motor Policy Wording (Ireland)", policy_type="motor",
   jurisdiction="IE", effective_from="2026-01-01", effective_to=None,
   version="v1", superseded=False, text="""
CLAUSE 4.1 TOTAL LOSS
A vehicle is treated as a total loss where the estimated cost of repair
exceeds fifty-five per cent of its pre-accident market value.

CLAUSE 2.4 EXCESS
The policyholder pays the excess stated in the schedule. Where the driver is
under twenty-five years of age, an additional young driver excess of EUR 300
applies.

CLAUSE 6.2 CLAIM NOTIFICATION
A claim must be notified within twenty-eight days of the incident.
"""),

# ---------- CLAIMS HANDLING PROCESS · UK ----------
_d(id="PRC-CLM-UK-2026", title="Claims Handling Procedure (UK)", policy_type="process",
   jurisdiction="UK", effective_from="2026-01-01", effective_to=None,
   version="v4", superseded=False, text="""
SECTION 1 INTAKE
A claim is registered on notification. The handler confirms cover, records the
incident details, and requests supporting documents within two working days.

SECTION 2 ASSESSMENT
The handler reviews the repair estimate against the policy schedule. Where the
estimate exceeds the authorisation threshold, the handler escalates before any
work is authorised.

SECTION 3 ENGINEER REFERRAL
An engineer inspection is required where the supplementary estimate exceeds the
threshold in the policy wording, or where damage is inconsistent with the
reported incident.

SECTION 4 SETTLEMENT
Settlement is authorised by a handler with the appropriate mandate. No claim is
settled without a recorded human decision.
"""),

_d(id="PRC-CLM-UK-2025", title="Claims Handling Procedure (UK)", policy_type="process",
   jurisdiction="UK", effective_from="2025-01-01", effective_to="2025-12-31",
   version="v3", superseded=True, text="""
SECTION 1 INTAKE
A claim is registered on notification. The handler confirms cover and requests
supporting documents within five working days.

SECTION 3 ENGINEER REFERRAL
An engineer inspection is required for all supplementary estimates regardless
of value.
"""),

# ---------- FRAUD ----------
_d(id="POL-FRD-UK-2026", title="Counter-Fraud Policy (UK)", policy_type="fraud",
   jurisdiction="UK", effective_from="2026-01-01", effective_to=None,
   version="v2", superseded=False, text="""
SECTION 2 INDICATORS
Indicators warranting referral include: inconsistency between the reported
incident and the damage described, a claim notified close to policy inception
or expiry, and repeated claims on the same vehicle within twelve months.

SECTION 3 REFERRAL
A referral to the counter-fraud team must be raised within two working days of
an indicator being identified. Referral does not delay the handling of the
claim unless the counter-fraud team instructs otherwise.

SECTION 5 CUSTOMER COMMUNICATION
No communication indicating suspicion of fraud is sent to a customer without
counter-fraud team approval.
"""),

# ---------- CONDUCT / VULNERABILITY ----------
_d(id="POL-CON-UK-2026", title="Customer Conduct Standards (UK)", policy_type="conduct",
   jurisdiction="UK", effective_from="2026-01-01", effective_to=None,
   version="v3", superseded=False, text="""
SECTION 1 FAIR OUTCOMES
Decisions must be capable of being explained to the customer in plain language.
Any declined claim must be accompanied by the reasons for the decision and the
route to challenge it.

SECTION 2 VULNERABLE CUSTOMERS
Where a customer discloses a circumstance affecting their ability to engage
with the process, the handler records the disclosure and adjusts the process
accordingly. Adjustments are recorded on the claim.

SECTION 4 AUTOMATED DECISIONS
No decision that significantly affects a customer is made solely by automated
means. A human with appropriate mandate reviews and records every decision.
"""),

# ---------- DATA PROTECTION ----------
_d(id="POL-DPA-EU-2026", title="Data Protection Standard (EU)", policy_type="data",
   jurisdiction="EU", effective_from="2026-01-01", effective_to=None,
   version="v5", superseded=False, text="""
SECTION 3 MINIMISATION
Only data necessary for the stated purpose is collected and retained. Identity
evidence is verified and the verdict retained; the evidence itself is not
copied into downstream systems.

SECTION 4 RETENTION
Claim records are retained for six years from closure. Records are held in a
form that cannot be altered after the decision is recorded.

SECTION 6 AUTOMATED PROCESSING
Where processing is automated, the data subject is informed, may contest the
outcome, and may request human intervention.
"""),

# ---------- PROPERTY (a different policy_type, to test filtering) ----------
_d(id="POL-PRP-UK-2026", title="Property Policy Wording (UK)", policy_type="property",
   jurisdiction="UK", effective_from="2026-01-01", effective_to=None,
   version="v2", superseded=False, text="""
CLAUSE 4.1 TOTAL LOSS
A property is treated as a total loss where the estimated cost of reinstatement
exceeds eighty per cent of the sum insured.

CLAUSE 2.4 EXCESS
The policyholder pays the excess stated in the schedule. Where the loss arises
from escape of water, an additional excess of GBP 500 applies.

CLAUSE 3.2 REPAIR AUTHORISATION
Reinstatement works above GBP 10,000 require a surveyor report before
authorisation.
"""),
]

# ---------- filler documents, to give retrieval something to get wrong ----------
_FILLER = [
    ("POL-LIA-UK-2026", "Liability Policy Wording (UK)", "liability", "UK", "2026-01-01", None, "v1", False,
     "CLAUSE 1.1 SCOPE\nCover applies to legal liability to third parties arising from use of the "
     "insured vehicle.\n\nCLAUSE 5.3 UNINSURED DRIVER\nNo cover applies where the vehicle was driven "
     "by a person not named on the schedule."),
    ("PRC-CMP-UK-2026", "Complaints Procedure (UK)", "process", "UK", "2026-01-01", None, "v2", False,
     "SECTION 1 RECEIPT\nA complaint is acknowledged within three working days.\n\nSECTION 3 FINAL "
     "RESPONSE\nA final response is issued within eight weeks, with details of the right to refer "
     "the matter onward."),
    ("PRC-ESC-UK-2026", "Escalation Matrix (UK)", "process", "UK", "2026-01-01", None, "v1", False,
     "SECTION 2 AUTHORITY LIMITS\nHandler grade 2 may authorise settlements up to GBP 10,000. "
     "Grade 3 up to GBP 50,000. Above that, a claims manager authorises."),
    ("POL-REI-UK-2026", "Reinsurance Notification (UK)", "reinsurance", "UK", "2026-01-01", None, "v1", False,
     "SECTION 1 THRESHOLD\nClaims with an estimated total incurred above GBP 250,000 are notified "
     "to reinsurers within ten working days."),
    ("PRC-REC-UK-2026", "Recoveries Procedure (UK)", "process", "UK", "2026-01-01", None, "v3", False,
     "SECTION 2 THIRD PARTY RECOVERY\nWhere a third party is liable, recovery action is commenced "
     "within thirty days of settlement."),
    ("POL-MOT-EU-2026", "Motor Policy Wording (EU passporting)", "motor", "EU", "2026-01-01", None, "v1", False,
     "CLAUSE 4.1 TOTAL LOSS\nA vehicle is treated as a total loss where the estimated cost of repair "
     "exceeds sixty-five per cent of its pre-accident market value.\n\nCLAUSE 6.2 CLAIM NOTIFICATION\n"
     "A claim must be notified within thirty days of the incident."),
    ("POL-CON-IE-2026", "Customer Conduct Standards (Ireland)", "conduct", "IE", "2026-01-01", None, "v1", False,
     "SECTION 4 AUTOMATED DECISIONS\nNo decision that significantly affects a customer is made solely "
     "by automated means."),
    ("PRC-CLM-IE-2026", "Claims Handling Procedure (Ireland)", "process", "IE", "2026-01-01", None, "v2", False,
     "SECTION 2 ASSESSMENT\nThe handler reviews the repair estimate against the policy schedule. "
     "Escalation applies above the authorisation threshold."),
    ("POL-FRD-UK-2024", "Counter-Fraud Policy (UK)", "fraud", "UK", "2024-01-01", "2025-12-31", "v1", True,
     "SECTION 3 REFERRAL\nA referral to the counter-fraud team must be raised within five working "
     "days of an indicator being identified."),
    ("POL-DPA-UK-2024", "Data Protection Standard (UK)", "data", "UK", "2024-01-01", "2025-12-31", "v3", True,
     "SECTION 4 RETENTION\nClaim records are retained for three years from closure."),
]
for _id, _t, _pt, _j, _ef, _et, _v, _s, _tx in _FILLER:
    DOCS.append(_d(id=_id, title=_t, policy_type=_pt, jurisdiction=_j,
                   effective_from=_ef, effective_to=_et, version=_v,
                   superseded=_s, text=_tx))


# ==========================================================================
# QUERY SET
#
# Each query names the DECISION CONTEXT — jurisdiction and decision date —
# because that is what a real claim carries. The correct answer depends on it.
# ==========================================================================
QUERIES: list[dict] = [
    {"id": "Q01", "type": "version_trap",
     "q": "What is the total loss threshold?",
     "jurisdiction": "UK", "decision_date": "2026-04-01",
     "policy_type": "motor",
     "expect_doc": "POL-MOT-UK-2026", "expect_text": "sixty per cent",
     "trap": "2024 says seventy, 2025 says sixty-five, Ireland says fifty-five, "
             "property says eighty. Four wrong answers, all well written."},

    {"id": "Q02", "type": "version_trap",
     "q": "At what driver age does the young driver excess apply, and how much?",
     "jurisdiction": "UK", "decision_date": "2026-04-01",
     "policy_type": "motor",
     "expect_doc": "POL-MOT-UK-2026", "expect_text": "twenty-five",
     "trap": "2024 says twenty-one/GBP 150. 2025 says twenty-three/GBP 200."},

    {"id": "Q03", "type": "jurisdiction_trap",
     "q": "How many days do I have to notify a claim?",
     "jurisdiction": "IE", "decision_date": "2026-04-01",
     "policy_type": "motor",
     "expect_doc": "POL-MOT-IE-2026", "expect_text": "twenty-eight days",
     "trap": "UK and EU both say thirty days. Ireland says twenty-eight."},

    {"id": "Q04", "type": "jurisdiction_trap",
     "q": "What is the total loss threshold?",
     "jurisdiction": "IE", "decision_date": "2026-04-01",
     "policy_type": "motor",
     "expect_doc": "POL-MOT-IE-2026", "expect_text": "fifty-five per cent",
     "trap": "Same question as Q01, different country, different answer."},

    {"id": "Q05", "type": "type_trap",
     "q": "What is the repair authorisation threshold?",
     "jurisdiction": "UK", "decision_date": "2026-04-01",
     "policy_type": "motor",
     "expect_doc": "POL-MOT-UK-2026", "expect_text": "5,000",
     "trap": "Property says GBP 10,000 with a surveyor report. Same clause "
             "number, same wording, wrong product."},

    {"id": "Q06", "type": "historic",
     "q": "What was the total loss threshold for a claim decided in June 2025?",
     "jurisdiction": "UK", "decision_date": "2025-06-01",
     "policy_type": "motor",
     "expect_doc": "POL-MOT-UK-2025", "expect_text": "sixty-five per cent",
     "trap": "The CURRENT version is wrong here. A decision is judged against "
             "the rules in force on the day it was made."},

    {"id": "Q07", "type": "paraphrase",
     "q": "When must we bring in an engineer to look at extra damage found later?",
     "jurisdiction": "UK", "decision_date": "2026-04-01",
     "policy_type": "motor",
     "expect_doc": "POL-MOT-UK-2026", "expect_text": "twenty per cent",
     "trap": "The document says 'supplementary estimate' and 'engineer "
             "inspection'. The question says neither. Keyword search struggles."},

    {"id": "Q08", "type": "paraphrase",
     "q": "Can the system decide a claim on its own without a person looking at it?",
     "jurisdiction": "UK", "decision_date": "2026-04-01",
     "policy_type": "conduct",
     "expect_doc": "POL-CON-UK-2026", "expect_text": "solely by automated means",
     "trap": "No shared vocabulary between question and clause."},

    {"id": "Q09", "type": "clause_lookup",
     "q": "What does clause 3.5 say?",
     "jurisdiction": "UK", "decision_date": "2026-04-01",
     "policy_type": "motor",
     "expect_doc": "POL-MOT-UK-2026", "expect_text": "supplementary",
     "trap": "A clause number is an identifier. Vector search is blind to it."},

    {"id": "Q10", "type": "negative",
     "q": "What is the courtesy car entitlement?",
     "jurisdiction": "UK", "decision_date": "2026-04-01",
     "policy_type": "motor",
     "expect_doc": None, "expect_text": None,
     "trap": "It is not in the corpus. The honest answer is that it is absent."},
]


def in_force(doc: PolicyDoc, on: str) -> bool:
    if doc.effective_from > on:
        return False
    if doc.effective_to and doc.effective_to < on:
        return False
    return True
