"""
Builds dataset.csv and dataset.json for the CAI Semantic Equivalence Benchmark v0.2.

Adds domain column to existing 300 pairs and appends 80 new policy-domain pairs.
Run: python build_dataset.py
"""

import csv
import json

DOMAIN_MAP = {
    **{i: "factual"             for i in range(1,  21)},
    **{i: "everyday_reasoning"  for i in range(21, 41)},
    **{i: "math_logic"          for i in range(41, 61)},
    **{i: "counterfactual"      for i in range(61, 81)},
    **{i: "ethics"              for i in range(81, 111)},
    **{i: "ai_safety"           for i in range(111, 131)},
    **{i: "practical_planning"  for i in range(131, 151)},
    **{i: "causal_reasoning"    for i in range(151, 171)},
    **{i: "philosophy"          for i in range(171, 191)},
    **{i: "social_emotional"    for i in range(191, 211)},
    **{i: "creative_writing"    for i in range(211, 231)},
    **{i: "summarization"       for i in range(231, 271)},
    **{i: "cai_meta"            for i in range(271, 301)},
}

POLICY_PAIRS = [
    # --- ECOMMERCE (301-320) ---
    (301, "ecommerce", "Can I return a product after 30 days?", "I bought something 5 weeks ago. Can I still return it?"),
    (302, "ecommerce", "Do you offer price matching?", "If I find this cheaper elsewhere will you match the price?"),
    (303, "ecommerce", "What is your refund policy for digital products?", "I bought a digital download. Can I get my money back?"),
    (304, "ecommerce", "How long does standard shipping take?", "When will my order arrive if I pick the cheapest shipping?"),
    (305, "ecommerce", "Is there a restocking fee for returns?", "Will I be charged anything if I send this back?"),
    (306, "ecommerce", "Can I exchange a product without a receipt?", "I lost the receipt. Can I still swap this for a different size?"),
    (307, "ecommerce", "Do you ship internationally?", "I am in Canada. Can you deliver to me?"),
    (308, "ecommerce", "What happens if my order arrives damaged?", "My package showed up broken. What do I do?"),
    (309, "ecommerce", "Can I cancel an order after it has shipped?", "I want to cancel but the tracking says it is already on its way."),
    (310, "ecommerce", "Is there a warranty on electronics?", "My laptop broke after 8 months. Is it still covered?"),
    (311, "ecommerce", "Can I use two discount codes on one order?", "I have a promo code and a loyalty code. Can I apply both?"),
    (312, "ecommerce", "Do sale items qualify for returns?", "I bought this on clearance. Can I return it?"),
    (313, "ecommerce", "How do I track my order?", "Where is my package right now?"),
    (314, "ecommerce", "Can I change my shipping address after placing an order?", "I entered the wrong address. Is it too late to fix it?"),
    (315, "ecommerce", "Do gift cards expire?", "I have a gift card from two years ago. Can I still use it?"),
    (316, "ecommerce", "What is the return window for clothing?", "I need to return a shirt I bought last month. Is that allowed?"),
    (317, "ecommerce", "Can I return an opened item?", "I already opened the box. Can I still send it back?"),
    (318, "ecommerce", "Do you offer free returns?", "Will I have to pay to ship this back?"),
    (319, "ecommerce", "What is your policy on late deliveries?", "My order was supposed to arrive three days ago. What can you do?"),
    (320, "ecommerce", "Can I get a refund instead of store credit?", "I was offered store credit for a return. Can I get cash back instead?"),

    # --- HR (321-340) ---
    (321, "hr", "How many vacation days do employees get per year?", "How much paid time off am I entitled to annually?"),
    (322, "hr", "Can I carry over unused PTO to next year?", "I did not use all my vacation days. Do they roll over?"),
    (323, "hr", "What is the company policy on remote work?", "Am I allowed to work from home?"),
    (324, "hr", "How does the parental leave policy work?", "I am expecting a baby. How much leave can I take?"),
    (325, "hr", "What happens to my benefits if I go on unpaid leave?", "If I take unpaid leave does my health insurance continue?"),
    (326, "hr", "How many sick days are employees allowed?", "How many days can I call in sick before it is a problem?"),
    (327, "hr", "What is the process for requesting a salary review?", "How do I ask for a raise?"),
    (328, "hr", "Can I be terminated without a warning?", "Is it legal for the company to fire me without notice?"),
    (329, "hr", "What is the severance policy for laid-off employees?", "If I get laid off how much severance will I receive?"),
    (330, "hr", "Does the company cover relocation costs?", "If I move for this job will the company pay for it?"),
    (331, "hr", "How does overtime pay work?", "If I work more than 40 hours a week do I get paid extra?"),
    (332, "hr", "What is the policy on personal use of company devices?", "Can I use my work laptop for personal things?"),
    (333, "hr", "How do I report a workplace harassment complaint?", "Someone is harassing me at work. Who do I tell?"),
    (334, "hr", "What are the grounds for termination?", "What can get an employee fired?"),
    (335, "hr", "Does the company match 401k contributions?", "Will my employer match what I put into my retirement account?"),
    (336, "hr", "What is the policy on bereavement leave?", "My parent passed away. How much time off do I get?"),
    (337, "hr", "Can I take leave for jury duty?", "I was called for jury service. Will I be paid during that time?"),
    (338, "hr", "What is the dress code policy?", "Is there a required dress code at this company?"),
    (339, "hr", "How are performance reviews conducted?", "When and how do I get evaluated at this job?"),
    (340, "hr", "Can I work a second job while employed here?", "Am I allowed to have a side job while working here?"),

    # --- HEALTHCARE (341-360) ---
    (341, "healthcare", "Is this procedure covered by my insurance?", "Will my plan pay for this surgery?"),
    (342, "healthcare", "Do I need a referral to see a specialist?", "Can I book an appointment with a cardiologist directly?"),
    (343, "healthcare", "What is my deductible for the year?", "How much do I have to pay out of pocket before insurance kicks in?"),
    (344, "healthcare", "Does my plan cover mental health services?", "Will insurance pay for therapy sessions?"),
    (345, "healthcare", "Is prior authorization required for this medication?", "Does my doctor need to get approval before prescribing this drug?"),
    (346, "healthcare", "What is my out-of-pocket maximum?", "What is the most I will ever have to pay in a single year?"),
    (347, "healthcare", "Does this plan cover prescription drugs?", "Will my insurance pay for medications?"),
    (348, "healthcare", "Am I covered for emergency room visits?", "If I go to the ER will my insurance cover it?"),
    (349, "healthcare", "What is the copay for a primary care visit?", "How much do I pay when I see my regular doctor?"),
    (350, "healthcare", "Does my insurance cover out-of-network providers?", "Can I see a doctor who is not in the network?"),
    (351, "healthcare", "How do I appeal a denied claim?", "My insurance denied my claim. How do I fight that?"),
    (352, "healthcare", "Is preventive care covered at no cost?", "Do I have to pay for my annual physical?"),
    (353, "healthcare", "Does my plan cover chiropractic care?", "Will insurance pay for chiropractor visits?"),
    (354, "healthcare", "What is the waiting period before coverage begins?", "When does my insurance start covering me?"),
    (355, "healthcare", "Does the plan cover maternity care?", "Will my insurance cover prenatal appointments and delivery?"),
    (356, "healthcare", "Is vision care included in my plan?", "Does my insurance cover eye exams and glasses?"),
    (357, "healthcare", "Are lab tests covered under my plan?", "Will my insurance pay for blood work?"),
    (358, "healthcare", "Does my insurance cover physical therapy?", "Will insurance pay for PT sessions after my surgery?"),
    (359, "healthcare", "What is the process for getting a second opinion covered?", "Can I see another doctor for a second opinion and have it covered?"),
    (360, "healthcare", "How do I add a dependent to my health plan?", "I just had a baby. How do I put them on my insurance?"),

    # --- LEGAL (361-380) ---
    (361, "legal", "Is this considered a breach of contract?", "The other party did not do what we agreed. Do I have a case?"),
    (362, "legal", "Can I sue for emotional distress?", "I suffered psychological harm because of someone else. Can I take legal action?"),
    (363, "legal", "What is the statute of limitations for this type of claim?", "How long do I have to file this lawsuit?"),
    (364, "legal", "Does this clause make the entire contract unenforceable?", "If one part of the contract is illegal does the whole thing fall apart?"),
    (365, "legal", "Am I liable for what happens on my property?", "If someone gets hurt on my land am I responsible?"),
    (366, "legal", "Can my employer monitor my personal email on a work device?", "Is it legal for my company to read my personal emails if I send them from a work laptop?"),
    (367, "legal", "What counts as defamation?", "Someone said false things about me publicly. Is that illegal?"),
    (368, "legal", "Can I record a phone call without telling the other person?", "Is it legal to record a conversation without consent?"),
    (369, "legal", "What rights do I have if I am arrested?", "If the police arrest me what am I allowed to do?"),
    (370, "legal", "Is a verbal agreement legally binding?", "We made a deal but did not sign anything. Is that enforceable?"),
    (371, "legal", "Can a landlord enter my apartment without notice?", "Is my landlord allowed to come in without asking first?"),
    (372, "legal", "What is the difference between civil and criminal liability?", "How is a civil lawsuit different from a criminal charge?"),
    (373, "legal", "Do I need a lawyer to write a will?", "Can I create a valid will without an attorney?"),
    (374, "legal", "What constitutes intellectual property theft?", "If someone copies my work without permission is that illegal?"),
    (375, "legal", "Can my employer change my contract terms without my consent?", "Is it legal for my company to change my work agreement without asking me?"),
    (376, "legal", "What is the legal definition of negligence?", "When is someone legally at fault for not being careful enough?"),
    (377, "legal", "Does this non-compete clause hold up in court?", "Can my employer actually enforce the non-compete I signed?"),
    (378, "legal", "What are my rights if a product injures me?", "I was hurt by a defective product. What can I do legally?"),
    (379, "legal", "Is it legal to use someone else's photo without permission?", "Can I post a picture of someone online if they did not say it was OK?"),
    (380, "legal", "What happens if the other party violates a court order?", "Someone ignored a court order that was against them. What are my options?"),
]


def build():
    existing = []
    with open("dataset_original.csv", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            pid = int(row["pair_id"])
            existing.append({
                "pair_id": pid,
                "domain": DOMAIN_MAP[pid],
                "prompt_A": row["prompt_A"],
                "prompt_B": row["prompt_B"],
            })

    policy_rows = [
        {"pair_id": pid, "domain": domain, "prompt_A": a, "prompt_B": b}
        for pid, domain, a, b in POLICY_PAIRS
    ]

    all_rows = existing + policy_rows

    with open("dataset.csv", "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["pair_id", "domain", "prompt_A", "prompt_B"])
        writer.writeheader()
        writer.writerows(all_rows)

    with open("dataset.json", "w", encoding="utf-8") as f:
        json.dump(all_rows, f, indent=2, ensure_ascii=False)

    print(f"Built {len(all_rows)} pairs.")
    domain_counts = {}
    for r in all_rows:
        domain_counts[r["domain"]] = domain_counts.get(r["domain"], 0) + 1
    for d, n in sorted(domain_counts.items()):
        print(f"  {d}: {n}")


if __name__ == "__main__":
    build()
