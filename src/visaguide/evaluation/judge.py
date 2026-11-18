"""LLM-as-judge for answer quality. Two independent rubrics, two calls:

1. Faithfulness and citation accuracy (judge sees: sources + generated answer).
   The answer is split into atomic claims. For each claim:
     supported_by_any   - backed by any retrieved source?  (faithfulness, groundedness)
     supported_by_cited - backed by the sources it cites?  (citation accuracy)
   The judge never sees the reference answer here, so "matches what we expected"
   can't leak into "is supported by the evidence".

2. Correctness (judge sees: question + reference answer + generated answer).
   correct / partial / incorrect against the reference's key facts.

Why separate calls: a single prompt scoring several dimensions tends to let one
impression colour the others (halo effect). Why atomic claims: "is this answer
faithful?" is vague; "is THIS claim stated in the sources?" is checkable, and it
gives a graded score instead of a yes/no.

The judge is itself a model and can be wrong, so we measure its agreement with
human labels (scripts/judge_agreement.py) before trusting its numbers.
"""

from dataclasses import dataclass

from visaguide.generation.llm import LLMClient, LLMResponse

FAITHFULNESS_SYSTEM = """You are a strict evaluator of answers produced by a \
retrieval-augmented assistant. You check whether each claim in the answer is supported \
by the provided sources. Use ONLY the sources; your own knowledge of the topic is \
irrelevant, even if a claim is true in reality.

Steps:
1. Split the answer into atomic factual claims (one fact each). Ignore citation markers, \
greetings, and generic advice such as "confirm with the authority" or "this is not legal \
advice". If the answer only says the information is not available, return no claims.
2. For each claim, list the source ids the answer cites for it (from markers like [S2]).
3. supported_by_cited: true only if the cited source(s) state the claim (paraphrase is \
fine; an added number, condition, or deadline that is not stated is NOT supported).
4. supported_by_any: true if any of the provided sources states the claim."""

FAITHFULNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "claims": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "claim": {"type": "string"},
                    "cited_sources": {"type": "array", "items": {"type": "string"}},
                    "supported_by_cited": {"type": "boolean"},
                    "supported_by_any": {"type": "boolean"},
                },
                "required": ["claim", "cited_sources", "supported_by_cited", "supported_by_any"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["claims"],
    "additionalProperties": False,
}

CORRECTNESS_SYSTEM = """You grade an assistant's answer against a reference answer \
written by a human expert for the same question.

1. List the key facts in the reference answer that a student needs.
2. Grade the assistant's answer:
   - "correct": states all key facts (paraphrase is fine) and contradicts none.
   - "partial": states some key facts or is vaguer than the reference, but contradicts none.
   - "incorrect": contradicts a key fact, gives a wrong amount/deadline/rule, refuses or \
says the information is unavailable when the reference gives it, or does not answer.
Extra correct details beyond the reference do not lower the grade. Mentioning that \
sources disagree is fine if the reference's fact is included."""

CORRECTNESS_SCHEMA = {
    "type": "object",
    "properties": {
        "reference_key_facts": {"type": "array", "items": {"type": "string"}},
        "verdict": {"type": "string", "enum": ["correct", "partial", "incorrect"]},
        "explanation": {"type": "string"},
    },
    "required": ["reference_key_facts", "verdict", "explanation"],
    "additionalProperties": False,
}

CORRECTNESS_SCORE = {"correct": 1.0, "partial": 0.5, "incorrect": 0.0}


@dataclass
class JudgeResult:
    raw: dict | None
    response: LLMResponse


def judge_faithfulness(llm: LLMClient, sources_block: str, answer: str) -> JudgeResult:
    user = f"{sources_block}\n\n<answer>\n{answer}\n</answer>"
    resp = llm.generate_json(FAITHFULNESS_SYSTEM, user, FAITHFULNESS_SCHEMA)
    return JudgeResult(resp.data, resp)


def judge_correctness(llm: LLMClient, question: str, reference: str, answer: str) -> JudgeResult:
    user = (f"<question>{question}</question>\n<reference_answer>{reference}</reference_answer>\n"
            f"<assistant_answer>{answer}</assistant_answer>")
    resp = llm.generate_json(CORRECTNESS_SYSTEM, user, CORRECTNESS_SCHEMA)
    return JudgeResult(resp.data, resp)


def faithfulness_scores(claims: list[dict]) -> tuple[float | None, float | None]:
    """(faithfulness, citation_accuracy) = share of claims supported by any / by cited source.
    None when there are no claims (e.g. a refusal), so refusals don't inflate the mean."""
    if not claims:
        return None, None
    n = len(claims)
    return (sum(c["supported_by_any"] for c in claims) / n,
            sum(c["supported_by_cited"] for c in claims) / n)
