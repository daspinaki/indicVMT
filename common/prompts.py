"""Prompt templates for the eight prompting strategies (paper, Appendix F).

The template text is reproduced exactly as printed in the paper (Figs. 18-26),
including its original spelling. Every strategy is built through
`build_prompt`, so all four models receive identical prompts.
"""

import textwrap

from .exemplars import INDIC_MMT_PROMPTS_3, INDIC_MMT_PROMPTS_3_NO_TDV

# ---------------------------------------------------------------------------
# Fig. 18 - video description prompt (used offline with MiniCPM-V to build TDV)
# ---------------------------------------------------------------------------
DESCRIPTION_PROMPT = (
    "You are given a video. Provide a detailed, structured description that highlights visual cues useful for Video-Guided Machine Translation.\n"
    "Focus on the following aspects:\n"
    "1. Scene context: Specify indoor/outdoor, time of day, lighting, and background environment.\n"
    "2. Objects: List all visible objects with size, shape, color, material, and relative positions.\n"
    "3. Entities: Describe people, animals, or other beings — include appearance, clothing, facial expressions, posture, and interactions.\n"
    "4. Actions and dynamics: Explain what each entity is doing, gestures, movements, and how they interact with objects or others.\n"
    "5. Temporal flow: Summarize the sequence of events across frames rather than one static moment.\n"
    "6. Communication cues: If someone is speaking, describe who appears to be speaking, their body language, and reactions of others.\n"
    "7. Ambiguity resolution: Add details that can disambiguate meaning (e.g., homonyms, cultural context, visible numbers, signs, or symbols).\n\n"
    "Important instructions:\n"
    "- Be factual and precise; avoid guessing or inventing details.\n"
    "- Use clear and concise sentences.\n"
    "- Write in a way that helps disambiguate translations by grounding abstract terms with visible details."
)

# ---------------------------------------------------------------------------
# Fig. 19 - Zero-shot
# ---------------------------------------------------------------------------
ZERO_SHOT_SYSTEM = (
    "You are a professional machine translation system. "
    "You translate English sentences into the target language."
)
ZERO_SHOT = """English sentence:
{english_src}

Task:
Translate the English sentence into the {target} language.
Produce a fluent and natural translation."""

# ---------------------------------------------------------------------------
# Fig. 20 - Zero-shot + Video (frames)
# ---------------------------------------------------------------------------
ZERO_SHOT_VIDEO = """You are a translation assistant. Look at the provided video frames and the English sentence below, then translate it to {target}. Output ONLY the translation, nothing else.

English sentence: {english_src}

Consider the visual context from the video frames to inform your translation, particularly for context-dependent words.

{target}:"""

# ---------------------------------------------------------------------------
# Fig. 21 - Zero-shot + TDV (text description of video)
# ---------------------------------------------------------------------------
ZERO_SHOT_TDV = """Video Description: {desc}

English sentence:
{english_src}
Translate the English sentence into the {target} language.
Ground the translation in the video description.
Produce a fluent and natural translation.
Think about the entities and actions described in the video description to inform your translation.
Only ouput the translated sentence without any additional text."""

# ---------------------------------------------------------------------------
# Fig. 22 - Few-shot only (language-specific exemplars, no video description)
# ---------------------------------------------------------------------------
FEW_SHOT_ONLY = """English sentence: {english_src}
Translate the English sentence into the {target} language.
{few_shot_prompt}
Produce a fluent and natural translation. Only output the translated sentence."""

# ---------------------------------------------------------------------------
# Fig. 23 - Hindi few-shot + TDV (Hindi exemplars for every target language)
# ---------------------------------------------------------------------------
GROUNDED_SYSTEM = (
    "You are a professional visually grounded machine translation system. "
    "Use the video description to resolve ambiguities and ensure faithful translation."
)
HINDI_FEW_SHOT_TDV = """Video description:
{desc}

English sentence:
{english_src}

Task:
Translate the English sentence into the {target} language.
Ground the translation in the video description.
Produce a fluent and natural translation.

{few_shot_prompt}

Think about the entities and actions described in the video description to inform your translation.
Only output the final translated sentence without any additional explanations."""

# ---------------------------------------------------------------------------
# Fig. 24 - Language-specific few-shot + TDV
# ---------------------------------------------------------------------------
LANG_FEW_SHOT_TDV = """Video description:
{desc}

English sentence:
{english_src}

Task:
Translate the English sentence into the {target} language.
Ground the translation in the video description.
Produce a fluent and natural translation.

Contextual Examples for {target}:
{few_shot_prompt}

Think about the entities and actions described in the video description to inform your translation.
Only output the final translated sentence without any additional explanations."""

# ---------------------------------------------------------------------------
# Fig. 25 - Dependent cascade (frames + source sentence, single call)
# ---------------------------------------------------------------------------
DEPENDENT_CASCADE = (
    "English sentence: {english_src}\n\n"
    "Using the provided image frame, identify important visual information relevant to the source sentence "
    "and use it to translate the sentence into {target}. "
    "Focus on objects, actions, and scene context that help resolve ambiguity.\n\n"
    "Output ONLY the translated sentence."
)

# ---------------------------------------------------------------------------
# Fig. 26 - Independent cascade (frames, single call)
# ---------------------------------------------------------------------------
INDEPENDENT_CASCADE = (
    
    "Using the provided image frames, identify important visual information in the frame."
    "Now use it to translate the below given sentence into {target}.\n\n"
    "English sentence: {english_src}.\n\n"
    "Output ONLY the translated sentence."
)


# strategy -> (system prompt, user template, needs frames, needs description)
STRATEGIES = {
    "zero_shot":           (ZERO_SHOT_SYSTEM, ZERO_SHOT,          False, False),
    "zero_shot_video":     (None,             ZERO_SHOT_VIDEO,    True,  False),
    "zero_shot_tdv":       (None,             ZERO_SHOT_TDV,      False, True),
    "fewshot_only":        (None,             FEW_SHOT_ONLY,      False, False),
    "hindi_fshot_tdv":     (None,  HINDI_FEW_SHOT_TDV, False, True),
    "lang_fshot_tdv":      (None,             LANG_FEW_SHOT_TDV,  False, True),
    "dependent_cascade":   (None,             DEPENDENT_CASCADE,  True,  False),
    "independent_cascade": (None,             INDEPENDENT_CASCADE, True, False),
}


def _few_shot_block(strategy, lang):
    if strategy == "fewshot_only":
        block = INDIC_MMT_PROMPTS_3_NO_TDV[lang]
    elif strategy == "hindi_fshot_tdv":
        block = INDIC_MMT_PROMPTS_3["hindi"]
    elif strategy == "lang_fshot_tdv":
        block = INDIC_MMT_PROMPTS_3[lang]
    else:
        return ""
    return textwrap.dedent(block).strip()


def needs_frames(strategy):
    return STRATEGIES[strategy][2]


def needs_description(strategy):
    return STRATEGIES[strategy][3]


def build_prompt(strategy, english_src, lang, target_name, desc=None):
    """Return (system_prompt or None, user_prompt) for one instance.

    lang:        language key, e.g. "tamil" (selects the exemplars)
    target_name: language name inserted into the prompt, e.g. "Tamil"
    """
    system, template, _, uses_desc = STRATEGIES[strategy]
    if uses_desc and not desc:
        raise ValueError(f"Strategy '{strategy}' requires a video description.")
    user = template.format(
        english_src=english_src,
        target=target_name,
        desc=desc,
        few_shot_prompt=_few_shot_block(strategy, lang),
    )
    return system, user
