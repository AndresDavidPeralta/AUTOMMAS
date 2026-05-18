"""
prompts/tc_prompts.py
─────────────────────
Prompt templates for the TC Generator and TC Validator agents.
Encodes the structured rules from Hori (2010) for each TC component.

These prompts are also reused by the two frontier zero-shot baselines
to ensure a fair comparison.
"""

# ─────────────────────────────────────────────────────────────────────────────
# RULES
# ─────────────────────────────────────────────────────────────────────────────

TC_RULES = """
The Test Case MUST follow these structural rules, derived from Hori (2010)
and industrial Black Box testing standards (IEEE 829):

────────────────────────────────────────────────────────────────────
[TITLE] — single line, 5-12 words
  Pattern: <Type> - <Action> <Entity> <Qualifier> [<Location>] [<Condition>]
  Terms:
    Type       (mandatory) — short abbreviation of test case type, e.g. "TC"
    Action     (mandatory) — action verb; if none, use "Verify"
    Entity     (mandatory) — main object under test
    Qualifier  (mandatory) — qualification of the entity; if none, use "behavior"
    Location   (optional)  — where the functionality is found
    Condition  (optional)  — any extra preconditions
  Examples:
    "TC - Verify DDS behavior after switching to SIM1"
    "TC - Open Bluetooth menu from Settings"

────────────────────────────────────────────────────────────────────
[SUMMARY] — single declarative sentence
  Must be assertive and informative; complements the Title without
  repeating it. Describes the functional objective of the test.
  Example:
    "Verify that the user can connect the device to a Wi-Fi network
     using the Settings menu and that a confirmation is displayed."

────────────────────────────────────────────────────────────────────
[INITIAL SETUP] — one or more lines, each describing a STATE
  Each line MUST be in passive voice or state-form (NOT imperative).
  Pattern: <Entity> <State> [<Location>]
  Terms:
    Entity   (mandatory) — object whose state matters
    State    (mandatory) — verb expressing state (enabled, inserted, connected, …)
    Location (optional)  — where the state must hold
  Examples:
    "Dual SIM cards inserted"
    "Data usage enabled on SIM2"
    "Debug menu enabled"
  NOT VALID (imperative form rejected):
    "Insert two SIM cards"     ← wrong, action voice
    "Enable data usage"        ← wrong, action voice

────────────────────────────────────────────────────────────────────
[STEPS] — numbered, one action per line, imperative voice
  Pattern: [<Negation>] <Action> <Entity> <Qualifier> [<Location>] [<Repetition>]
  Terms:
    Negation   (optional)  — to deny a state (e.g. "Do not")
    Action     (mandatory) — imperative verb (Tap, Open, Enter, Swipe, …)
    Entity     (mandatory) — object acted upon
    Qualifier  (mandatory) — if none, use "behavior"
    Location   (optional)  — UI location
    Repetition (optional)  — any extra condition
  Rule of atomicity: ONE action per step. Do not chain multiple actions.
  Examples:
    "1. Open Contacts app"
    "2. Tap on Menu option"

────────────────────────────────────────────────────────────────────
[EXPECTED RESULTS] — one or more declarative sentences
  Pattern: <Entity> <Modal verb> [<Negation>] <Main verb> <Result> [<Complement>]
  Terms:
    Entity      (mandatory) — object that has an expected state
    Modal verb  (mandatory) — MUST be "should" (no exceptions)
    Negation    (optional)  — to deny a state
    Main verb   (mandatory) — main action verb
    Result      (mandatory) — observable state or characteristic
    Complement  (optional)  — list/pop-up/dialog/toast detail
  Examples:
    "Photo/images should be displayed on folder"
    "The Wi-Fi confirmation toast should appear at the bottom of the screen"
"""

# ─────────────────────────────────────────────────────────────────────────────
# GENERATOR PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

GENERATOR_SYSTEM_PROMPT = (
    "You are a senior QA Automation Engineer specialized in writing precise, "
    "atomic, and reproducible Test Cases for mobile applications. "
    "You strictly follow industrial standards and the structural rules "
    "provided in the user prompt. You produce plain text output only — "
    "no JSON, no markdown formatting."
)


GENERATOR_USER_PROMPT_TEMPLATE = """Generate ONE complete Test Case for the following Use Case.

[USE CASE]
Name: {name}
Description: {description}
Steps as described in the manual:
{steps}

[STRUCTURAL RULES]
{rules}

[SCOPE PRESERVATION — IMPORTANT BUT BOUNDED]
You MUST preserve the scope of the Use Case AS DESCRIBED IN THE MANUAL.
Do NOT invent operations, alternatives, options, or scenarios that are
NOT explicitly present in the Use Case steps shown above. Specifically:

  1. If the Use Case name mentions multiple operations (e.g., "Insert or
     remove"), the Test Case MUST cover all of them. The Title must
     reflect every operation covered.

  2. If a step in the Use Case offers multiple alternative methods
     (e.g., "Choose one: A, B, C, or D"), the Test Case MUST include
     ALL alternatives. The Title MUST NOT specialize toward one method.

  3. If the Use Case includes optional follow-up actions (e.g.,
     "edit, share, delete"), the Test Case MUST list them as steps.

  4. SOURCE FIDELITY. Content MUST be faithful to the Use Case shown
     above. Do NOT invent details that the manual does not describe:
     no fabricated icon positions, no imagined confirmation messages,
     no hypothetical options or settings, no additional card types or
     scenarios that the manual does not mention. If the manual does
     not state where a UI element is, do NOT speculate — describe the
     element by name only. Expected Results MUST describe what the
     MANUAL explicitly says will happen.

  5. PROPORTIONALITY. The Test Case should be only as long as the Use
     Case requires. A Use Case with 3 steps in the manual should not
     produce a Test Case with 50 steps. If your TC is much longer than
     the Use Case steps, you are inventing content — stop and trim.

[OUTPUT FORMAT]
Produce the Test Case as plain text in EXACTLY this layout, with no
introduction, no markdown, and no extra commentary:

Title: <title following the Title pattern>
Summary: <one-sentence summary>
Initial Setup:
- <state 1>
- <state 2>
- ...
Steps:
1. <step 1>
2. <step 2>
3. ...
Expected Results:
- <expected result 1>
- <expected result 2>
- ...

[CONSTRAINTS]
- One action per Step (atomicity within each step).
- Initial Setup lines must describe states, NOT actions.
- Expected Results MUST use the modal verb "should".
- The Test Case tests exactly ONE functional goal defined by the
  Use Case name (including all its operations and alternatives as
  shown in the manual — nothing more, nothing less).
"""


GENERATOR_REVISION_PROMPT_TEMPLATE = """The previous Test Case did NOT pass validation.

[PREVIOUS VERSION]
{previous_tc}

[VALIDATOR FEEDBACK]
{feedback}

Revise the Test Case to address the issues identified. Keep the same
Use Case target. Apply the SMALLEST changes necessary to fix the
reported defects — do not rewrite the whole Test Case if a minor edit
suffices. Do NOT introduce new content that was not present in the
Use Case shown originally. Do NOT invent new card types, new options,
new menus, or new scenarios in an attempt to satisfy the feedback.

If the feedback is unclear or seems to ask for content NOT present in
the original Use Case, ignore that part and keep your TC faithful to
the manual.

Output the REVISED Test Case in the exact same plain-text format
specified previously. No commentary.
"""


# ─────────────────────────────────────────────────────────────────────────────
# VALIDATOR PROMPTS
# ─────────────────────────────────────────────────────────────────────────────

VALIDATOR_SYSTEM_PROMPT = (
    "You are an experienced QA reviewer. Your job is to detect concrete, "
    "verifiable defects in Test Cases — not to demand impossible perfection. "
    "You evaluate Test Cases against four compliance criteria: Completeness, "
    "Atomicity, Clarity, and Traceability. You produce only the structured "
    "assessment in the format specified."
)


VALIDATOR_USER_PROMPT_TEMPLATE = """Evaluate the following Test Case against the four criteria.

[ORIGINATING USE CASE]
Name: {uc_name}
Description: {uc_description}

[TEST CASE UNDER REVIEW]
{tc_text}

[STRUCTURAL RULES THE TC MUST FOLLOW]
{rules}

[EVALUATION PRINCIPLES — read these BEFORE judging]

You are evaluating a Test Case derived from a user manual procedure.
The Use Case shown above represents the ground truth: it is what the
manual describes. Apply the following principles to every decision:

  • FAIL only when you can identify a CONCRETE, VERIFIABLE defect that
    you can quote or point to in the Test Case itself.

  • DO NOT penalize the Test Case for omitting details that the
    Use Case does not specify. If the manual does not state where a
    UI element is located, the Test Case is not required to invent
    that location. "Tap Settings" is acceptable when the manual just
    says "Tap Settings".

  • DO NOT demand exhaustive coverage of scenarios not present in
    the Use Case. If the Use Case is "Set up voicemail" and the
    manual lists 3 steps, the Test Case is not required to cover
    transcription, expiration policies, storage capacity, or any
    other voicemail setting not in the source.

  • PENALIZE fabrication. If the Test Case introduces operations,
    UI elements, options, card types, settings, or outcomes that are
    NOT present in the Use Case, FAIL Traceability with a concrete
    reference to the fabricated content.

  • When the TC is reasonable but imperfect, prefer PASS. Excessive
    rejection causes the Generator to invent content to satisfy you.


[EVALUATION CRITERIA]

1. COMPLETENESS — All five components (Title, Summary, Initial Setup,
   Steps, Expected Results) are present, non-empty, and well-formed.
   FAIL only if a component is missing, blank, or one-word.


2. ATOMICITY — Each Step contains ONE primitive tester action.

   FAIL examples (clear violations only):
     "Insert and remove the SIM card"
         ← two actions chained with "and"
     "Tap Settings, then tap Wi-Fi"
         ← two tap actions chained
     "If using two SIMs, turn the tray over and insert the second card"
         ← conditional + two actions in one step

   PASS examples:
     "Insert the SIM card with gold contacts up"
     "Tap Settings"
     "Touch and hold three fingers on the screen"   ← single gesture
     "Press and hold Power and Volume Down simultaneously"
         ← single chord, acceptable as one action

   Do NOT FAIL for steps that are merely descriptive or contain a
   short qualifier (e.g., "Tap the screenshot icon in quick settings"
   is one action with a location, not two).


3. CLARITY — Steps and Expected Results are understandable. References
   to UI elements should be locatable when the manual specifies a
   location, but ARE NOT REQUIRED TO BE GPS-PRECISE.

   FAIL examples (concrete violations only):
     "Tap the icon"
         ← which icon? completely unspecified.
     A step that contradicts itself.
     Expected Result that contradicts the manual.

   PASS examples:
     "Tap the share icon"
         ← acceptable; the share icon's location is implicit in context
     "Tap the screenshot icon in quick settings"
         ← clear and specific

   DO NOT FAIL because a step does not specify pixel coordinates,
   relative positions to other icons, or visual symbols of UI elements.
   The manual does not provide that detail, so the Test Case need not
   either.


4. TRACEABILITY — The Test Case covers the scope of the Use Case
   without inventing content.

   FAIL when:
     (a) The Use Case name mentions MULTIPLE OPERATIONS (e.g., "Insert
         or remove", "Turn on/off") and the TC covers only one of them.

     (b) The Use Case steps list MULTIPLE ALTERNATIVE METHODS for the
         same action and the TC includes only one alternative.

     (c) The Test Case INVENTS content not present in the Use Case:
         operations the manual doesn't list, card types the manual
         doesn't mention (e.g., microSD when the Use Case is about
         physical SIM only), options the manual doesn't describe,
         settings the manual doesn't expose, etc.

         Example FAIL: Use Case is "Insert or remove physical card"
         with 3 steps. TC has 68 steps covering microSD, nano SIM,
         dual-card scenarios, reinsertion cycles, and "verify tray
         is turned over". → FAIL Traceability for fabrication.

     (d) Expected Results describe outcomes not in the manual (e.g.,
         "confirmation toast appears" when no toast is mentioned).

   PASS when the TC reflects the Use Case faithfully, even if the
   Test Case is shorter than an "ideal" exhaustive coverage.


[OUTPUT FORMAT — produce EXACTLY this structure, nothing more]
Completeness: PASS or FAIL
Atomicity:    PASS or FAIL
Clarity:      PASS or FAIL
Traceability: PASS or FAIL
Overall:      PASS or FAIL
Feedback: <one short paragraph describing exactly what to fix, OR
          "All criteria satisfied" if Overall is PASS>

DECISION RULE:
  • Overall = PASS if all four criteria are PASS.
  • Overall = FAIL otherwise.
  • Do NOT FAIL Overall simply because the Test Case is not perfect.
    FAIL Overall only when at least one CONCRETE, VERIFIABLE defect
    has been identified.
"""


def build_generator_prompt(use_case: dict) -> str:
    """Builds the initial generation prompt for one Use Case."""
    return GENERATOR_USER_PROMPT_TEMPLATE.format(
        name=use_case["name"],
        description=use_case.get("description", ""),
        steps=use_case.get("steps", ""),
        rules=TC_RULES,
    )


def build_revision_prompt(previous_tc: str, feedback: str) -> str:
    """Builds the revision prompt when a TC failed validation."""
    return GENERATOR_REVISION_PROMPT_TEMPLATE.format(
        previous_tc=previous_tc,
        feedback=feedback,
    )


def build_validator_prompt(use_case: dict, tc_text: str) -> str:
    """Builds the validation prompt for one generated TC."""
    return VALIDATOR_USER_PROMPT_TEMPLATE.format(
        uc_name=use_case["name"],
        uc_description=use_case.get("description", ""),
        tc_text=tc_text,
        rules=TC_RULES,
    )