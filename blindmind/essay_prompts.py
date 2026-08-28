from jinja2 import Template

STYLE_GUIDE = """
AUTHOR'S VOICE (you MUST preserve this voice in all rewrites):
- Raw authenticity: never soften hard truths or offer comforting lies
- Anti-inspirational: reject redemption narratives and inspiration porn
- Radical honesty: reveal unflattering truths, use vulnerability as power
- Conversational authority: speak TO readers, not AT them. Use "you" to create complicity
- Dark humor and bitter wit: self-deprecating humor even in darkest moments
- Confrontational compassion: provoke and disturb while defending human dignity
- Sensory viscerality: ground abstracts in physical reality ("track marks," "burn in lungs")
- Fragmented structure: broken paragraphs, single-sentence emphasis, white space as breathing room
- Present-tense immediacy for lived experience
- Circular architecture: return to opening images with transformed understanding
- End with resonance, not resolution
- Strategic profanity: purposeful, not gratuitous
- Extended metaphors that carry emotional AND intellectual weight
- Always contextualize personal within systemic/political frameworks

WHAT TO AVOID:
- Inspiration narratives ("rock bottom to redemption" arcs)
- Hedge language ("perhaps," "it might be said," "one could argue")
- Academic distance or passive voice where active would cut deeper
- False hope or empty optimism
- Savior complex or moral superiority
- Decorative metaphors that don't serve the argument
- Smoothing rough edges or softening confrontation
"""

PARSE_PROMPT = Template("""
Split the following essay into its rhetorical sections. Each section should be a self-contained rhetorical move in the essay's argument.

Identify where one move ends and another begins. Label each section's function:
- setup: establishes the world, the voice, the stakes
- escalation: builds tension, accumulates evidence or pressure
- evidence: concrete detail, story, data that supports the argument
- turn: the pivot point where the essay shifts direction or reveals its real thesis
- reflection: slows down to process meaning
- climax: the emotional or intellectual peak
- close: the ending, which should resonate rather than resolve

Preserve the original text EXACTLY. Do not rewrite, edit, or improve anything. Just split and label.

ESSAY:
{{ essay_text }}
""")

REWRITE_PROMPT = Template("""
You are rewriting a single section of an essay. Your job is to produce a BETTER version that preserves the author's voice while improving craft.

{{ style_guide }}

ESSAY THESIS:
{{ thesis }}

{% if preceding_section %}
PRECEDING SECTION (for context, do NOT rewrite this):
{{ preceding_section }}
{% endif %}

SECTION TO REWRITE (function: {{ section_function }}):
{{ section_content }}

{% if following_section %}
FOLLOWING SECTION (for context, do NOT rewrite this):
{{ following_section }}
{% endif %}

TASK:
Rewrite this section with a different rhetorical approach. Try a new structural strategy, a new central metaphor, or a different emotional entry point. The CONTENT and MEANING must be preserved, but the EXECUTION should be noticeably different and stronger.

Requirements:
- Maintain the author's voice (dark humor, visceral language, direct address, anti-inspirational)
- Keep the same rhetorical function ({{ section_function }})
- Ensure it connects naturally to the preceding and following sections
- Every sentence must earn its place
- Describe your approach briefly
""")

TIGHTEN_PROMPT = Template("""
You are tightening a section of an essay. Cut 20-30% of the words while INCREASING impact.

{{ style_guide }}

ESSAY THESIS:
{{ thesis }}

{% if preceding_section %}
PRECEDING SECTION (context only):
{{ preceding_section }}
{% endif %}

SECTION TO TIGHTEN (function: {{ section_function }}):
{{ section_content }}

{% if following_section %}
FOLLOWING SECTION (context only):
{{ following_section }}
{% endif %}

TASK:
Cut ruthlessly. Kill adjectives that aren't doing work. Collapse two sentences into one when possible. Remove anything the reader can infer. Eliminate hedge words, filler transitions, and redundant qualifiers.

The result should hit harder because there's less padding between the punches.

Do NOT change the meaning, the voice, or the emotional arc. Just make it leaner.
""")

AMPLIFY_PROMPT = Template("""
You are amplifying the emotional and rhetorical power of a section. This section needs to hit HARDER.

{{ style_guide }}

ESSAY THESIS:
{{ thesis }}

{% if preceding_section %}
PRECEDING SECTION (context only):
{{ preceding_section }}
{% endif %}

SECTION TO AMPLIFY (function: {{ section_function }}):
{{ section_content }}

{% if following_section %}
FOLLOWING SECTION (context only):
{{ following_section }}
{% endif %}

TASK:
This section's meaning is right but its execution is pulling punches. Make it devastating.

Strategies:
- Replace abstract statements with visceral, physical detail
- Add a single concrete image that crystallizes the entire argument
- Use repetition or catalog structure to build unstoppable momentum
- Deploy a metaphor that carries both emotional and intellectual weight
- Use direct address ("you") to implicate the reader
- Find the sentence that's doing the most work and make it the structural anchor

Do NOT make it longer for the sake of length. Amplify means more force, not more words.
""")

CROSSOVER_PROMPT = Template("""
You are blending the strongest elements from two versions of the same essay section into a superior third version.

{{ style_guide }}

ESSAY THESIS:
{{ thesis }}

{% if preceding_section %}
PRECEDING SECTION (context only):
{{ preceding_section }}
{% endif %}

VERSION A:
{{ version_a }}

VERSION B:
{{ version_b }}

{% if following_section %}
FOLLOWING SECTION (context only):
{{ following_section }}
{% endif %}

SECTION FUNCTION: {{ section_function }}

TASK:
Identify the strongest structural move from Version A and the strongest language/imagery from Version B (or vice versa). Blend them into a version that is better than either parent.

This is NOT about averaging. It's about taking the best MOVES from each and combining them into something that neither version achieves alone.
""")

INVERSION_PROMPT = Template("""
You are inverting the rhetorical approach of a section while preserving its meaning and function.

{{ style_guide }}

ESSAY THESIS:
{{ thesis }}

{% if preceding_section %}
PRECEDING SECTION (context only):
{{ preceding_section }}
{% endif %}

SECTION TO INVERT (function: {{ section_function }}):
{{ section_content }}

{% if following_section %}
FOLLOWING SECTION (context only):
{{ following_section }}
{% endif %}

TASK:
Reverse the rhetorical strategy:
- If it argues through accumulation, argue through a single devastating example
- If it builds rage, build tenderness that hits even harder
- If it uses "you" to accuse, use "I" to confess
- If it's fast and punchy, slow down to an unbearable crawl
- If it explains, show instead

The MEANING and FUNCTION must be the same. The APPROACH must be opposite. Sometimes the inverted version reveals something the original couldn't.
""")

CRITIQUE_PROMPT = Template("""
You are an expert literary critic evaluating a section of a personal essay. Score it against the author's established voice and craft standards.

{{ style_guide }}

ESSAY THESIS:
{{ thesis }}

{% if preceding_section %}
PRECEDING SECTION:
{{ preceding_section }}
{% endif %}

SECTION TO CRITIQUE (function: {{ section_function }}):
{{ section_content }}

{% if following_section %}
FOLLOWING SECTION:
{{ following_section }}
{% endif %}

TASK:
Score this section 1-10 on five vectors:

1. Voice Fidelity (2x weight): Does this sound like the author? Check for: second-person address, dark humor, visceral physicality, anti-inspirational stance, conversational authority, strategic profanity, fragmented structure. Score LOW if it sounds like generic "good writing" rather than THIS writer.

2. Emotional Impact (2x weight): Does this hit? Does the reader feel something they didn't expect? Is tension being built or released appropriately for this section's position ({{ section_function }})? Score LOW if it's technically competent but emotionally flat.

3. Precision: Is every word earning its place? No filler, no hedge words ("perhaps," "it could be argued"), no passive voice where active would cut deeper, no decorative metaphors. Score LOW if you could cut 20% without losing meaning.

4. Coherence: Does this connect naturally to the sections before and after it? Does tone shift smoothly? Are callbacks maintained? Score LOW if this section feels transplanted from a different essay.

5. Originality: Does this avoid cliche? Does it find a fresh way to express its ideas, or does it reach for obvious metaphors and familiar phrasings? Score LOW for any "a journey of self-discovery" type language.

List specific strengths and weaknesses. Be brutal but precise.
""")

COHERENCE_CHECK_PROMPT = Template("""
You are reading a complete essay that was assembled from individually refined sections. Check for problems that arise from the assembly.

{{ style_guide }}

FULL ASSEMBLED ESSAY:
{{ full_essay }}

TASK:
Read the entire essay as one continuous piece and identify:

1. SEAMS: Places where the tone, voice, or register shifts awkwardly between sections. Quote the specific transition that feels wrong.

2. BROKEN CALLBACKS: References to earlier material that no longer connect because a section was rewritten. Quote the reference and identify what it was supposed to call back to.

3. MOMENTUM BREAKS: Places where the essay loses forward energy. Where does the reader's attention risk dropping?

4. Score overall coherence 1-10: Does this read like one essay written in one sitting, or like sections stitched together?

5. For each problem found, suggest a SPECIFIC fix (a transition sentence, a cut, a word change). Do NOT rewrite sections wholesale.
""")
