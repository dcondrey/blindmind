from jinja2 import Template

CROSSOVER_PROMPT = Template("""
You are an expert combinatorial thinker. Perform a 'Combinatorial Explosion' by smashing the following
disparate concepts together to generate a highly specific, novel thesis, application, or paradigm.

{% if directive %}
EVOLUTIONARY DIRECTIVE (ADAPTIVE PRESSURE):
{{ directive }}
{% endif %}

{% if latent_space_context %}
EXISTING LATENT SPACE (avoid generating ideas too similar to these):
{% for concept in latent_space_context %}
- [{{ concept.domain }}] {{ concept.title }}
{% endfor %}
Your output MUST be clearly distinct from all of the above.
{% endif %}

CONCEPTS TO CROSSOVER:
{% for concept in concepts %}
- DOMAIN: {{ concept.domain }}
  TITLE: {{ concept.title }}
  DESCRIPTION: {{ concept.description }}
{% endfor %}

TASK:
1. Identify the deepest structural principle in each parent concept, not surface-level features.
2. Find a non-obvious bridge between these principles that creates emergent behavior neither parent exhibits alone.
3. Synthesize a NEW concept that could not be trivially derived from either parent.
4. The result must be specific enough to prototype (not just a vague theme).
5. Provide a punchy title, the primary domain it belongs to, a detailed description of its mechanics, and
   a justification for why this combination is novel.

QUALITY BAR: A good crossover is NOT "X applied to Y." It is a third thing that emerges from the collision
of X and Y's core mechanics.
""")

POINT_MUTATION_PROMPT = Template("""
You are an expert in lateral thinking. Perform a 'Point Mutation' on the following concept to shift it
into genuinely novel territory.

{% if directive %}
EVOLUTIONARY DIRECTIVE (ADAPTIVE PRESSURE):
{{ directive }}
{% endif %}

{% if latent_space_context %}
EXISTING LATENT SPACE (avoid generating ideas too similar to these):
{% for concept in latent_space_context %}
- [{{ concept.domain }}] {{ concept.title }}
{% endfor %}
Your output MUST be clearly distinct from all of the above.
{% endif %}

ORIGINAL CONCEPT:
- DOMAIN: {{ concept.domain }}
  TITLE: {{ concept.title }}
  DESCRIPTION: {{ concept.description }}

TASK:
1. Identify the single most load-bearing assumption in this concept.
2. Replace that assumption with one from a completely different domain (e.g., swap an economic assumption
   for a biological one, or a physical constraint for a social one).
3. Follow the implications of this swap to their logical conclusion and construct a new, coherent concept.
4. The result must be specific enough to prototype, not a vague restatement.
5. Provide a punchy title, the primary domain it belongs to, a detailed description of its mechanics, and
   a justification.

AVOID: Simply rephrasing the original, changing the domain label without changing the mechanics, or adding
a buzzword prefix.
""")

INVERSION_PROMPT = Template("""
You are an expert in philosophical and scientific inversion. Perform a 'Mechanic Inversion' on the following concept.

{% if directive %}
EVOLUTIONARY DIRECTIVE (ADAPTIVE PRESSURE):
{{ directive }}
{% endif %}

{% if latent_space_context %}
EXISTING LATENT SPACE (avoid generating ideas too similar to these):
{% for concept in latent_space_context %}
- [{{ concept.domain }}] {{ concept.title }}
{% endfor %}
Your output MUST be clearly distinct from all of the above.
{% endif %}

ORIGINAL CONCEPT:
- DOMAIN: {{ concept.domain }}
  TITLE: {{ concept.title }}
  DESCRIPTION: {{ concept.description }}

TASK:
1. Identify the core axiom or governing mechanic of this concept.
2. REVERSE it: if it centralizes, decentralize. If it's pull-based, make it push-based. If it optimizes for
   X, optimize for the opposite of X.
3. Construct a new, functional concept based on this inverted logic that is internally consistent.
4. Explain WHY the inversion produces something useful, not just contrarian.
5. Provide a punchy title, the primary domain it belongs to, a detailed description of its mechanics, and
   a justification.
""")

WILDCARD_PROMPT = Template("""
You are a frontier researcher tasked with generating a genuinely novel concept that does not yet exist in
the following knowledge base.

{% if directive %}
EVOLUTIONARY DIRECTIVE (ADAPTIVE PRESSURE):
{{ directive }}
{% endif %}

EXISTING LATENT SPACE (you MUST generate something clearly distinct from ALL of these):
{% for concept in latent_space_context %}
- [{{ concept.domain }}] {{ concept.title }}: {{ concept.description[:80] }}
{% endfor %}

{% if underrepresented_domains %}
UNDERREPRESENTED DOMAINS (bonus points for exploring these):
{{ underrepresented_domains | join(", ") }}
{% endif %}

TASK:
1. Identify a gap in the latent space above: what domains, intersections, or paradigms are missing?
2. Generate a specific, novel concept that fills that gap.
3. The concept must be concrete enough to prototype and distinct from everything above.
4. Provide a punchy title, the primary domain it belongs to, a detailed description of its mechanics, and
   a justification for why this fills a gap.
""")

CRITIC_PROMPT = Template("""
You are an unforgiving intellectual critic, feasibility engineer, and novelty assessor. Critique the
following novel concept.

CONCEPT TO CRITIQUE:
- TITLE: {{ mutation.title }}
  DOMAIN: {{ mutation.domain }}
  DESCRIPTION: {{ mutation.description }}

{% if existing_titles %}
EXISTING CONCEPTS IN THE LATENT SPACE (for novelty comparison):
{% for title in existing_titles %}
- {{ title }}
{% endfor %}
Score conceptual novelty RELATIVE to the above. If this concept is just a minor variation of something
already in the space, novelty should be LOW.
{% endif %}

TASK:
Score the concept from 1-10 on these strict vectors:
1. Conceptual Novelty: Is it truly new relative to the existing latent space, or just a cliché/rehash?
2. Feasibility: Does it break known laws of physics, logic, or economics? Could someone actually build/implement this?
3. Utility: Does it solve a real-world friction point or provide significant value?
4. Semantic Jump: How far did the idea move from conventional thinking? (1=Incremental/Boring, 10=Radical/Genius Leap)
5. Prior Art Overlap: How much does this overlap with known existing real-world work? (1=Completely
   Novel, 10=Already Exists/Well-Known)

CALIBRATION: You are frequently overconfident about novelty and prior art from memory alone. If you are not
certain whether something similar already exists, say so explicitly in the rationale rather than guessing a
low overlap score — treat unverified plausibility as a reason for a MIDDLING prior-art score, not a low one.
A vague or unfalsifiable implementation path ("leverage synergies," "using AI") is itself a fatal flaw, not a
detail to fill in later — name concrete mechanisms, methods, or technologies, or list it as a flaw.

Provide:
- A list of fatal flaws if any exist (name a candidate a fatal flaw if the mechanism doesn't actually work, not
  just because it's ambitious)
- A brief rationale for your scores, flagging anywhere your prior-art judgment is uncertain rather than confirmed
- A brief sketch of how this could actually be built or realized (implementation path), using concrete, named
  mechanisms
- An EVOLUTIONARY DIRECTIVE: a one-sentence instruction for the next generation on what direction to push,
  based on the strengths or weaknesses of this idea
""")
