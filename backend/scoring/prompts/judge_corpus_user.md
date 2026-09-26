# Topic

Title: {{topic_title}}

<fixed_content>
{{fixed_content}}
</fixed_content>

# Corpus ({{pool_size}} submissions)

<existing_submissions>
{{existing_submissions}}
</existing_submissions>

# Task: judge EVERY submission in the corpus

For EACH submission above, treat it as the new submission and ALL OTHER
submissions as the existing pool. Never compare an item with itself: if two
items say the same thing, BOTH are paraphrases of each other and both should be
scored accordingly.

Apply the method, scales, anchors and caps from your instructions to each item
independently, as if it were the only one being judged. Do not grade on a
curve across the corpus; use the absolute anchors.

Return exactly one entry per submission, using its id, covering all
{{pool_size}} ids. Keep each rationale to one short sentence.
