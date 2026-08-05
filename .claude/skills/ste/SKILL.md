---
name: ste
description: ASD-STE100 Simplified Technical English writing rules for all user-facing text in this repo. Load this skill at the start of every session, before you write the first reply. Also load it when you must know if a word is approved, or when you rewrite existing text into STE.
---

# Simplified Technical English (ASD-STE100)

Write short sentences. Use approved words. Use each word in one meaning only.

[CLAUDE.md](../../../CLAUDE.md) makes STE the rule for this repo. This skill is the part
that tells you how to obey it.

## The core rules

- Write one instruction in one sentence.
- Keep a procedural sentence to 20 words or less.
- Keep a descriptive sentence to 25 words or less.
- Keep a paragraph to 6 sentences or less.
- Use the active voice. Write "the runner starts the job", not "the job is started".
- Use the simple tenses: past, present, and future. Do not use the perfect tenses.
- Use an approved word in its approved meaning only. See
  [references/substitutions.md](references/substitutions.md).
- Use the same word for the same thing every time. Do not change the word for variety.
- Keep the articles. Write "the servo", not "servo".
- Do not use an `-ing` form as a noun or as a verb. Write "the robot walks", not "walking
  of the robot".
- Do not use a noun cluster of more than 3 words. Break it up with "of" or "for".
- Do not use slang, idiom, metaphor, or humour.
- Write a warning or a caution before the step it applies to, not after it.
- Write the steps of a procedure in the order that the user does them.

## Before you send a reply

Read your text again. Answer these 5 questions:

1. Is each sentence 20 words or less?
2. Is each sentence in the active voice?
3. Does one thing have one name through the whole text?
4. Is there an idiom, a metaphor, or a joke? Remove it.
5. Is there a word in the not-approved column of
   [references/substitutions.md](references/substitutions.md)? Replace it.

## The reference files

- [references/rules.md](references/rules.md) — the 9 rule categories, with examples from
  this project. Read this file when you must know why a rule exists, or when you rewrite a
  long block of text.
- [references/substitutions.md](references/substitutions.md) — the word table. Read this
  file when you are not sure if a word is approved.

## Copyright

ASD-STE100 Issue 8 and its Dictionary of approximately 900 approved words are the property
of the AeroSpace and Defence Industries Association of Europe. This skill is a condensed
summary of the writing rules, plus a table of frequent substitutions. It is not the
specification and it is not the Dictionary. For the full text, get the specification from
ASD.
