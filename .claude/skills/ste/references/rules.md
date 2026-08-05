# The 9 rule categories

Each category below has the rule, the reason for it, and a before/after example. The
examples come from this project: the 12 servos, the pushrod knee, the three IMUs, the
potentiometers, and the training runs.

---

## 1. Words

**Use an approved word in its approved meaning only.** Most English words have many
meanings. STE gives each approved word one meaning and one part of speech. "Fit" is a
verb, so you can fit a pot to a joint. "Fit" is not an adjective, so a part is not "fit
for the job".

**Use one name for one thing.** Do not change "servo" to "actuator" to "motor" in one
paragraph. The reader must not ask if the three words are three different parts.

- Before: The actuator drives the shank. This motor is a DS3218MG.
- After: The servo drives the shank. This servo is a DS3218MG.

**Do not use a technical name that you did not define.** The first time you use a name,
say what it is.

---

## 2. Nouns and noun clusters

**Do not put more than 3 nouns together.** A long cluster hides which noun is the subject.
Break the cluster with "of", "for", "in", or "that".

- Before: policy observation space joint angle feedback limit
- After: the limit of the joint angle feedback in the observation space of the policy

**Give a name to a thing, not a process.** Write "the calibration of the pot", not "the pot
calibration procedure operation".

---

## 3. Verbs

**Use the active voice.** The active voice says who does the thing. The passive voice hides
it, and the reader must then guess.

- Before: The queue is read by the runner and the job is started.
- After: The runner reads the queue and starts the job.

**Use only the simple tenses: past, present, and future.**

- Before: The run had been stopped before the checkpoint was written.
- After: The run stopped. The script did not write the checkpoint.

**Do not use an `-ing` form as a noun or as a verb.** The `-ing` form is permitted only in
a technical name that is already in use (for example, "a ball bearing").

- Before: Training the policy needs 5000 robots.
- After: The policy needs 5000 robots to train.

**Use the imperative for an instruction.** Write "Start the runner", not "The runner should
now be started".

---

## 4. Sentences

**Keep a procedural sentence to 20 words or less. Keep a descriptive sentence to 25 words
or less.**

**Write one instruction in one sentence.** If a step has two actions, write two sentences,
or write two steps.

- Before: Open the second terminal and start the runner, but do not open the viewer.
- After:
  1. Open the second terminal.
  2. Start the runner.
  3. Do not open the sim viewer.

**Keep the articles.** "The", "a", and "an" tell the reader if the thing is new or known.
Do not remove them to make the text shorter.

- Before: Set weight in config file.
- After: Set the weight in the config file.

**Use a connecting word to show the relation between two clauses.** Use "because", "if",
"but", and "and". Do not join two clauses with a comma only.

---

## 5. Procedures

**Write the steps in the order that the user does them.**

**Start each step with the verb.**

- Before: The `run.json` file must first be checked for the reward weights.
- After: Check the reward weights in the `run.json` file.

**Give one step for one action.** If the reader must do 3 things, give 3 steps.

**Say the condition first, then the action.**

- Before: Stop the run if the reward does not increase after 500 iterations.
- After: If the reward does not increase after 500 iterations, stop the run.

---

## 6. Descriptive writing

**Keep a paragraph to 6 sentences or less.**

**Give each paragraph one topic. Say the topic in the first sentence.**

**Divide long text with headings.** A heading tells the reader what is below it, so the
reader can find the part they need.

**Do not put more than one idea in one sentence.**

- Before: The knee is a pushrod linkage, so the servo angle is not the joint angle, which
  is why a linear pot on the rod is necessary, because it measures the true angle.
- After: The knee is a pushrod linkage. Thus the servo angle is not the joint angle. A
  linear pot on the rod measures the true joint angle.

---

## 7. Warnings and cautions

**Write the warning or the caution before the step it applies to.** A warning after the
step is too late.

**Start the warning with a clear command or a clear condition.**

- Before: Damage to the machine can occur as a result of opening the sim viewer while
  training is in progress.
- After:
  > **WARNING:** Do not open the sim viewer while a training run is active. A second CUDA
  > process on the 12 GB card can stop the machine and kill the run.

**Say what can occur, and say how to prevent it.**

---

## 8. Punctuation

**Use the hyphen only to make a compound word clear.** Do not use it to join words that do
not need it.

**Do not use a slash (`/`) to mean "and" or "or".** Write the word.

- Before: Check the thigh/calf joint.
- After: Check the thigh joint and the calf joint.

**Use a colon to introduce a list.**

**Do not use parentheses inside a sentence to add a second thought.** Write a second
sentence.

**Do not use an abbreviation that you did not define.** Write the full name first, then the
abbreviation in parentheses: "analog-to-digital converter (ADC)".

---

## 9. Writing practice

**Do not write text that only fills space.** Remove "it is important to note that" and "as
you can see".

**Do not use a metaphor, an idiom, or a joke.**

- Before: The reward weights are a moving target, so the dashboard is chasing its tail.
- After: The reward weights change often. The dashboard shows old values.

**Do not use "etc." or "and so on".** Give the full list, or say how many items there are.

**Give the exact number.** Write "12 servos", not "a number of servos".

**Write what is true.** If a test failed, say that it failed, and show the output.
