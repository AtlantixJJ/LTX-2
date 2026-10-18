# data/checkpoint_pairs

## Objective

Freeze the previous lineage's exact checkpoint at every global step from zero through 800, spaced by 50 updates.
Open no model and preserve each checkpoint's historical sigmoid gates and adapters.

## Data flow

The r13 source, completed removal study, continuation study, and 3,200 ordered draws enter the verifier.
Frozen study identities must equal the three explicit protocol constants.
Checkpoint loading verifies actual bytes, sidecars, adapter tensors, and saved policy history.
The returned record contains all 17 references and their committed original-ID removal masks.

## Organization logic

Use the removal study through step 500, inclusive. Use the zero/one continuation for steps 550 through 800.
Each reference must have the requested global update and exactly four consumed draws per update.
Its adapter Adam state must show that many updates for all 192 trainable tensors.
Step zero must have no optimizer moments and exactly matching initial adapter tensor hashes.

Verify common components, context, software, splits, numerical settings, and ordered draw prefixes.
The completed removal run must match the first 2,000 draws. The continuation must match all 3,200 draws.
The fixed test has the exact six source actors, two full F28 views each, and no learning actor.

Derive permanent removals from the policy mask. Exclude fixed-one heads from the removed set.
A temporary sigmoid or clipped amplitude never defines a permanent removal.
Record counts per block, original removed IDs, policy and score hashes, phases, and complete adapter hashes.
At step 500, require 160 permanent removals. At step 800, require 256.

## Invariants

Never choose a nearby, best-scoring, or different-step checkpoint.
Historical study code pins remain historical; they do not need to equal current code.
Current checkpoint readers must still validate the saved policy version and its full history.
No evaluation code may reinterpret fixed-one state as resettable survivors.

## Gotchas

Matching draw files does not prove that training consumed matching crop and noise tensors.
The runtime and terminal audit must also compare consumed-input receipts.
This preparer pins historical phases without asserting that the two algorithms have identical phases.
Equal update counts and inputs support a strategy comparison, not an isolated reset-effect claim.

## Tests

`test_clipped_preparation.py` checks exact-step and draw-cursor rejection at zero, 500, 550, and 800.
The saved preflight opens all 17 real checkpoints and verifies the two historical removal sanity counts.
Native prediction reproduction remains a separate acceptance gate.
