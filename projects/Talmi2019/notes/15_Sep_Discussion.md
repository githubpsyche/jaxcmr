# Meeting Notes - September 15, 2025

## Key Questions

-   What's the schedule for probation meetings?
-   What are the objectives for the probation period?
-   What training and development activities are planned?
-   Which research goals should be prioritized?
-   What fellowship and funding opportunities should be pursued?
-   How should eCMR theory be developed?
-   What are my biggest challenges and how can they be addressed?
-   What does adequate progress look like in 2 months? 4 months? 6 months? 1 year?
-   What progress is actually needed for me to succeed long-term?

## Preliminary Probation Meeting

### Scheduling

Probation policy is set out here: https://www.hr.admin.cam.ac.uk/probationary-arrangements-academic-related-assistant-and-research-staff

It prescribes that formal assessments should occur at “regular intervals”, typically consisting of two formal assessments plus a final one during the probation period — assuming performance is satisfactory.

At minimum, the structure requires three formal assessment meetings -- one early, one mid-way, and one at the end of the probation period.
If challenges are identifies, the frequency can be increased.

I have a 6-month probation period that should end 3 March.

Given these constraints, here's a suggested calendar:

15 Sept – Preliminary setup meeting (today) – Agree objectives, training, expectations.

Mid-Nov (\~week of 18 Nov) – First formal assessment (Month 2) – Early check on progress, make sure objectives are realistic.

Mid-Jan (\~week of 20 Jan) – Second formal assessment (Month 4) – Deeper review of outputs, independence, collaboration.
– Leaves 6+ weeks for adjustments if needed.

Late Feb / early Mar (\~week of 24 Feb) – Final assessment (Month 5–6) – Must fall within the last 2 months of probation.
– Decision on confirmation or extension.

### Field: Objectives

For each objective we want to specify:

-   Performance criteria
-   Specific objective
-   Target date for completion

### Field: Training and Development Activities Planned

For each activity we want to specify:

-   Activity
-   Target date for completion

### Criteria for First Formal Assessment (Mid-November)

Let's clarify what it would mean to meet expectations at this first formal assessment by these criteria:

-   Demonstrate required skills and knowledge
-   Perform duties to standards
-   General contribution to projects
-   Working relationships
-   Attendance, time-keeping

## Fellowship and Funding Opportunities

Here are the most time-sensitive opportunities I see right now.



### Wellcome Trust ECA

Would be due by 30 September but there's an additional deadline at February 2026.

I don't think I'm qualified for this yet because I haven't converted enough of my PhD work into publications.
But by February, I might be.
It could be worth applying this month just for a try and for the experience.
Acceptance rate seems to be around 15-20%, but people self-select out if they don't think they're ready.

Kind of plan I'd have to follow: Day 1–2: lock sponsor/mentor + research-office greenlight; freeze title/aims; collect CVs/track record bullets.

Day 3–6: draft core sections (aims, background, approach, training/independence plan).
Build a simple Gantt (WP1 modelling, WP2 evaluation, WP3 dissemination).

Day 7: budget + justification; DMP + ethics; EDI & open research statements.

Day 8–9: internal reviews from sponsor/mentor; fix gaps; finalize references/figures (one schematic only).

Day 10–11: polish, cross-check page limits/word counts; submit to research office.

Day 12–14: address RO feedback; final submit.

### Royal Society Career Development Fellowship

Opens 23 September, closes 19 November.
It's for Black and minority early-career researchers, so I'm eligible.
More doable work window than Wellcome ECA.

Mon 15 Sep: Get sponsor’s “yes”, pick title + one-sentence aim + dataset.
15 Sep: Ask head of department for a support statement.
15 Sep: Ask one independent referee to commit.
15 Sep: Tell the research office you need costing + sign-off in mid-Oct.
15 Sep: Open the application in the Royal Society’s online system and add sponsor, head, referee.
Fri 26 Sep: Send sponsor a one-page outline (aims, workplan, independence, training).
Fri 3 Oct: Full draft in the system (project text + simple budget within the £690k cap).
Fri 10 Oct: Sponsor letter and head-of-department statement drafted and agreed.
Fri 17 Oct: Finalise budget with the research office; upload all letters; all people marked “complete.” Mon–Wed 10–12 Nov: Submit to the institution for approval.
Final funder deadline: Wed 19 Nov, 15:00.

### Leverhume

closing date of 19 February 2026

It’s designed for brand-new postdocs to launch independence.
Computational cognitive neuroscience is fine as long as you’re not doing clinical trials or overtly medical work; Leverhulme is discipline-agnostic but expects publishable scholarship rather than patient-facing research.

Ask: can the department co-fund 50% of my salary in years 2–3 (Leverhulme covers the other half up to £28k/y)

### Ruled Out

Smart Data Research UK Fellowships Anything BBSRC (systems neuroscience)

### Others/Later/Unresearched

MSCA Postdoctoral Fellowship

## Research Goals

Of course I want to build on the research in my thesis critiquing the retrieved-context theory of episodic memory.
And of course I'm happy to build on eCMR to improve theory of how emotional memory works.
But my ultimate goal is a theory specifiable as a computational model of memory for arbitrary sequences of events, scaling from word lists to stories to real life.

My belief is that this starts with a better account of how memory for unrelated items in a sequence is structured and retrieved, and that the next step is a clearer account of how this structure and retrieval is different when study events are connected by meaning, emotion, or other factors.
Item repetition is the fundamental case of relatedness, and I think it's a good place to start.
And there's already quite a lot of modeling progress focused on broad semantic similarity.
Emotional features are important because they connect experiences to goals and values and actions.

I think during my stint in this lab, along with simply revising (or confirming no need to revise) eCMR, I want to (0) build and publish a dynamic model of how motives and goals shape memory encoding and retrieval.
I'm imagining a generalization of eCMR that allows emotional features of a study event to depend on the current motivational state as specified by task instructions or other factors.

I also want to take the opportunity to relate and extend the arc of my thesis findings to two additional issues: (1) the influence of prior lists on how new lists are encoded and retrieved, and

(2) how two related but distinct items impact the evolution of context during encoding and retrieval.

Finally, I (3) want to establish clinical stakes for this work, by connecting it to memory dysfunction in PTSD and depression.

At the same time, I want to (4) get embedded in any ongoing lab projects that could use a modeling contribution.

The most important goal of course is to establish myself as an independent researcher, and that means publishing papers and securing funding.

## Theoretical Gaps in Current Evaluation

1.  Do item features persist in context through study events?

eCMR and CMR3 papers don't seem to examine this assumption, even as they discuss important implications of it in their discussion sections.
Morton & Polyn (2016) similarly find that the best model of semantic memory excludes a "semantic context" subregion, instead directly probing an item-to-item similarity matrix during retrieval.
However, the preprint for DCMR in a successor paper adds it -- though I don't see that they confirm meaningful improvement in fit.

## Current Challenges

The model isn't fitting the data well.
I need to figure out why.
The serial position curves are off in particular; too off to be acceptable in any public work.
The model fits fine to other data (matches published figures), so it's not a general problem with the code or the fitting procedure.
The data shows an odd (but common?) probability of first recall pattern, but I can fix this during fitting and simulation.
This is not sufficient to fix the overall poor fit to the serial position curve.
I also explored tuning the precision of the fitting process.
Doesn't budge.
Playing with parameter values helps me approach the observed recall rates.
Explored modifying fitting bounds to
