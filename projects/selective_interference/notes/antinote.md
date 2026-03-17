What do I need in order to pull this off?

First, I need a good grasp of the experiment design. Sort of. 

It looks like the key things still missing is:

1.  An understanding of how participants interact with the task. It's not simple free recall. How much does that matter?
2. An understanding of what's presented to participants on each trial, both during study and recall phases. How does that affect analyses and modeling?
3. An understanding of how to represent that in my usual dataframes and modeling and analysis code.
4. Differences between VRT and voluntary conditions. 

What's the next document I need?
The updated RR.

> During the VRT, participants perform a “go/no-go” vigilance task that requires pressing a key whenever a single digit is displayed, except when that digit is a ‘3’. 
> 
> Blurred scenes, taken from the films seen previously, appear in the background behind some digits. In the involuntary condition, participants are asked to pause the task (by pressing a key) whenever a “visual image or bodily sensation related to the film pops into mind”. 
> 
> The voluntary condition is matched to the involuntary condition, except participants will be asked to deliberately recall film content and pause the task whenever they succeeded (meeting the “retrieval intentionality criterion”)

So in VRT digits are displayed and participants press a key unless it's "3". For some digits, a blurred scene is displayed. In involuntary condition, pause by pressing a new key whenever something is recalled. In voluntary condition, we ask for deliberate recall.

Key insight: data structure should be identical across conditions! 

Okay, so the next thing to think about is how to figure the cued stuff into the structure. 

===

Notes from Aryan meeting:
Check clip of utterance
And clip of cue presented right before 
Do they match?

Used sentence transformers. 
Line all_miniLm-L6-v2
Line 250
10% confused to wrong clip.
Ask for poster or other reference.

External details are an issue. 
They do not match to any clip but will be matched to one by default.
Taguette is applied after raw data!

AI-steered baguette scoring can be done directly.
Classify between internal and external detail.

match_status: whether cue background matches the assigned clip!

Tasks from after summer:
Focus on segmenting. Within-clip matching? Identifying hotspots?
Extracting semantic features from clips. 
Stuff like arousal, valence, threat?

Cite conference poster!

