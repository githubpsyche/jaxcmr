I want to write a specific aims for my current postdoc and beyond. 

What are the constraints to satisfy?

Let's just enumerate and then organize...

## Retrieval Intentionality
Our project should aim to characterize the difference between voluntary (strategic) and involuntary (spontaneous) recall within an (emotional) retrieved-context model.

It's apparently contentious whether there really is a difference, or at least the extent of the difference is ambiguous. 

On the one hand, it's frequently implied that involuntary recall is spontaneous, un-cued -- not just unintended, but arising without basis in an ongoing retrieval cue and thus through a separate mechanism.

On the other hand, some models deliberately argue against including mechanisms distinguishing involuntary and voluntary recall, arguing that evidence that such recalls ever happen is rare and implying that involuntary recalls are in fact probably always cued to some extent by mental context. In other words, they draw no distinction between involuntary and voluntary recall in their mechanistic frameworks at all. 

In the middle is room for a position where there are shared mechanisms (e.g., retrieval based on contact with an internal contextual cue) but also dissociated mechanisms such as a control mechanism that filters retrievals under strategic recall but which can fail to suppress non-strategic recalls -- such as the proposal specified in CMR2 for monitoring retrieval and suggested in Talmi's grant as a candidate mechanism in prospective model of retrieval intentionality. 

Such framing gives room for an integrated account of voluntary and involuntary recall that nonetheless acknowledges and addresses differences between them.

## eCMR / CMR3
Our project must build on work done in papers like CMR3 and eCMR clarifying constraints and predictions for models of episodic memory and their interaction with emotion. 

There are many ways suggested to advance beyond CMR3, but we also seek to challenge it's approach for conceptualizing the difference between voluntary (strategic) and involuntary (spontaneous) recall. In CMR3's presentation, intrusions are actually the result of a spontaneous, unintended process that
arises as a byproduct during the subject’s intentional retrieval search.

> Although subjects may experience voluntary (strategic) and involuntary (spontaneous) autobiographical memories as distinctive processes, literature suggests that both processes share and operate through the same mechanisms in episodic memory (Berntsen, 2010). For example, in PTSD, when patients report experiencing a spontaneous memory intrusion, further investigation typically reveals that the memory was in fact cued by a temporal, spatial, or other contextual cue associated to the trauma (Ehlers & Clark, 2000), just as contextual cues guide recall of other episodic memories. Additionally, both in the laboratory and in daily life, even healthy subjects spontaneously recall neutral memories that are unintended and undesired, highly vivid, and which they may not recognize as being an intrusion (Berntsen & Rubin, 2002; Berntsen, 2010; Staugaard & Berntsen, 2014; Zaromb et al., 2006; Kahana et al., 2005; Roediger & McDermott, 1995).

On the other hand, CMR3 has CMR2's mechanisms, though, providing a post-retrieval recognition mechanism that plays an important role in monitoring potentially intrusive memories. So connecting to that spectrum of accounts of the voluntary/involuntary recall distinction, I feel like the CMR3 paper (if not the model) is a little ambiguous about where it stands.

Much of what I've considered for follow-up work only aims to either generalize CMR2's mechanism to account for flexible task specifications, or to additionally address encoding with a similar mechanism aimed at the same sort of performance optimization. Finally, I've been interested in developing an account of how these mechanisms work differently not just between tasks, but as participants gain experience with a task.

But Deborah seems dissatisfied with CMR3's treatment of voluntary and involuntary recall. I need to figure out if the dissatisfaction is with the language in the paper, or with the model, and resolve the tension.

One possibility is that we should aim to specify a model and constraints that articulate the hypothesis that involuntary memories are encoded and/or retrieved differently from voluntary memory. Then we can compare to other accounts that limit the dissociation. So on one end is a "separate route model" that can support retrieval of items in a manner detached from an internal contextual cue. A moderate account is a meta-cognitive control mechanism applied over shared representations. And an extreme account excludes a meta-cognitive control mechanism completely, probably equivalent to giving a higher baseline probability of retrieving an item independent of the retrieval cue (an item-specific alpha parameter in MCF?). Can these models truly be distinguished with the data we have? 

## Reward-Dependent Memory
First, there's a persistent assumption that rewarding items are processed similarly to emotional items, perhaps because rewards evoke emotional responses by giving concrete, personal stakes to encoding. 

Is interesting that they do this without necessarily adding semantic content to the item beyond this, perhaps providing a route to experimental control (at sacrifice of definitive generality).

Second, rewards configure what voluntary/strategic recall looks like (or at least should look like).

## Intrusive Memory
Definition: Involuntary and unwanted, difficult to dismiss, often distressing; carries a sense of “nowness” (vivid sensory re-experiencing).

To some extent we want to address how intrusive memories might emerge. We may not be able to address all these features. 

We can treat intrusions in a memory task as a stand-in, especially if the intrusions are more likely to be highly emotional items than to be neutral items.

Intrusions also relate to retrieval intentionality because retrieval intentions determine whether a recall event is involuntary as well as unwanted.

## Multi-trial learning
As modeled in CMR2, multi-trial learning provides a way to explore how past experience configures ongoing encoding and retrieval.

We are interested in it for producing a lab model of how involuntary and unwanted items can intrude on recall performance despite retrieval intentionality.

## Optimal Policy for Free Recall / Cognitive Constraints and Reward Environments Jointly Shape Memory Formation

Given a task goal and other conditions and cognitive constraints, these papers lay out a straightforward procedure and modeling framework for identifying optimal behavior. 

This is relevant to a project focused on retrieval intentionality because retrieval intentions define what [successful] "strategic" recall looks like in contrast to unregulated or spontaneous recall.

## Free Recall / Serial Recall / Externalized FR / Item Reward / Suppressed Recall
These study/recall paradigms aren't substantially different from my primary paradigm of free recall but interfaces with retrieval intentionality because they provide distinct specifications for what strategic recall looks like because they change what optimal performance looks like. 

Strategic recall depends on task specification as well as the timing of when that specification was provided (e.g., before or after encoding). 

We can address retrieval intentionality within the constraints of a single task (and this seems a great way to scope an initial paper), but a complete account must be flexible across task specification. I should avoid overgeneralizing too early though!

Data from Talmi lab also gives an opportunity to study recall when task instructions are specifically aimed at suppressing recall, as well as manipulations that might affect how participants succeed at this goal.

## Response Termination (and Latency?)
Models of optimal performance are highly contingent on theoretical commitments about when retrieval stops. Accounts of when retrieval stops are often highly interconnected with accounts of when retrieval is delayed. Existing mechanisms as implemented in CMR seem deficient. This seems a great opportunity for further, piecemeal study that advances the broader research agenda.

## EEG / Neurocognitive modeling

I will spend a lot of time developing a neurocognitive model linking CMR to EEG measurements during an experiment where participants study a sequence of items that can be emotional or neutral. There's also eeg/recall data manipulating item rewawrd in a free recall study. The broad approach is to allow EEG measurements during study and/or retrieval to configure CMR parameters during fitting or during simulation from study/recall event to study/recall event. If this model enhanced with brain signals provides better predictions or better fit than an uninformed model, we report that the underlying brain operations producing the signals represent information related to the parameter.

It would be best if a proposal didn't cleanly separate the EEG modeling work I'll do from this other modeling work I'll do. But since I am looking to propose new modeling and don't yet know the relevant EEG signals that I could tie to my model, this seems a bit downstream. At this time, I need to develop the neurocognitive model without other departures from the eCMR or CMR3 framework.

## Item-Independent Temporal Context
My thesis advances the claim and a set of results/methods evincing that states of temporal context evolve independently of the items encoded. I am intensely interested in testing the generality of these claims with respect to multi-list learning and to study event pairs that are not item repetitions but instead just experientially similar (e.g., similar word meaning or same-emotionalitty). This has clear ties to potential modeling of intrusions during sequence recall and their intersection with item emotionality.

## Where am I going with this?
For a one-year project where I won't collect new data, these constraints can be addressed with a multi-paper modeling effort where my specific aims are...?

Well, it's clear that the optimal policy stuff is only relevant to the extent that we want to 1) address how strategic recall might look differently across task specifications, and 2) specify a meta-cognitive control mechanism that regulates encoding and/or retrieval to address our findings. It looks like even CMR2. 
