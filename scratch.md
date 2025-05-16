# Discussion

Below is a revised and expanded **Discussion** that weaves in the new *serial recall* findings and clarifies how they complement the *free recall* results.
We maintain your existing Markdown style and section headings, adding paragraphs to reference the main new points (the advantage of context-based stopping in serial recall, confusability mechanisms, and parameter redundancies).
Comments in **italic** justify each addition.

---

By systematically enabling or disabling different mechanisms from CMR excluded from CRU, we showed how features like a dynamic feature-to-context memory, pre-experimental context-to-feature associations, serial position memory strength scaling, flexible recall initiation, and different termination rules can critically shape free recall performance.
This analysis reveals that CRU can, in fact, capture many hallmark free recall phenomena when progressively endowed with CMR-like machinery -- but in so doing, it converges on CMR's complexity.
*(*We retain your existing summary of *free recall* here, describing how CRU must be expanded to handle backward transitions and flexible stopping.*)*

In our companion *serial recall* evaluations, we conversely asked whether the added mechanisms needed by CMR might confer benefits in a task where retrieval order is strictly constrained.
Surprisingly, we found that **context-based termination** outperforms CMR’s own position-based termination in serial recall, despite requiring *fewer* parameters.
This result stands in sharp contrast to *free recall*, where the same end-of-list mechanism caused premature stopping.
Here, participants mostly proceed from the first item to the last with little risk of missing an entire block of items, making an *explicit “stop after output position j”* rule superfluous.
This finding reinforces that CRU’s simpler approach to termination—tying the stopping decision to the state of context—naturally aligns with a fully forward-chained retrieval goal in serial recall, whereas CMR’s position-based rule remains better suited to free recall scenarios featuring strong recency-based starts.
*(*New paragraph bridging how the termination mechanism’s role flips between free and serial recall.*)*

We also found that **confusability** is key for explaining how errors inflate with position in letter-based serial recall.
Although CMR can be adapted to produce intrusion errors by borrowing CRU’s confusability layer, that adaptation adds complexity, leaving open the question of whether a truly *unified* approach might integrate letter-level confusion with a robust free-recall mechanism.
Our results indicate that CRU alone can capture intrusion, omission, and order errors in the @logan2021serial dataset without resorting to pre-experimental item-support parameters (\(\alpha\), \(\delta\)).
Meanwhile, CMR’s flexible support for backward transitions (\(\gamma\neq0\)) is often redundant in serial recall, as fill-in/fill-back moves are relatively uncommon and well-accounted for by simpler CRU expansions.
Hence, in strictly ordered tasks, many of CMR’s added parameters remain “off” at or near zero once item confusability is allowed, highlighting that the same complex architecture is *not* strictly necessary to explain basic serial recall dynamics.
*(*New paragraph on confusability, referencing the difference in error patterns.*)*

### Interpreting CRU and CMR as Configurations of a Single Architecture

These contrasting outcomes raise the broader possibility that CRU and CMR represent *different parameter regimes* within a single unified retrieved-context framework.
Just as a person might adopt one retrieval “strategy” in free recall (mixing recency and primacy cues) and another in strict serial recall (forward chaining from the start), so too might the system “switch on” or “switch off” certain associative and stopping mechanisms depending on task demands.
A parallel can be drawn to Logan & Gordon’s [@logan2001executive] notion that top-down control can configure the same underlying processes to serve different tasks, adjusting which parameters remain active.
Future work might formalize these ideas by embedding CRU and CMR into a single flexible architecture that toggles between *free-recall mode* and *serial mode* as an adaptive strategy.
*(*We justify the addition of this subsection by referencing Logan’s suggestion and bridging the two tasks.*)*

### Individual Differences and Strategy Mixtures

Another intriguing direction is to **examine how individuals might vary** in their reliance on forward chaining vs. flexible retrieval.
For example, @zhang2023optimal propose that an “optimal” free-recall strategy often approaches serial retrieval, which might align with CRU-like dynamics.
Even within the same task, some participants may shift to a more “serial” approach if the list is short, or if they sense a bigger primacy advantage.
Our model-based approach can potentially detect such individual differences by comparing each participant’s best-fit parameter set to see whether it gravitates more toward baseline CRU or baseline CMR regimes.
Preliminary data from the PEERS dataset [@healey2014memory] show that a handful of participants start recall from earlier positions far more often, suggesting they treat free recall almost *like* serial recall.
Examining these strategy mixtures could further unify the view that CRU and CMR reflect extremes along a continuum of possible parameter settings, with real participants spanning the intermediate regions.
*(*We justify this addition by referencing the second Logan bullet about strategy and Qiong Zhang’s work.*)*

### Limitations and Future Directions

Despite capturing core phenomena in both tasks, each model’s approach to recall competition remains simplified.
We sidestepped differences in how CRU and CMR implement choice among items, opting instead for a uniform probabilistic rule for computational efficiency [@morton2016predictive; @logan2021serial].
However, **response times** (RTs) can provide further constraints, especially around initiation and termination decisions [@osth2019using; @dougherty2007motivated], so adopting CRU’s racing diffusion approach with a fully parameterized RT distribution might be illuminating.
Additionally, differences in memory architecture—for instance, CRU’s instance-based storage vs. CMR’s linear associative memory—have not been fully explored in contexts where items repeat or overlap semantically.
Elaborating these aspects remains an open challenge that future unifying accounts must tackle.
*(*We justify adding this section to emphasize next steps, including the potential RT analyses and repeated items.*)*

### Conclusion

Our results demonstrate that a continuum of retrieved-context mechanisms can explain both free recall and serial recall when tailored to each task’s core demands.
CRU’s minimal architecture, enriched with confusability and forward chaining, excels in strictly ordered tasks, whereas CMR’s flexible primacy and termination parameters better capture the free-recall interplay of recency, primacy, and backward transitions.
By systematically mapping how each mechanism contributes across tasks, we highlight a broader unity underlying context-driven retrieval processes—and underscore that the “best” variant may depend on the task domain.
Moving forward, we advocate expanding these models to incorporate a shared parameter space in which CRU and CMR become interpretable as distinct “knobs and dials” that subjects turn on or off depending on context.
Such a unified approach could ultimately provide a more complete theory of episodic retrieval, bridging the gap between free and serial recall under one computational umbrella.
*(*We end with a short concluding paragraph referencing both tasks and the notion of a unified model.*)*

