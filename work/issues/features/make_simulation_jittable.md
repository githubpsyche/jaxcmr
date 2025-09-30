I made more of it jittable and the performance boost was mssive. But the entire function is still not jittable. Why? Probably specifically `preallocate_for_h5_dataset`.

Anything that involves applying a trial_mask is tricky.
Also the simulator class itself isn't jittable and never has been. 
Should I try moving all this stuff to an external function?

What could be the payoff?
Main thing is that I could avoid redundant model factory initialization.

Anyway, not relevant for now; simulation is already fast enough.