> Sometimes we need to add more operations and make sure they work properly on all types; sometimes we need to add more types and make sure all operations work properly on them. Sometimes, however, we need to add both - and herein lies the problem.

Besides the issues of speed and correctness, my ultimate goal for this codework is a library that is easily extended to produce novel variations of CMR while reusing as much code as possible.
This goal corresponds to the expression problem that has long been central to the design of programming languages.

The key limitations of class-based object-oriented programming are that methods aren't **open** -- they're bound to classes, and classes are bound to packages. 
This means that you can't add new methods to existing classes without either editing the original code or performing a form of inheritance that isolates the new methods from the original class.

Multiple dispatch is relatively well-known as a solution to the expression problem. 
The talk "The Unreasonable Effectiveness of Multiple Dispatch" by Stefan Karpinski and Viral Shah is a good introduction to the topic.
In the talk, they review how multiple dispatch facilitates two kinds of code reuse: generic functions that can be applied to many different types and circumstances, and common types that can be shared by very different packages.

The kind of extension I'm looking to do is not quite to add new functions to a pre-existing type. And while I definitely seek to apply pre-existing functions to a new type, I also want to add new functions to that new type.
This new-for-new thing, though, is probably not a relevant feature of the expression problem.

The recipe seems to be:

1. Define new types to which existing operations can be applied via multiple dispatch
2. Add methods to existing functions that work in a specialized way for the new type

An itch I wish I could scratch better in this case is that I still feel like I'm doing a sort of broken inheritance when my new type isn't *really* a subtype of the original type whose methods I'm extending.

For example, I'm implementing variants of CMR that work in a similar way to the original, but some of the details are different.
These variants are not really subtypes of the original CMR, but rather alternative implementations of the same interface.
But I think that's fine for prototyping purposes.
When I find a model variant that works well and is sufficiently interesting, I can give it a more formal implementation.

An important realization for later is that it's type inheritance that's the problem, not an issue with how multiple dispatch handles novel subtypes.

The specific thing I want to confirm here is that I really can create model variants outside stable jaxcmr code and have them work with the existing code. For example, my data simulation code uses jaxcmr's `experience` function. If I add a new method to `experience` that works with my new model, will the simulation code work with it? 

A related question is whether I can debug and test external model variants implemented in this way. I think the answer is yes, but I'm not sure.

I'll test this by adding a model variant that modifies CMR in some simple but obvious way. I'll then run the simulation code and see if it works. Then I set a breakpoint in the new method and see if it gets hit.

Great. `test_model_variation.py` proves that I can add a new method to `_retrieve_item` and have it work with a new model type without modifying the original code or sacrificing debugging capabilities.

It seems that an interesting limitation is that I probably have to remove jit decorators from the original code in order to perform the external extension. That's fine; these are overriden in practice anyway.