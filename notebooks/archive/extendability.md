The big benefit of multiple dispatch is supposed to be the ability to add specialized functions to an external library without otherwise requiring a broader understanding of how the library works.

For example, I add a version of a function that only runs when an argument has a specific value. Library functions that depend on that function will automatically use the specialized version when appropriate.

But how will I apply that for modeling? How should I organize my library to make this sort of extension easy?

I already have in mind a base implementation of CMR. Following conventions established for Julia, I follow interface inheritance rather than implementation inheritance. Using accessor functions instead of field access will help with this.

Once I have the version of CMR implemented that uses linear associative memory, I'll want to make variants that use other constructs. For example, for ICMR, I'll want to replace uses of LinearAssociativeMfc with uses of InstanceMfc. For SemanticCMR, I'll want type instances to additionally include a semantic memory field and add an specialized version of the function that performs item recall.

What will happen if I want an instance-based semantic cmr? Would I have to implement another version of the type? Of the function? No, I think I just have to keep the mfc and mcf fields generic like I'm doing for CMR. I'll add an initialize_semantic_icmr that creates an instance of SemanticCMR with an InstanceMfc.

Every variant of CMR will indeed have its own type, and new versions of functions wherever they behave differently. And all functions will be agnostic to the internal structure of the type, with accessor functions mediating access. Cool.

Is this okay given that it's reproducing inheritance? I need a more precise understanding of the downsides of inheritance if so. I think it's fine.

Okay, but how do I add an implementation, concretely? Given that I have CMR and dependencies implemented, what would be my approach for adding a new model variant within my library?

Create a new type that inherits from CMR. Create a new version of each function that needs to behave differently.

So the only tricky part is doing the inheritance. This performs a form of coupling that I should be a little wary about but seems okay. And I need multiple dispatch to be coordinated perfectly.

In its explanation of dispatch scope, if my __init__.py imports two libraries that both dispatch a function to the same name, even if a client script only imports one of those libraries, the dispatch will take into account both libraries. Let's think through what that means.

Say I import a script defining CMR and a script defining InstanceCMR. If dispatch is cross-library, then a function that dispatches differently for CMR and InstanceCMR will dispatch correctly even if I only import InstanceCMR. The implication is that I don't have to enforce dispatch scope in my implementation of ICMR. I can import the CMR type but I don't, for example, have to import CMR functions that I'm overriding. A subtle downside is that the ICMR script might not work in isolation. But that's okay if an action like `from compmemjax.models import ICMR` will still dispatch to CMR functions correctly. Still, I'll definitely import functions that ICMR borrows from CMR. I just won't import functions that ICMR overrides / doesn't use. So the scripts actually will work in isolation. Okay.

This seems to give a meaningful role to __init__.py. I wonder though if scope issues can come to play in client code. 

## Organization
The ideal organization for a set of modules is enforcement of both internal and external cohesion. 
The former means that the package has a minimal interface which exposes only concepts which are strongly related to the service the component provides. 
The latter means that the code in the package is strongly interrelated and thus strongly related to the provided service.
Modules which have mutual dependencies should not be considered separate units of code at all as none of them can be understood in isolation from the others.
This is why I define a jaxcmr.models that unites code in CMR and ICMR (for example) instead of separately specifying jaxcmr.cmr and jaxcmr.icmr. Similarly, mfc implementations should not be defined separately from the models that use them.
Stuff outside of jaxcmr.models needs to not depend on model implementations at all. 
That fits with how I defined `jax.analyses` to work with datasets irrespective of how they were generated.
What about my fitting library?
I've shifted to specifying likelihood functions with the models that they're associated with; there is no serious way to separate them that will work across model implementations.
But fitting consists of more than likelihood functions.
scipy's differential evolution algorithm takes a loss function as an argument.
So it's independent of  likelihood function implementation, but it's not independent of likelihood function.
That's how I organize interdependencies in my library -- by interface, not by implementation.
Ideally, a really simple interface.
By contrast, ICMR depends on some units of implementation in CMR, and CMR depends on some units of implementation in MFC. So uniting them in a single module is appropriate. Great. 

Could I have defined LinearAssociativeMemory implementations into separate scripts without losing something?
They share implementations of hebbian_associate, so more than interface is shared.

Nah, I'd rather be able to think about them separately. 