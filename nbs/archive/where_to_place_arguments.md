There are two ways I can convey arguments between functions and they seem to be in tension a little. 

One option is to pass arguments directly as arguments to the function.
I directly call them out in the function signature and then use them in the function body.
This is really direct, clear about function dependencies, and easy to understand and test.

Another is to encapsulate them in a data structure and pass that data structure as an argument.
This is a little more indirect, but it allows me to pass a lot of arguments at once and without as much boilerplate or thinking about which model attributes I need to pull out in order to call the function.

The compromise seems to be to include implementations following the first pattern and just have them called by functions following the second pattern.

Yeah, ok.

I'm definitely going to implement at least two variants of OneWayMemory: LinearAssociativeMemory and InstanceMemory, for example.

The parameters used to probe LinearAssociativeMemory instances and InstanceMemory instances are different: Instance memories have a trace-based activation scaling parameter along with a feature-based one, while linear associative memories only have a feature-based one.

Storing these in parent classes seems inappropriate, as it would require different implementations of functions retrieving activations depending on the type of memory treated as an attribute. 

Therefore, I seem to have a good reason to instead store these parameters in the delegate class instances themselves, and exclude mention of them from the shared interface for retrieving activations.