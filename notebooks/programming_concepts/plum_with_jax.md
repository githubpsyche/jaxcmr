It seems that only the jit applied to the final plum.dispatch function matters for whether a set of functions are compiled or not. 
And critically, the settings for the jit are shared across all functions in the dispatch.
This is a major limitation when I want to use different jit settings for different functions in the dispatch, or want to compile some functions but not others.

And I can't dispatch over compiled functions, because plum.dispatch rejects compiled functions.

Solution isn't too bad: just define functions with separate names and unique compilation settings. 
Call them with same-named function.

Downside: I can't as easily turn compilation for the given function, maybe making debugging harder. Jax provides a disable_jit function. Okay, I'll just test using that whenever I need to, and go ahead and add jit to my library where appropriate.