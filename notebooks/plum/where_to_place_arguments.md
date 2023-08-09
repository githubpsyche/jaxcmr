There are two ways I can convey arguments between functions and they seem to be in tension a little. 

One option is to pass arguments directly as arguments to the function.
I directly call them out in the function signature and then use them in the function body.
This is really direct, clear about function dependencies, and easy to understand and test.

Another is to encapsulate them in a data structure and pass that data structure as an argument.
This is a little more indirect, but it allows me to pass a lot of arguments at once and without as much boilerplate or thinking about which model attributes I need to pull out in order to call the function.

The compromise seems to be to include implementations following the first pattern and just have them called by functions following the second pattern.

Yeah, ok.