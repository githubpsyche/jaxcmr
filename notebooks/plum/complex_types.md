# How Multiple Dispatch Enables Polymorphism

Different kinds of polymorphism are enabled by multiple dispatch.

"Free" polymorphism occurs when a parent function calls other functions on an object. 
As long as called functions are defined for the object's type, the parent function will work, allowing behavior to be extended without modifying the parent function by defining new functions for new types.

A more constrained, hierarchical pattern is possible by defining functions for abstract types.
Abstract types have no data layout.
They can only be used for sub-typing and for type declarations on function methods.
To informally define interface constraints for subtypes, we can similarly define abstract functions that are implemented by subtypes.
Concrete functions for abstract types can be defined that work on any subtype that implements any of the abstract functions.

Example: a `Shape` abstract type with an `area` abstract functions and a `combined_area` concrete function.
The `combined_area` function might accept any number of shapes and return the sum of their areas, e.g.,

```python
def combined_area(a: Shape, b: Shape):
    return area(a) + area(b)
```

When we define a concrete subtype of `Shape`, we need only implement the `area` function to be able to use the `combined_area` function on instances of the subtype.

```python
class Circle(Shape):
    diameter: float

def radius(circle: Circle):
    return circle.diameter / 2

def area(circle: Circle):
    return math.pi * radius(circle) ** 2
```

Python's method resolution algorithm can find the right execution path for each shape, even though the exact code is different in every case. What's more, in cases where the code is type stable and we are perform jit compilation on applicable functions, this polymorphism has no runtime cost.

## Types of Polymorphism in Multiple Dispatch
1. Ad-hoc Polymorphism

As seen in the Shape example, the specific implementation of a function like area can be customized for each subtype, allowing different behavior for each class. 
This promotes code reusability without sacrificing flexibility. 
New shapes can be introduced without modifying existing code, and as long as they conform to the expected interface, they'll work seamlessly with existing functions.

1. Parametric Polymorphism

Multiple dispatch can also allow for generic functions that can work with any type as long as they satisfy certain constraints. This enables even greater code reuse, as one function can handle various types, leading to more concise and maintainable code.

3. Inclusion Polymorphism
   
By using abstract types and defining a hierarchy, it's possible to create an inclusion relationship between types. This allows you to write code that operates on abstract types and works with any concrete subtype, leading to more versatile and reusable code.

Avoiding speculative generality is a key principle of software design.
The implication from the comparison above seems to be that I should start with ad-hoc polymorphism and only move to parametric or inclusion polymorphism when I have a concrete need for it.