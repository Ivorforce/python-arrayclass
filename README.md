# arrayclass

A small `@dataclass`-like decorator for python classes. The class will store its values in a single contiguous [numpy](https://numpy.org) array. It can also be converted to and from plain numpy arrays.

## Installation

`poetry add dataclasses` or `pip install dataclasses`

## Usage

```py
def simulate(steps: int, fs: float) -> np.ndarray:
    # Define the state class.
    # Imagine we had 20 variables here...
    @arrayclasses.arrayclass
    class State:
        x: float
        y: float
    
    def step(t, xy):
        # normally, this would be `x, y, ... = xy`
        s = arrayclasses.from_array(State, xy)
        a = 1 - np.sqrt(s.x ** 2 + s.y ** 2)
        w = 2 * np.pi / (1 * fs)
        
        # normally, this would be `return (..., ...)`
        return State(
            x=a * s.x - w * s.y,
            y=a * s.y + w * s.x,
        )
    
    solved = integrate.solve_ivp(
        fun=step,
        y0=State(-1, 0),
        t_span=(0, steps),
        method="RK45"
    )
    return solved.y
```

## Features

```
@arrayclasses.arrayclass(dtype=object)  # You can coerce the array dtype manually
class Object:
    x: float  # A single value.
    y: tuple[float, float]  # Will yield np.ndarray windows, not tuples. This may be subject to change in the future.

a = Object(x=5, y=(2, 3))
print(len(a))  # 3
print(tuple(a))  # (5, 2, 3)
```

## Why would I need this?

You may be forced, or inclined, to use numpy arrays in some situations where classes would be more appropriate.

An example might be `scipy.integrate` - You are working with an array of numbers that really wants to be a class.

Packing and unpacking tuples is a common workaround. However, when you approach 10 or 20 variables, this gets quite messy fast. Now, you might prefer to use an `@arrayclass` to get nicer code that plays well with your IDE.
