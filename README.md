# arrayclass

A small `@dataclass`-like decorator for python classes. The class will store its values in a single contiguous [numpy](https://numpy.org) array. It can also be converted to and from plain numpy arrays.

## Installation

`poetry add dataclasses` or `pip install dataclasses`

## Usage

```py
import arrayclasses

@arrayclasses.arrayclass
class State:
    x: float
    y: tuple[float, float]
    z: float

# Object creation
state = State(x=5, y=(0, 1), z=0)
print(np.x)  # Prints 5.0
print(np.y)  # Prints np.array([0.0, 1.0])
state.y = 2.0
print(np.y)  # Prints np.array([2.0, 2.0])

# Array conversion.
state = arrayclasses.from_array((5, 0, 1, 0))
print(np.array(state))  # prints np.array([5.0, 0.0, 1.0, 0.0])
```
