import arrayclasses
import numpy as np


@arrayclasses.arrayclass
class State:
    x: float
    y: tuple[float, float]
    z: float


def test_wrapper():
    # Object creation
    state = State(x=5.0, y=(0, 1), z=0)
    np.testing.assert_equal(state.x, 5.0)
    np.testing.assert_array_equal(state.y, (0.0, 1.0))
    
    state.y = 2
    np.testing.assert_equal(state.y, (2.0, 2.0))

    # Array conversion.
    state = arrayclasses.from_array(State, (3, 0, 2, 0))
    np.testing.assert_equal(state.x, 3.0)
    np.testing.assert_equal(state.y, (0.0, 2.0))
    
    np.testing.assert_array_equal(np.array(state), [3.0, 0.0, 2.0, 0.0])
