# Python-Perceptron

An Basic implementation of the perceptron, the build block a neural net.

## Usage

Clone the repo:

    git clone https://github.com/paolodelia99/Python-Perceptron.git
    cd Python-Perceptron
    
```python
from src.perceptron import Perceptron
from src.functions.activationFunctions.heaviside import Heaviside
from src.functions.errorFunctions.quadratic_loss import QuadraticLoss

dataset = [[2.7810836, 2.550537003, 0],
               [1.465489372, 2.362125076, 0],
               [3.396561688, 4.400293529, 0],
               [1.38807019, 1.850220317, 0],
               [3.06407232, 3.005305973, 0],
               [7.627531214, 2.759262235, 1],
               [5.332441248, 2.088626775, 1],
               [6.922596716, 1.77106367, 1],
               [8.675418651, -0.242068655, 1],
               [7.673756466, 3.508563011, 1]]

p = Perceptron(2, 0.1, Heaviside(), QuadraticLoss())
p.train(dataset, 3, 30)

for d in dataset:
    assert p.evaluate(d[0], d[1]) == d[2]

```

For info about the what is the preceptron check out the [notebook](./demo/What_is_a_perceptron.ipynb) with the fully explanation.

If you wanna see more about how to use the perceptron checkout the [demos](./demo).

## Author

Paolo D'Elia

## Contributing

Feel free report issues and contribute to the project, making it better.

## License 

MIT
