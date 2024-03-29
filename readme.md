# Understand Neural Nets

> from the core.

A breakdown of NN from absolute scratch. requirements:

+ Numpy: because array interfaces are tedious to build by hand

That's it. The rest is pure python.

## Features

+ Pure readable python
+ No maths, no conceptually complicated learning required
+ Do three things easily:
    + Initialise
    + Train
    + Predict

Some of the really challenging setups aren't presented here, such as a GPT algorithm, or an LSTM or GNU or even **buzzword here**.

This code is more for:

+ Multi-layer NN for _things_
+ Build a teenie tiny neural net to do tiny things
+ fill up functions, tweak values through exposed functions
+ free forward and backpass functions
+ free customloss
+ prediction functions
+ Easy save/restore of weights and biases.
+ Daft toys; "What if I tried the same with X hidden layer" - with little effort
+ functional switching for activations
+ Silly easy reading style
+ Some discussion for learning in the process.
+ Easily run a single training step (for stream-in training!)


## Can I build useful things.

The code produced is a [Multi Layer Perceptron](https://ml-cheatsheet.readthedocs.io/en/latest/architectures.html#mlp) (That's the classical idea of neurons on a mesh) and with that you can build functional things. However some of the more interesting and _insert AI word here_ will need more library support. At that point TensorFlow and other industry leading tools should be considered.

That said, Neural Nets have served for very interesting things. Given good training (and a good shape) you can build real nets and deploy the weights within your own python work. For example, it's entirely possible to build things like:

+ Color detection
+ Number things
+ Word counting/prediction/spelling things.

even the crazy things _are_ possible; it would just take some effort:

+ Mini neural driving cars
+ Those 'generic' training games for tiny agents
+ Shake detectors (phones etc), motion and movement detection
+ Classifiers

Stuff like that... If you can _convert something to floats_, it can pretty-much be digested with a neural net.


## Getting started

Some cleanup is required however we have many iterations, view `np7` for
the most recent version

Creating a multi-layer neural-net:

```py
from np7 import *

nn = NN(Shape(1,[1],1))
nn.wb=nn.init_network()
# Training data: (input, expected output)
training_data = [(0.1, 0.2),
                 (0.3, 0.6),]
nn.train(training_data)
nn.query(.1) # is a 'predict' without close match
```

Predict data:

```py
letter_to_num = {'w': 0.1, 'h': 0.2, 'e': 0.3, 'n': 0.4, 'a': 0.5, 't': 0.6}
num_to_letter = {v: k for k, v in letter_to_num.items()}
nn.predict(.1, translate_in=letter_to_num, translate_out=num_to_letter)
```

**That's it.** No crazy logic. Just neurons.

## Loading existing weights.

The Neural Net can accept a `Shape` or some weights and biases. Like this:

```py
weights_v1, biases_v1 = (
    [
        array([[ 2.32254687, -3.86480048]]),
        array([[ 2.04736964], [-3.87521947]])
    ],
    [
        array([[-0.36708326,  0.07149161]]),
        array([[-0.45136009]])
    ]
)
v1 = NN(wb=WeightsBiases(weights_v1, biases_v1))
# See its shape:
v1.wb.shape()
# [1, [2], 1]
```

And run it:

```py
v1.predict(.2)
```

## Training

Training is a case of feeding the net with examples of _in_ and _out_; what you expect to see when some information is entered into the neural net. In this example we feed it a single letter `w`, and expect a letter in return `h`. The goal is to ensure the net returns the correct letter back.

Training alters the weights and biases over many iterations, such that the _maths_ within produces correct numbers.

Our training data should match the shape of the input and output. In this example we have `1` input and `1` output. The NN expects floats:

```py
from np7 import *

nn = NN(Shape(1,[1],1))
nn.wb=nn.init_network()

# Training data: (input, expected output)
training_data = [(0.1, 0.2),
                 (0.3, 0.6),]
nn.train(training_data)
```

If we have `2` inputs and `2` outputs, the format of the training data should change to reflect this:

```py
from np7 import *

nn = NN(Shape(2,[1],2))
nn.wb=nn.init_network()

# Training data: (input, expected output)
training_data = [( array([0.1,0.1]), array([0.2, 0.2])),
                 ( array([0.3,0.3]), array([0.6, 0.6])),]
nn.train(training_data)
```

## What can it do?!

Under the hood here we have a easily pluggable net of sigmoid values. The
inputs, output, and internal shape are goverened by the weights and biases.
Some things I'd played with:

+ Letter and word prediction
+ binary, and label classification
+ stream  Long-in, short-out capture
+ Recurrent NN

So with that it can do all sorts of things,
It just essentially needs the right layer setup and training.
Anyone can do that.

### Setup:

+ the input is a range of values: [.1, .2, .3]
+ The output is values [.4, .5]
+ The hidden layers help with evaluation.

We'll do letter prediction between two words "When" and "what". The catch
statements here are "whet" or "whan", they should never be suggested.
Somewhere deep inside maths is the ability to do this:

    chars       chars as numbers

    w ? == h    .1 ? == .2
    h ? == e    .2 ? == .4
    e ? == n    .4 ? == .5
    a ? == t    .3 ? == .6

If we convert each character to a float (something computers can read), then
run our magic `?` function, we return the _next correct letter_.

To set this up, we have 1 char in, one char out.

    input_size = 1
    output_size = 1

Between input and output we need something that _changes_ the values, because
if we wired the input function directly to the output function, the results
wouldn't change (\*this is extrapolation and in reality they would change,
just without enough "headspace" or computational bandwidth between the
two states, to apply any meaningful change.)

we can add _one_ mid-step. This is in the middle, it's hidden,

    input_size = 1
    hidden_layers = [1]
    output_size = 1

So input connects to one hidden function, that connects to the output function
If we change it to include two layer between input and output, each layer
with one float:

    hidden_layers = [1,1]

The input connects to layer one, then layer two, then the output.
Finally we can seriously expand this

    hidden_layers = [2, 3, 2]

One input, connects to _two_ nodes on the first hidden layer. Those two nodes
connect to _three_ nodes in the next layer. Those three nodes connect to _two_
more hidden nodes - the last hidden layer. Finally those two nodes connect
to the output layer. Like a giant interconnected mesh with all _nodes_ on a
layer recieving the value of _all_ nodes from the previous layer and so on.

The result is a cascade of numbers, the input number get computed through
this mesh web of numbers, using the `sigmoid` function `sigmoid(input, current_weight)`
The output is the last value squeezed out the `output` node.