# nd4clj

An implementation of [core.matrix](https://github.com/mikera/core.matrix) protocols with [nd4j](https://github.com/deeplearning4j/nd4j/). Most importantly this is supposed to allow the usage of the jcublas backend of nd4j and the integration of [deeplearning4j](http://deeplearning4j.org/) code in Clojure projects.

## Requirements

- be `core.matrix` compliant
- don't introduce significant performance overhead
- make implementation of machine learning algorithms in Clojure more feasible
- avoid wrapping INDArray Matrices for direct interop with `deeplearning4j`

## TODO

- make shaping of Nd4j compatible with core.matrix, e.g. introduce Vector type, see https://github.com/mikera/core.matrix/wiki/Vectors-vs.-matrices
- fix other outstanding issues in compliance tests
- evaluate the GPU backend in comparison to `theano`, e.g. with [boltzmann](https://github.com/whilo/boltzmann)
- implement example with deeplearning4j and `core.matrix` dependent code, e.g. boltzmann, incanter 2.0


## Usage

Not really usable yet, as it is not compliant to `core.matrix`.

## License

Copyright © 2015 Christian Weilbach

Distributed under the Eclipse Public License either version 1.0 or (at
your option) any later version.
