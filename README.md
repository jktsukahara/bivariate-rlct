# bivariate-rlct
A SageMath algorithm which computes the local Real Log Canonical Threshold of any bivariate rational polynomial. Companion to the paper *Exact Algebraic Computation of Learning Coefficients for Two-Dimensional Singular Models*, by Sergeant-Perthuis, Tsigaridas and Tsukahara. 

# Local RLCTs of bivariate polynomials

This repository contains a SageMath implementation of an algorithm that computes the **real log canonical threshold (RLCT)** of a bivariate polynomial with rational coefficients at the origin.

It accompanies the paper:

> Grégoire Sergeant-Perthuis (1), Elias Tsigaridas (2), Jules Tsukahara (2) ((1) CQSB, Sorbonne Université, (2) OURAGAN, INRIA), *Exact Algebraic Computation of Learning Coefficients for Two-Dimensional Singular Models*, 2026. [LINK](https://arxiv.org/abs/2608.20183)

## Files

| File | Contents |
|---|---|
| `rlct.py` | All the functions, with documentation and examples. |
| `rlct.ipynb` | A notebook demonstrating how to use the algorithm |

## Requirements

- [SageMath](https://www.sagemath.org/)

## Usage

Put `rlct.py` in the same folder as your notebook or script, then:

```python
sage: from rlct import *
sage: x, y = polygens(QQ, 'x,y')
sage: rlct(x^2 + y^2)
1
sage: rlct(x^2 * y^3)
1/3
sage: f = y^4 + y^3 + (1 + 4*x)*y^2 + (2*x + 3*x^2)*y + x^2
sage: rlct(f)
7/10
```

The polynomial must have rational coefficients, that is, lie in a ring such as `QQ['x,y']`.

To see the intermediate steps of the algorithm, use `rlct(f, verbose=True)`.

### Main functions

- `rlct(f, verbose=False)`: the RLCT of `f` at the origin.
- `PNN_polynomial(W, r)`: the polynomial attached to a polynomial neural network with weight matrices `W` and activation exponent `r`. Its RLCT can then be computed with `rlct`.
- `print_poly(f, diag=False, ticks=True, points_only=False)`: plots the Newton polyhedron of `f`.

Each function is documented; type for example `rlct?` in a Sage session to read it. Functions whose names start with `_` are internal helpers.

## Running without installing Sage

You can run the notebook online on [CoCalc](https://cocalc.com), which has SageMath preinstalled. Upload `rlct.py` and `rlct.ipynb` to a CoCalc project, then open the notebook with the SageMath kernel.

## Tests

The examples in the documentation double as tests. From the folder containing `rlct.py`, run:

```
sage -t rlct.py
```

## How to cite

If you use this code, please cite the paper above.

## License

LICENSE_NAME. See the `LICENSE` file.
