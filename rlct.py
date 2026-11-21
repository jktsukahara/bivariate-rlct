from sage.all import (QQ, Integer, Polyhedron, gcd, line, points, sgn,
                      vector, PolynomialRing)
from sage.geometry.newton_polygon import NewtonPolygon


# ---------------------------------------------------------------------------
# Main functions
# ---------------------------------------------------------------------------

def rlct(f, verbose=False):
    r"""
    Compute the real log canonical threshold of ``f`` at the origin.

    The algorithm repeatedly applies the changes of variables of
    :func:`_change_of_variables` until the Newton polygon of the polynomial
    no longer changes. The RLCT is then `1/c`, where `c` is the Newton
    distance (see :func:`_newton_distance`) of the final polynomial.

    INPUT:

    - ``f`` -- a polynomial in a bivariate polynomial ring over `\QQ`
      (for example, an element of ``QQ['x,y']``)
    - ``verbose`` -- boolean (default: ``False``); if ``True``, print the
      facet polynomials and the polynomial after each change of variables,
      and the number of changes of variables used

    OUTPUT:

    A positive rational number: the RLCT of ``f`` at the origin. If the
    algorithm has not terminated after 10 changes of variables, the string
    ``"hasn't terminated after 10 iterations"`` is returned instead.

    EXAMPLES::

        sage: from rlct import rlct
        sage: x, y = polygens(QQ, 'x,y')
        sage: rlct(x^2 + y^2)
        1
        sage: rlct(x^2 * y^3)
        1/3
        sage: f = y^4 + y^3 + (1 + 4*x)*y^2 + (2*x + 3*x^2)*y + x^2
        sage: rlct(f)
        7/10

    With ``verbose=True``, the intermediate steps are printed::

        sage: rlct(f, verbose=True)
        x^2 + 2*x*y + y^2
        Change of Variable: x^4 - 4*x^3*y + 6*x^2*y^2 - 4*x*y^3 + y^4 - 2*x^2*y + x*y^2 + y^3 + y^2
        x^4 - 2*x^2*y + y^2
        Change of Variable: x^8 - 4*x^7 + 4*x^6*y + 7*x^6 - 12*x^5*y + 6*x^4*y^2 - 3*x^5 + 15*x^4*y - 12*x^3*y^2 + 4*x^2*y^3 - 2*x^3*y + 9*x^2*y^2 - 4*x*y^3 + y^4 + x*y^2 + y^3 + y^2
        -3*x^5 + y^2
        number of change of variables: 1
        7/10

    The coefficients must be rational::

        sage: x, y = polygens(RR, 'x,y')
        sage: rlct(x^2 + y^2)
        Traceback (most recent call last):
        ...
        ValueError: f must be a polynomial in two variables over QQ
    """
    R = f.parent()
    if R.base_ring() != QQ or R.ngens() != 2:
        raise ValueError("f must be a polynomial in two variables over QQ")

    counter = 0
    c_temp = 0

    while True:
        

        c = _newton_distance(f)

        if c == c_temp:
            break
        else:
            c_temp = c

        if _vertex_check(f):
            break
        g = _main_facet(f)
        if verbose:
            print("Main facet: " + str(g))
        if g == "parallel to axis":
            break
        else:
            w = _weight(g)

            if _weight_check(w):
                break
            else:
                phi = _change_of_variables(g, w)
                new_f = phi(f)
                if new_f == f:
                    break
                if counter > 10:
                            return "hasn't terminated after 10 iterations"

                f = new_f
                counter = counter + 1
                
                if verbose:
                    print("Change of Variables: " + str(phi))
                    print("Intermediate Polynomial: " + str(f))

    if verbose:
        print("Number of change of variables: " + str(counter))

    return 1 / c


def PNN_polynomial(W, r):
    r"""
    Return the polynomial attached to a polynomial neural network.

    Let `W_1, \dots, W_L` be the matrices in ``W`` and let
    `a = (a_1, \dots, a_d)` be a vector of new variables, where `d` is the
    number of columns of `W_1`. Write `\sigma` for the map that raises every
    coordinate of a vector to the power `r`. This function computes

    .. MATH::

        v = W_L \, \sigma(W_{L-1} \cdots \sigma(W_1 a)),

    views each entry of `v` as a polynomial in `a_1, \dots, a_d`, and
    returns the sum of the squares of all of their coefficients.

    INPUT:

    - ``W`` -- a nonempty list of matrices with entries in a polynomial ring
      (for example, ``QQ['x,y']``), with compatible sizes
    - ``r`` -- a positive integer, the exponent of the activation map

    OUTPUT:

    A polynomial in the same ring as the entries of the matrices in ``W``.

    EXAMPLES::

        sage: from rlct import PNN_polynomial
        sage: x, y = polygens(QQ, 'x,y')
        sage: M = matrix([[x + y, x^2], [x^2, y^2]])
        sage: PNN_polynomial([M], 2)
        2*x^4 + y^4 + x^2 + 2*x*y + y^2
        sage: g = PNN_polynomial([M, M], 2)
        sage: g.parent()
        Multivariate Polynomial Ring in x, y over Rational Field
    """
    Rx = W[0][0, 0].parent()
    d = W[0].ncols()
    Ra = PolynomialRing(Rx, d, 'a')
    aa = Ra.gens()

    v = vector(aa)

    for w in W[:-1]:
        v = w * v
        v = v.apply_map(lambda p: p**r)

    v = W[-1] * v

    g = sum(c * c for poly in v for c in poly.coefficients())
    return g


def print_poly(f, diag=False, ticks=True):
    r"""
    Plot the Newton polyhedron of ``f``.

    The plot shows the points in the support of ``f``, the filled-in Newton
    polyhedron (see :func:`_newton_boundary`) and its boundary. If
    ``diag = True``, also plot the diagonal. If ``ticks = True``, show the
    coordinate ticks.
    INPUT:

    - ``f`` -- a polynomial in two variables in some polynomial ring
    - ``diag`` -- (default: ``False``) a boolean
    - ``ticks`` -- (default: ``True``) a boolean

    OUTPUT:

    ``None``. The plot is displayed.

    EXAMPLES::

        sage: from rlct import print_poly
        sage: x, y = polygens(QQ, 'x,y')
        sage: print_poly(x^4 + x*y + y^3, diag=True)
    """
    exps = f.exponents()
    pts = points(exps, size=30, zorder=3)

    P = Polyhedron(vertices=(exps))
    pos_quad = Polyhedron(rays=[[0, 1], [1, 0]])
    P = P + pos_quad
    xexps = [e[0] for e in exps]
    yexps = [e[1] for e in exps]
    max_val = max(max(xexps), max(yexps)) + 1
    tick_list = list(range(max_val + 1))
    max_P = Polyhedron(vertices=([0, 0], [0, max_val], [max_val, 0],
                                 [max_val, max_val]))
    max_P = max_P & P
    max_P = max_P.render_solid(alpha=0.2, zorder=1)
    wire_P = P.render_wireframe(color='black', zorder=2, thickness=2)
    T = pts + max_P + wire_P

    if diag:
        T = T + line([(0, 0), (max_val, max_val)], color='green', zorder=2)

    if ticks:
        T = T.show(xmax=max_val, ymax=max_val, ticks=[tick_list, tick_list])
    else:
        T = T.show(xmax=max_val, ymax=max_val, ticks=[[], []])

    return T


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _newton_boundary(f):
    r"""
    Compute the Newton polyhedron of a bivariate polynomial.

    This is the Newton polytope of ``f`` plus the positive quadrant. Its
    boundary is the Newton boundary (Newton polygon) of ``f``.

    INPUT:

    - ``f`` -- a polynomial in two variables in some polynomial ring

    OUTPUT:

    The Newton polyhedron ``P`` of ``f``, an unbounded polyhedron in `\QQ^2`.

    EXAMPLES::

        sage: from rlct import _newton_boundary
        sage: x, y = polygens(QQ, 'x,y')
        sage: _newton_boundary(x^2 + y^3).vertices()
        (A vertex at (0, 3), A vertex at (2, 0))
    """
    P = f.newton_polytope()
    pos_quad = Polyhedron(rays=[[0, 1], [1, 0]])
    P = P + pos_quad
    return P


def _facet_polynomials(f):
    r"""
    Compute the facet polynomials of ``f``.

    These are the polynomials obtained by restricting the support of ``f``
    to each facet of its Newton polyhedron (see :func:`_newton_boundary`).

    INPUT:

    - ``f`` -- a polynomial in two variables in some polynomial ring

    OUTPUT:

    A list ``L`` of polynomials in the polynomial ring of ``f``.

    EXAMPLES::

        sage: from rlct import _facet_polynomials
        sage: x, y = polygens(QQ, 'x,y')
        sage: _facet_polynomials(x^4 + x*y + y^3)
        [x^4, x^4 + x*y, y^3 + x*y, y^3]
    """
    R = f.parent()
    P = _newton_boundary(f)
    L = [R({exp: cf for exp, cf in f.dict().items() if exp in facet})
         for facet in P.facets()]
    return L


def _newton_distance(f):
    r"""
    Compute the Newton distance of ``f``.

    This is the coordinate `c` of the point `(c, c)` where the boundary of
    the Newton polyhedron of ``f`` meets the diagonal ray.

    INPUT:

    - ``f`` -- a polynomial in two variables in some polynomial ring

    OUTPUT:

    A positive rational number ``c``.

    EXAMPLES::

        sage: from rlct import _newton_distance
        sage: x, y = polygens(QQ, 'x,y')
        sage: _newton_distance(x^2 + y^3)
        6/5
    """
    P = _newton_boundary(f)
    c = P.intersection(Polyhedron(rays=[[1, 1]])).vertices()[0][0]
    return c


def _vertex_check(f):
    r"""
    Check whether the diagonal meets the Newton boundary of ``f`` at a vertex.

    INPUT:

    - ``f`` -- a polynomial in two variables in some polynomial ring

    OUTPUT:

    ``True`` if the diagonal meets the Newton boundary of ``f`` at a vertex;
    ``False`` otherwise.

    EXAMPLES::

        sage: from rlct import _vertex_check
        sage: x, y = polygens(QQ, 'x,y')
        sage: _vertex_check(x^3 + x*y + y^3)
        True
        sage: _vertex_check(x^2 + y^3)
        False
    """
    P = _newton_boundary(f)
    for vertex in P.vertices():
        diff = vertex[0] - vertex[1]
        if diff == 0:
            return True
    return False


def _facet_sign_list(f):
    r"""
    Return the signs of `i - j` for the exponents `(i, j)` of ``f``.

    INPUT:

    - ``f`` -- a polynomial in two variables in some polynomial ring

    OUTPUT:

    A list containing `1`, `0` or `-1` for each term of ``f``, depending on
    whether the term lies below, on or above the diagonal.

    EXAMPLES::

        sage: from rlct import _facet_sign_list
        sage: x, y = polygens(QQ, 'x,y')
        sage: _facet_sign_list(x^4 + x*y + y^3)
        [1, -1, 0]
    """
    exps = f.exponents()
    diffs = []
    for exp in exps:
        diffs.append(sgn(exp[0] - exp[1]))
    return diffs


def _main_facet(f):
    r"""
    Return the facet polynomial of ``f`` whose facet crosses the diagonal.

    INPUT:

    - ``f`` -- a polynomial in two variables whose Newton polygon has no
      vertex on the diagonal (see :func:`_vertex_check`). This is not
      checked.

    OUTPUT:

    The facet polynomial (see :func:`_facet_polynomials`) that has terms on both
    sides of the diagonal. If there is none, the string
    ``"parallel to axis"`` is returned.

    EXAMPLES::

        sage: from rlct import _main_facet
        sage: x, y = polygens(QQ, 'x,y')
        sage: _main_facet(x^2 + 2*x*y^2 + y^4 + y^5)
        y^4 + 2*x*y^2 + x^2
    """
    polynomials = _facet_polynomials(f)
    for poly in polynomials:
        signs = _facet_sign_list(poly)
        if 1 in signs and -1 in signs:
            return poly
    return "parallel to axis"


def _weight(f):
    r"""
    Return the reduced weight vector of a facet polynomial.

    If the first and last exponents of ``f`` are `(i_1, j_1)` and
    `(i_2, j_2)`, this is `(|i_1 - i_2|, |j_1 - j_2|)` divided by the gcd of
    its two entries.

    INPUT:

    - ``f`` -- a facet polynomial in two variables with at least two terms,
      as returned by :func:`_main_facet`. This is not checked.

    OUTPUT: a list of two positive integers with gcd 1

    EXAMPLES::

        sage: from rlct import _weight
        sage: x, y = polygens(QQ, 'x,y')
        sage: _weight(x^2 + 2*x*y^2 + y^4)
        [1, 2]
    """
    exps = f.exponents()
    w = [Integer(abs(exps[0][0] - exps[-1][0])),
         Integer(abs(exps[0][1] - exps[-1][1]))]
    gcdw = gcd(w[0], w[1])
    w_red = [w[0] / gcdw, w[1] / gcdw]
    return w_red


def _weight_check(w):
    r"""
    Return whether neither entry of the weight vector ``w`` equals 1.

    INPUT:

    - ``w`` -- a list of two positive integers, as returned by
      :func:`_weight`

    OUTPUT:

    A boolean: ``True`` if both entries of ``w`` are different from 1, and
    ``False`` otherwise.

    EXAMPLES::

        sage: from rlct import _weight_check
        sage: _weight_check([2, 3])
        True
        sage: _weight_check([1, 2])
        False
    """
    if w[0] != 1 and w[1] != 1:
        return True
    return False


def _rational_root(f):
    r"""
    Return a rational root of maximal multiplicity of ``f``, if there is one.

    INPUT:

    - ``f`` -- a univariate polynomial over `\QQ`

    OUTPUT:

    If the factor of maximal multiplicity in the squarefree decomposition of
    ``f`` is linear, its root (a rational number). Otherwise ``False``.

    EXAMPLES::

        sage: from rlct import _rational_root
        sage: t = polygen(QQ, 't')
        sage: _rational_root((t + 2)^3 * (t - 1))
        -2
        sage: _rational_root((t^2 + 1)^2 * (t - 1))
        False
    """
    g = f.squarefree_decomposition()
    L = list(g)
    h = max(L, key=lambda fac: fac[1])[0]
    if h.degree() > 1:
        return False
    a = h[1]
    b = h[0]
    r = -b / a
    return r


def _change_of_variables(g, w):
    r"""
    Return the change of variables that simplifies the facet polynomial ``g``.

    If ``w = [k, 1]``, the map is `(x, y) \mapsto (x, y + \rho x^k)`, and if
    ``w = [1, k]``, it is `(x, y) \mapsto (x + \rho y^k, y)`. Here `\rho` is
    a rational root of maximal multiplicity (see :func:`_rational_root`) of
    ``g`` restricted to `x = 1` (respectively `y = 1`).

    INPUT:

    - ``g`` -- a facet polynomial in a bivariate polynomial ring over `\QQ`,
      as returned by :func:`_main_facet`
    - ``w`` -- the weight of ``g`` (see :func:`_weight`), with at least one
      entry equal to 1. This is not checked.

    OUTPUT:

    A ring endomorphism of the parent ring of ``g``.

    EXAMPLES::

        sage: from rlct import _change_of_variables
        sage: x, y = polygens(QQ, 'x,y')
        sage: phi = _change_of_variables(x^2 + 2*x*y + y^2, [1, 1])
        sage: phi(x^2 + 2*x*y + y^2)
        y^2
    """
    R = g.parent()
    x, y = R.gens()
    if w[1] == 1:
        h = g.specialization({x: 1})
        h = R(h / y**min(h.exponents()))
        h = h.univariate_polynomial()
        root = _rational_root(h)
        phi = R.hom([x, y + root * x**int(w[0])])
    else:
        h = g.specialization({y: 1})
        h = R(h / x**min(h.exponents()))
        h = h.univariate_polynomial()
        root = _rational_root(h)
        phi = R.hom([x + root * y**int(w[1]), y])
    return phi