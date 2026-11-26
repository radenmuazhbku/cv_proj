"""
Clifford algebra layers for PyTorch (any signature, any number of grades).

Tensors are multivector-valued:  x: (..., channels, 2**n)  with blades indexed by
bitmask (index 0 = scalar, 1 = e1, 2 = e2, 3 = e12, 4 = e3, ...).

Signature (p,q,r) is given as a tuple of e_i^2 values, e.g.
    (-1,)            complex numbers          d=2
    (-1,-1)          quaternions              d=4
    (1,1,1)          Cl(3,0)                  d=8
    (1,-1,-1,-1)     spacetime algebra        d=16
    (0,0,0,0)        exterior algebra of R^4  d=16  (geometric product == wedge)

Building blocks
    Clifford            tables + left/right multiplication matrices + products
    CliffordLinear      y_o = sum_i a_oi x_i  |  x_i b_oi  |  a_oi x_i b_oi  |  per-grade scalars
    CliffordProduct     degree-p chains (U1 x)(U2 x)...(Up x) with a chosen product type
    CliffordInteraction linear + any number of product terms (a "linear layer" with
                        lots of internal multiplicative interactions, no activation)

Product types for CliffordProduct (grade filters of the geometric product):
    geometric         full product                     (all grades)
    outer             wedge                            (grade = gA + gB)
    left_contraction  A _| B                           (grade = gB - gA)
    inner             symmetric/Hestenes inner         (grade = |gA - gB|)
Chains are folded left to right; outer and geometric are associative, the inner
products are not (document-level caveat, not a bug).
"""
import math

import torch
import torch.nn as nn


# ----------------------------------------------------------------------------
# Tables
# ----------------------------------------------------------------------------
def build_tables(signature):
    """sign[A,B]: e_A e_B = sign[A,B] e_{A^B};  grade[A] = popcount(A)."""
    n = len(signature)
    d = 1 << n
    sign = torch.zeros(d, d)
    for a in range(d):
        for b in range(d):
            s, t = 0, a >> 1
            while t:  # swaps needed to bring e_a e_b to canonical order
                s += bin(t & b).count("1")
                t >>= 1
            v = -1.0 if s & 1 else 1.0
            common = a & b
            for i in range(n):
                if (common >> i) & 1:
                    v *= signature[i]
            sign[a, b] = v
    grade = torch.tensor([bin(i).count("1") for i in range(d)])
    return sign, grade


def product_mask(grade, kind="geometric", out_grades=None):
    d = grade.numel()
    A = torch.arange(d)
    gA, gB, gK = grade[:, None], grade[None, :], grade[A[:, None] ^ A[None, :]]
    if kind == "geometric":
        m = torch.ones(d, d, dtype=torch.bool)
    elif kind == "outer":
        m = gK == gA + gB
    elif kind == "left_contraction":
        m = gK == gB - gA
    elif kind == "inner":
        m = gK == (gA - gB).abs()
    else:
        raise ValueError(kind)
    if out_grades is not None:
        m = m & torch.isin(gK, torch.tensor(list(out_grades)))
    return m


class Clifford(nn.Module):
    """Holds the algebra and provides multiplication primitives (no d^3 tensor)."""

    def __init__(self, signature):
        super().__init__()
        self.signature = tuple(signature)
        self.n = len(signature)
        self.d = 1 << self.n
        sign, grade = build_tables(signature)
        A = torch.arange(self.d)
        I = A[:, None] ^ A[None, :]                       # I[i,j] = i xor j
        self.register_buffer("sign", sign, persistent=False)
        self.register_buffer("grade", grade, persistent=False)
        self.register_buffer("I", I, persistent=False)
        self.register_buffer("TL", sign[I, A[None, :]], persistent=False)    # sign[K^B, B]
        self.register_buffer("TR", sign[A[:, None], I].T.contiguous(), persistent=False)  # sign[A, A^K]^T

    def grade_mask(self, grades):
        if grades is None:
            return torch.ones(self.d, device=self.sign.device)
        return torch.isin(self.grade, torch.tensor(list(grades), device=self.grade.device)).float()

    def left(self, a):
        """(..., d) -> (..., d, d) matrix L with (a x) = L @ x."""
        return a[..., self.I] * self.TL

    def right(self, b):
        """(..., d) -> (..., d, d) matrix R with (x b) = R @ x."""
        return b[..., self.I] * self.TR

    def product_table(self, kind="geometric", out_grades=None):
        m = product_mask(self.grade, kind, out_grades).to(self.sign.device)
        S = self.sign * m
        A = torch.arange(self.d, device=S.device)
        return S[A[:, None], self.I]                      # Sg[A,K] = S[A, A^K]

    def gp(self, u, v, Sg):
        """Product of multivectors (..., d) x (..., d) -> (..., d) using table Sg."""
        return (u.unsqueeze(-1) * Sg * v[..., self.I]).sum(-2)


def _flatten(x, c, d):
    assert x.shape[-2:] == (c, d), f"expected (..., {c}, {d}), got {tuple(x.shape)}"
    lead = x.shape[:-2]
    return x.reshape(-1, c, d), lead


# ----------------------------------------------------------------------------
# Linear (degree 1 in x)
# ----------------------------------------------------------------------------
class CliffordLinear(nn.Module):
    """
    sides = "left"   y_o = sum_i a_oi x_i            (complex / quaternion layers live here)
            "right"  y_o = sum_i x_i b_oi
            "both"   y_o = sum_i a_oi x_i b_oi       (sandwich-like; b starts near 1)
            "grade"  y_o = sum_i w_{oi,grade(K)} x_i,K   (one real weight per grade)
    weight_grades restricts a, b to given grades, e.g. (0,2) / even grades for rotors.
    """

    def __init__(self, c_in, c_out, signature, sides="left", weight_grades=None, bias=True):
        super().__init__()
        self.alg = Clifford(signature)
        self.c_in, self.c_out, self.sides = c_in, c_out, sides
        d = self.alg.d
        self.register_buffer("wmask", self.alg.grade_mask(weight_grades), persistent=False)
        std = 1.0 / math.sqrt(c_in * d)
        shape = (c_out, c_in, d)
        if sides in ("left", "both"):
            self.a = nn.Parameter(torch.randn(shape) * std)
        if sides in ("right", "both"):
            b = torch.randn(shape) * (std if sides == "right" else 0.02)
            if sides == "both":
                b[..., 0] += 1.0
            self.b = nn.Parameter(b)
        if sides == "grade":
            self.w = nn.Parameter(torch.randn(c_out, c_in, self.alg.n + 1) / math.sqrt(c_in))
        self.bias = nn.Parameter(torch.zeros(c_out, d)) if bias else None

    def weight_matrix(self):
        """(c_out, c_in, d, d): the real matrix this layer applies per channel pair."""
        al = self.alg
        if self.sides == "left":
            return al.left(self.a * self.wmask)
        if self.sides == "right":
            return al.right(self.b * self.wmask)
        if self.sides == "both":
            return al.left(self.a * self.wmask) @ al.right(self.b * self.wmask)
        raise RuntimeError

    def forward(self, x):
        x2, lead = _flatten(x, self.c_in, self.alg.d)
        if self.sides == "grade":
            y = torch.einsum("oiK,niK->noK", self.w[..., self.alg.grade], x2)
        else:
            y = torch.einsum("oiKA,niA->noK", self.weight_matrix(), x2)
        if self.bias is not None:
            y = y + self.bias
        return y.reshape(*lead, self.c_out, self.alg.d)

    @torch.no_grad()
    def load_complex(self, W):
        """Set a complex layer from a complex matrix W (c_out, c_in). signature must be (-1,)."""
        assert self.sides == "left" and self.alg.signature == (-1,)
        self.a[..., 0], self.a[..., 1] = W.real, W.imag


# ----------------------------------------------------------------------------
# Products (degree >= 2 in x)
# ----------------------------------------------------------------------------
class CliffordProduct(nn.Module):
    """
    y_o = sum_r R_or * ( (U_1 x)_r  *  (U_2 x)_r  *  ...  *  (U_p x)_r )
    U_j are left-multiplication maps c_in -> rank channels, '*' is the chosen product
    type, out_grades filters the final product. weight_grades=(0,) makes each U_j a
    real channel mixer, so the multivector structure only enters through '*'
    (this is the Clifford analogue of an FM / wedge-of-projections).
    """

    def __init__(self, c_in, c_out, signature, degree=2, rank=4, product="geometric",
                 out_grades=None, weight_grades=None):
        super().__init__()
        assert degree >= 2
        self.alg = Clifford(signature)
        self.c_in, self.c_out, self.degree, self.rank = c_in, c_out, degree, rank
        d = self.alg.d
        self.scalar_only = tuple(weight_grades) == (0,) if weight_grades is not None else False
        self.register_buffer("wmask", self.alg.grade_mask(weight_grades), persistent=False)
        std = 1.0 / math.sqrt(c_in * (1 if self.scalar_only else d))
        self.factors = nn.ParameterList(
            [nn.Parameter(torch.randn(rank, c_in, d) * std) for _ in range(degree)])
        self.register_buffer("S_mid", self.alg.product_table(product, None), persistent=False)
        self.register_buffer("S_last", self.alg.product_table(product, out_grades), persistent=False)
        self.readout = nn.Parameter(torch.randn(c_out, rank) / math.sqrt(rank))

    def pair_terms(self):
        """Nonzero (blade_A, blade_B) pairs coupled by the final product, per rank channel."""
        return int((self.S_last != 0).sum())

    def forward(self, x):
        x2, lead = _flatten(x, self.c_in, self.alg.d)
        z = None
        for j, f in enumerate(self.factors):
            if self.scalar_only:
                u = torch.einsum("ri,niB->nrB", f[..., 0], x2)
            else:
                u = torch.einsum("riKB,niB->nrK", self.alg.left(f * self.wmask), x2)
            if z is None:
                z = u
            else:
                z = self.alg.gp(z, u, self.S_last if j == self.degree - 1 else self.S_mid)
        y = torch.einsum("or,nrK->noK", self.readout, z)
        return y.reshape(*lead, self.c_out, self.alg.d)


# ----------------------------------------------------------------------------
# Everything together
# ----------------------------------------------------------------------------
class CliffordInteraction(nn.Module):
    """
    A linear-layer-shaped module: linear Clifford map + sum of product terms.
        products = [dict(degree=2, rank=4), dict(degree=3, rank=4, product="outer"), ...]
    Any key of CliffordProduct (except c_in, c_out, signature) may be given per term.
    """

    def __init__(self, c_in, c_out, signature, linear_sides="both", linear_grades=None,
                 products=(), bias=True):
        super().__init__()
        self.c_in, self.c_out, self.d = c_in, c_out, 1 << len(signature)
        self.linear = (CliffordLinear(c_in, c_out, signature, linear_sides, linear_grades, bias)
                       if linear_sides else None)
        self.products = nn.ModuleList(
            [CliffordProduct(c_in, c_out, signature, **kw) for kw in products])

    def forward(self, x):
        y = 0.0
        if self.linear is not None:
            y = y + self.linear(x)
        for p in self.products:
            y = y + p(x)
        return y


# ----------------------------------------------------------------------------
# Examples / tests
# ----------------------------------------------------------------------------
def n_params(m):
    return sum(p.numel() for p in m.parameters())


def dense_poly_params(n_in, n_out, degree):
    """Params of a dense layer holding all monomials of exactly `degree` in n_in inputs."""
    return math.comb(n_in + degree - 1, degree) * n_out


if __name__ == "__main__":
    torch.manual_seed(0)
    close = lambda a, b: torch.allclose(a, b, atol=1e-5)

    # ---- 1. Complex layer == complex matmul --------------------------------
    # Complex numbers = Cl(0,1): blades [1, e1], e1^2 = -1, so e1 plays the role of i.
    # Data layout is an extra trailing dim of size 2 (real, imag).
    lay = CliffordLinear(5, 3, signature=(-1,), sides="left", bias=False)
    x = torch.randn(4, 5, 2)
    W = torch.complex(lay.a[..., 0], lay.a[..., 1]).detach()            # (3,5) complex
    yz = torch.complex(x[..., 0], x[..., 1]) @ W.T
    y = lay(x)
    assert close(y[..., 0], yz.real) and close(y[..., 1], yz.imag)
    # and the other way around: initialise from an existing complex matrix
    W2 = torch.randn(3, 5, dtype=torch.cfloat)
    lay.load_complex(W2)
    yz = torch.complex(x[..., 0], x[..., 1]) @ W2.T
    y = lay(x)
    assert close(y[..., 0], yz.real) and close(y[..., 1], yz.imag)
    print("complex   : layer == complex matmul   (weight (3,5,2) = real+imag ndarray dim)")

    # ---- 2. Quaternion layer == Hamilton-product matmul -----------------------
    # Cl(0,2): blades [1, e1, e2, e12] = [1, i, j, k], e1^2 = e2^2 = -1, e1 e2 = e12.
    def hamilton(p, q):
        pr, pi, pj, pk = p.unbind(-1)
        qr, qi, qj, qk = q.unbind(-1)
        return torch.stack([pr * qr - pi * qi - pj * qj - pk * qk,
                            pr * qi + pi * qr + pj * qk - pk * qj,
                            pr * qj - pi * qk + pj * qr + pk * qi,
                            pr * qk + pi * qj - pj * qi + pk * qr], -1)

    ql = CliffordLinear(5, 3, signature=(-1, -1), sides="left", bias=False)
    x = torch.randn(4, 5, 4)
    ref = hamilton(ql.a.detach()[None], x[:, None]).sum(2)              # (4,3,4)
    assert close(ql(x), ref)
    print("quaternion: layer == Hamilton-product matmul (4x fewer params than dense real)")

    # ---- 3. Rotor sandwich a x a~ (the 'both sides' case) -----------------------
    sl = CliffordLinear(1, 1, signature=(-1, -1), sides="both", bias=False)
    with torch.no_grad():
        q = torch.randn(4)
        q = q / q.norm()
        sl.a[0, 0] = q
        sl.b[0, 0] = q * torch.tensor([1.0, -1.0, -1.0, -1.0])        # conjugate = inverse
    x = torch.randn(7, 1, 4)
    assert close(sl(x).norm(dim=-1), x.norm(dim=-1))                    # norm preserved
    print("sandwich  : unit-quaternion a x a^-1 preserves norm (rotation of the 3-vector part)")

    # ---- 4. d=16 examples ------------------------------------------------------
    # (A) Cl(4,0), a single 16-component multivector in and out (16-dim feature vector).
    #     Linear two-sided map + degree-2 and degree-3 full geometric products.
    sigA = (1, 1, 1, 1)
    A = CliffordInteraction(1, 1, sigA, linear_sides="both",
                            products=[dict(degree=2, rank=4), dict(degree=3, rank=4)])
    x = torch.randn(32, 1, 16, requires_grad=True)
    yA = A(x)
    yA.square().mean().backward()
    assert yA.shape == (32, 1, 16) and x.grad is not None
    print(f"\n(A) Cl(4,0) 16->16: {n_params(A)} params")
    print(f"    dense quadratic would need {dense_poly_params(16, 16, 2)}, "
          f"dense cubic {dense_poly_params(16, 16, 3)}")
    for p in A.products:
        print(f"    degree-{p.degree} term couples {p.pair_terms()} (blade,blade) pairs per rank channel")

    # (B) Wedge-only, high grade: 8 vector channels in R^4 -> wedge of 4 learned
    #     real mixes of them -> only the grade-4 blade (e1234, index 15) can be nonzero.
    B = CliffordProduct(8, 2, sigA, degree=4, rank=3, product="outer",
                        out_grades=(4,), weight_grades=(0,))
    xv = torch.zeros(5, 8, 16)
    xv[..., [1, 2, 4, 8]] = torch.randn(5, 8, 4)                       # grade-1 inputs only
    yB = B(xv)
    others = [i for i in range(16) if i != 15]
    assert yB[..., others].abs().max() < 1e-6 and yB[..., 15].abs().max() > 0
    print(f"\n(B) wedge^4 of learned vector mixes: output lives only in e1234, "
          f"params={n_params(B)}  (a 4x4 determinant of linear forms of x per rank channel)")

    # (C) Spacetime algebra Cl(1,3): 4 channels x 16 = 64 features.
    #     Even-grade (rotor-type) two-sided linear map  +  degree-2 geometric product
    #     filtered to scalar+bivector outputs (invariant / field-strength-like) +
    #     degree-3 wedge producing trivectors.
    sigC = (1, -1, -1, -1)
    C = CliffordInteraction(
        4, 4, sigC, linear_sides="both", linear_grades=(0, 2, 4),
        products=[dict(degree=2, rank=6, out_grades=(0, 2)),
                  dict(degree=3, rank=6, product="outer", out_grades=(3,), weight_grades=(0, 1))])
    x = torch.randn(2, 9, 4, 16, requires_grad=True)                    # extra leading dims ok
    yC = C(x)
    yC.sum().backward()
    assert yC.shape == (2, 9, 4, 16)
    print(f"\n(C) Cl(1,3) 64->64: {n_params(C)} params; dense quadratic on 64 inputs "
          f"would be {dense_poly_params(64, 64, 2)}")

    # (D) Exterior algebra as a Clifford algebra: signature (0,0,0,0) makes the
    #     geometric product *be* the wedge, so no filtering is needed.
    D = CliffordProduct(4, 4, (0, 0, 0, 0), degree=3, rank=4, product="geometric")
    print(f"(D) exterior algebra R^4 via degenerate signature: output shape "
          f"{tuple(D(torch.randn(3, 4, 16)).shape)}")
    print("\nall checks passed")