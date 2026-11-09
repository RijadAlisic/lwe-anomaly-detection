"""Secret-key LWE scheme used in the paper: Enc(m) = (P, P s + r m + e mod q), with P in Z_q^v.

A ciphertext of a vector of N scalars is stored as an array of shape (N, v + 1): [P | b].
Linear maps with integer matrices act directly on this array (this is the homomorphism).
"""
import numpy as np


class LWE:
    def __init__(self, v, q, r, sigma, rng):
        self.v, self.q, self.r, self.sigma, self.rng = v, q, r, sigma, rng
        self.new_key()

    def new_key(self):
        self.s = self.rng.integers(0, self.q, size=self.v, dtype=np.int64)

    def noise(self, shape):
        return np.rint(self.rng.normal(0.0, self.sigma, size=shape)).astype(np.int64)

    def encrypt(self, m):
        """m: integer array of shape (N,) -> ciphertext of shape (N, v+1)."""
        m = np.asarray(m, dtype=np.int64).reshape(-1)
        P = self.rng.integers(0, self.q, size=(m.size, self.v), dtype=np.int64)
        b = (self._dot(P, self.s) + self.r * m + self.noise(m.size)) % self.q
        return np.concatenate([P, b[:, None]], axis=1)

    def phase(self, ct):
        """b - P s mod q, centered: equals r m + e."""
        x = (ct[:, -1] - self._dot(ct[:, :-1], self.s)) % self.q
        return np.where(x > self.q // 2, x - self.q, x)

    def decrypt(self, ct):
        return np.rint(self.phase(ct) / self.r).astype(np.int64)

    def lin(self, M, ct):
        """Homomorphic integer linear map: M (k x N integer) applied to ciphertexts ct (N, v+1)."""
        return (np.asarray(M, dtype=np.int64) @ ct) % self.q

    def _dot(self, P, s):
        """Exact (P @ s) mod q without int64 overflow, for q < 2^42: Horner over 14-bit chunks of s."""
        acc = np.zeros(P.shape[0], dtype=np.int64)
        for sh in (28, 14, 0):
            chunk = (s >> sh) & ((1 << 14) - 1)
            acc = (acc * (1 << 14) + P @ chunk) % self.q
        return acc
