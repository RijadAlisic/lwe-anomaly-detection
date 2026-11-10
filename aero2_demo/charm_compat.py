"""Charm-Crypto integer compatibility shim.

The LWE ciphertext arithmetic in this project (aero2_unified_detect.py,
functions.py, ...) only needs a big-integer type supporting + - * % and
int() casting. Arbitrary-precision Python ints already satisfy that
natively, so when charm-crypto isn't installed we fall back to plain
`int` rather than hand-rolling a fixed-width replacement.

(For this project's actual parameters — q ~ 2**23, n ~ 116 — even int64
would never overflow, but native int sidesteps the question entirely and
is a closer behavioral match to charm.core.math.integer.Element.)
"""
import random
import sys
import warnings

try:
    from charm.core.math.integer import integer, randomPrime, bitsize
    CHARM_AVAILABLE = True

except ImportError:
    CHARM_AVAILABLE = False

    warnings.warn(
        "charm-crypto not found — falling back to native Python int "
        "arithmetic for LWE ciphertext operations.",
        RuntimeWarning,
        stacklevel=2,
    )
    print(
        "[charm_compat] WARNING: charm-crypto is not installed. "
        "Falling back to plain Python int (arbitrary precision) in place "
        "of charm.core.math.integer. This project's parameters (q ~ 2**23, "
        "n ~ 116) are well within int64/native-int range, so no overflow "
        "is expected, but this is unverified against charm's own numerics.",
        file=sys.stderr,
    )

    def integer(x):
        return int(x)

    def bitsize(x):
        return int(x).bit_length()

    def _is_probable_prime(n, k=40):
        if n < 2:
            return False
        for p in (2, 3, 5, 7, 11, 13, 17, 19, 23, 29, 31, 37):
            if n % p == 0:
                return n == p
        d, r = n - 1, 0
        while d % 2 == 0:
            d //= 2
            r += 1
        for _ in range(k):
            a = random.randrange(2, n - 1)
            x = pow(a, d, n)
            if x in (1, n - 1):
                continue
            for _ in range(r - 1):
                x = pow(x, 2, n)
                if x == n - 1:
                    break
            else:
                return False
        return True

    def randomPrime(bits, safe_prime=False):
        while True:
            candidate = random.getrandbits(bits) | (1 << (bits - 1)) | 1
            if safe_prime:
                if _is_probable_prime(candidate) and _is_probable_prime((candidate - 1) // 2):
                    return candidate
            elif _is_probable_prime(candidate):
                return candidate
