# jrnmm

[![status](http://www.repostatus.org/badges/latest/concept.svg)](http://www.repostatus.org/#concept)
[![ci](https://github.com/dirmeier/jrnmm/actions/workflows/ci.yaml/badge.svg)](https://github.com/dirmeier/jrnmm/actions/workflows/ci.yaml)

> Sentence that is used as description in project.toml

## About

TODO

## Example usage

```

def run():
    sig = 2000
    mu = 220.0
    C = 135.0
    gain = 0.0
    y = simulate(
        jr.PRNGKey(12),
        1 / 128,
        8,
        initial_states=jnp.array([0.08, 18, 15, -0.5, 0, 0]),
        Cs=jnp.full(20, C),
        mus=jnp.full(20, mu),
        sigmas=jnp.full(20, sig),
        gains=jnp.full(20, gain),
    )
    print(y.shape)
    f, s = welch(y, fs=128, axis=1, nperseg=64)
    for i in range(20):
        plt.plot(10 * jnp.log10(s[i, :]))
    plt.show()


if __name__ == "__main__":
    run()
```

## Installation

To install the latest GitHub <RELEASE>, just call the following on the
command line:

```bash
pip install git+https://github.com/dirmeier/jrnmm@<RELEASE>
```

## Author

Simon Dirmeier <a href="mailto:sfyrbnd @ pm me">sfyrbnd @ pm me</a>
