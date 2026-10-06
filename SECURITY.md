# Security policy

## Supported versions

quantdiff is pre-1.0. Security fixes go into the latest release only.

| Version | Supported |
| --- | --- |
| 0.1.x (latest) | Yes |
| older | No |

## Reporting a vulnerability

Use GitHub private vulnerability reporting: open the repository's **Security** tab and choose
**Report a vulnerability**. Please do not open a public issue for security problems.

Include the quantdiff version, your OS and Python version, the backend involved, and a minimal
reproduction. A malicious server response or a crafted prompts file that triggers the problem is
ideal. We aim to acknowledge reports within 7 days and will credit you in the advisory unless you
ask us not to.

## Threat model

quantdiff connects to model servers you choose, sends them prompts, and renders what comes back.

**Trusted:** the user, the command line, the files the user passes (`--prompts`,
`--scoring-prompts`), and the model servers the user names. quantdiff does not try to protect you
from a server you chose to run.

**Untrusted:** everything a model generates, and the content of HTTP responses in general. A model
can produce arbitrary text, including code, markup, and strings shaped like control sequences. The
design rules that follow from this:

- **Model output is never executed** unless you pass `--allow-code-exec`. With it, only the `code`
  suite's generated functions run, each in a separate Python subprocess with a scrubbed
  environment (no inherited secrets), a wall clock timeout, and POSIX resource limits where the
  platform supports them. On timeout the whole process tree is killed (a process group on POSIX,
  a job object on Windows), so processes the code starts do not outlive it. This is isolation,
  not a security sandbox: the subprocess runs as your user and can reach your filesystem and
  network. Enable it only for models you would run code from anyway.
- **Code verdicts** come only from the test harness, never from files or printed output. The
  harness receives a random nonce before the generated code is imported, reports over a private
  channel, and ends the process immediately so no exit handler from the generated code runs
  afterwards. Known limit: the generated code runs in the harness's interpreter, so code written
  specifically to read the nonce out of the harness's stack frames can still report a pass.
  Reward hacking that games the tests themselves (for example returning an object that equals
  everything) is also not detected. Treat a code pass rate as a measurement of a model you
  already trust not to target quantdiff.
- **Schema patterns** in prompt files are trusted, but they are matched against untrusted model
  output, so a pattern that backtracks badly would let a model stall a run. quantdiff refuses
  patterns that repeat a group containing a quantifier or an alternation (`(a+)+`, `(\w+\s?)*`,
  `(a|aa)*`), the usual source of exponential backtracking, and does not match strings longer
  than 10,000 characters. Patterns with several adjacent overlapping quantifiers (`.*.*.*x`) are
  still accepted and can be slow, polynomially in the input length; avoid them in your own
  suites.
- **The HTML card** escapes every piece of model text and server metadata, and contains no
  JavaScript and no external resources. It is safe to open and to share.
- **The HTTP client** accepts only `http` and `https` URLs, refuses redirects (so a server cannot
  bounce a request and its headers to another host), refuses credentials embedded in URLs, sets
  explicit timeouts, and caps response size.
- **API keys** are referenced by environment variable name (`@env:VAR_NAME` in a spec). The value
  is read at request time, sent only to the server in that spec, and never written to reports,
  logs, or the cache.
- **Network access** is limited to the servers you name, plus `huggingface.co` for the chat
  template check. `--offline` removes the Hugging Face request.
- **The reference cache** is stored as JSON. quantdiff never uses pickle or any other format that
  can execute code on load.
- **Zero runtime dependencies.** quantdiff uses only the Python standard library. This was a
  deliberate choice after the March 2026 LiteLLM PyPI compromise: there is no transitive
  dependency tree to poison.

Out of scope: denial of service by a server you named (it can always just be slow), and anything
that requires `--allow-code-exec` to escape a subprocess running as your own user.

## Release integrity

Releases are built by GitHub Actions from a tagged commit and published to PyPI with
[Trusted Publishing](https://docs.pypi.org/trusted-publishers/). No long-lived API token exists.
Each distribution file carries a [PEP 740](https://peps.python.org/pep-0740/) attestation signed
by the release workflow's identity.

To verify a published file against this repository, use `pypi-attestations`, which fetches the
file and its provenance from PyPI:

```
pipx run pypi-attestations verify pypi \
  --repository https://github.com/quantdiff/quantdiff \
  pypi:quantdiff-0.1.0-py3-none-any.whl
```

You can also view the attestations on the PyPI project page under each file's **Provenance**
section. Each one should name `.github/workflows/release.yml` in this repository as the publisher.
