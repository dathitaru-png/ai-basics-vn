# Contributing to AI Basics VN

Thank you for helping more Vietnamese beginners learn AI, in the open.

Cảm ơn bạn đã giúp thêm nhiều người mới Việt Nam học AI một cách cởi mở.

## Getting started

```bash
git clone <your fork>
cd ai-basics-vn
pip install -e ".[test]"
python -m pytest                          # full test suite
aibasics demo all                         # smoke-test every demo
```

Python 3.10+ is required. The library must stay **dependency-free**
(standard library only); development tools may be proposed in an issue
first.

## What to work on

- New or improved lessons — plain, correct Vietnamese for absolute
  beginners, each with a small "Tự kiểm tra" section
- Translations and glossary entries (see `lessons/thuat-ngu.md`)
- More worked examples and exercises (synthetic data only, clearly
  labelled as fictional)
- Fixes that make the teaching code clearer — clarity beats cleverness

Check [ROADMAP.md](ROADMAP.md) and open issues for priorities.

## Pull request checklist

- [ ] Tests added or updated, and the full suite passes
- [ ] No new runtime dependencies
- [ ] Lesson text is accurate and understandable by an absolute beginner
- [ ] No real personal data in code, lessons, tests, or examples —
      synthetic data only, clearly labelled
- [ ] CHANGELOG.md updated under "Unreleased"

## Ground rules

This is teaching material. Contributions must stay honest about what
the algorithms can and cannot do, and must not present synthetic data
or toy results as real-world evidence. Simplifications are welcome
when the lesson says they are simplifications.

By contributing you agree to the [Code of Conduct](CODE_OF_CONDUCT.md)
and that your contributions are licensed under the project's MIT
license.
