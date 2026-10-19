"""pytest, for every test file in this repo: a scratch HOME (USERPROFILE on Windows) for the whole run, so nothing a
test starts can write into the real ~/.moth (a real moth.serve's pidfile serve-<port>.pid, the token, logs,
config.json, the home folder). Tests that need their own home still make one; this is the floor under all of them,
and child processes inherit it. The real home stays in MOTH_TEST_REAL_HOME (tests/test_cli_tools.py checks that it
isn't touched), and PYTHONUSERBASE keeps pointing at the real user site, so child Pythons still find packages
installed with `pip install --user`."""
import os
import site
import tempfile

if "MOTH_TEST_REAL_HOME" not in os.environ:   # once (a nested pytest keeps the first scratch home)
    os.environ["MOTH_TEST_REAL_HOME"] = os.path.expanduser("~")
    os.environ.setdefault("PYTHONUSERBASE", site.getuserbase())
    _home = tempfile.mkdtemp(prefix="moth-test-home-")
    os.environ["HOME"] = os.environ["USERPROFILE"] = _home
    # moth's XS repo isn't published (moth/weights.py UNRELEASED: never contacted); the download / update tests run
    # against a fake Hub under this stand-in id, as a real release would
    os.environ.setdefault("MOTH_REPO", "test-owner/moth-xs")
