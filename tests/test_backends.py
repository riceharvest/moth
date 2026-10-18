"""moth's swappable model backends (moth/backends, moth/docs/MODEL_BACKENDS.md).

  python tests/test_backends.py      (from ~/moth; a few seconds, CPU only, temp dirs, no downloads)

Covers: moth.json parsing and defaults (a folder without one = moth-112m as before), model specs (folder / HF repo id
/ openai:URL#model), `moth config model`, launch passing specs through, `moth update` leaving models it doesn't
manage alone (and the pet's --check not erroring), doctor's model + privacy lines without contacting other hosts;
with torch: a tiny random tiny-agent checkpoint gives the exact same turns with no moth.json, with a moth.json spelling
out every default, and built the old way (Moth.__new__ + setup, as the benchmark does); the openai backend against
the e2e fake OpenAI server (reaction nudge only on that turn, tools off, a strict server's 400 retried); with
transformers installed: a tiny random GPT-2 with a chat template behind moth (phrase ban included).
"""
from __future__ import annotations

import contextlib
import importlib.util
import io
import json
import os
import shutil
import sys
import tempfile
import threading
import unittest
from unittest import mock

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "tests", "e2e"))
TOKENIZER = os.path.join(ROOT, "data", "tiny-agent-112m", "tokenizer.json")

from moth import backends  # noqa: E402
from moth.backends import manifest as M  # noqa: E402


def _torch_ok() -> bool:
    try:
        import torch.nn  # noqa: F401  (a broken / stub torch has no nn)
        import tokenizers  # noqa: F401
        return os.path.isfile(TOKENIZER)
    except Exception:  # noqa: BLE001
        return False


TORCH = _torch_ok()
HAVE_TF = importlib.util.find_spec("transformers") is not None
needs_torch = unittest.skipUnless(TORCH, f"needs torch + tokenizers and {TOKENIZER}")


class Tmp(unittest.TestCase):
    def setUp(self):
        self.d = tempfile.mkdtemp()
        self.cfg = os.path.join(self.d, "config.json")
        self.env = mock.patch.dict(os.environ, {"MOTH_CONFIG_FILE": self.cfg, "MOTH_CACHE": os.path.join(self.d, "c"),
                                                "HF_HUB_OFFLINE": "1"})
        self.env.start()

    def tearDown(self):
        self.env.stop()
        shutil.rmtree(self.d, ignore_errors=True)

    def folder(self, name, manifest=None, config=None):
        p = os.path.join(self.d, name)
        os.makedirs(p, exist_ok=True)
        if manifest is not None:
            with open(os.path.join(p, "moth.json"), "w") as f:
                json.dump(manifest, f)
        if config is not None:
            with open(os.path.join(p, "config.json"), "w") as f:
                json.dump(config, f)
        return p


class Manifests(Tmp):
    def test_missing_manifest_is_moth_112m(self):
        m = M.load(self.folder("plain"))
        self.assertTrue(m.is_default)
        self.assertEqual((m.backend, m.character, m.chat_format, m.tools, m.lowercase), ("tiny-agent", "moth", "chatml", True, True))
        self.assertEqual(m.sampling, {"temperature": 0.8, "top_p": 0.95, "rep_penalty": 1.3, "max_tokens": 200})
        from moth.agent import agent_turn, clean_agent_message
        msg = "## Done\n- fixed `x.py`"
        self.assertEqual(m.card(clean_agent_message(msg)), agent_turn(msg))   # the trained card, byte for byte
        self.assertEqual(m.system_prompt("SYS"), "SYS")                       # SYSTEM_STUDENT is the caller's default
        self.assertIsNone(m.reaction)

    def test_full_manifest(self):
        m = M.load(self.folder("m2", {"backend": "tiny-agent", "character": "moth 2", "system": "you are moth 2.",
                                      "agent_card": "<agent>{message}</agent>", "sampling": {"temperature": 0.6},
                                      "tools": False, "repo": "someone/moth-2", "version": "2.0", "context": 2048,
                                      "future_key": 1}))
        self.assertFalse(m.is_default)
        self.assertEqual((m.character, m.system_prompt("SYS"), m.card("hi"), m.repo, m.version, m.context),
                         ("moth 2", "you are moth 2.", "<agent>hi</agent>", "someone/moth-2", "2.0", 2048))
        self.assertEqual(m.sampling["temperature"], 0.6)
        self.assertEqual(m.sampling["top_p"], 0.95)            # the rest keep moth's defaults
        self.assertEqual(m.card_parts(), ("<agent>", "</agent>"))
        self.assertIn("unknown key 'future_key' ignored", m.notes)

    def test_general_models_default_differently(self):
        m = M.parse({"backend": "transformers", "model": "x/y"})
        self.assertEqual((m.chat_format, m.tools, m.reaction), ("template", False, M.GENERIC_REACTION))
        self.assertIn("you are moth", m.system_prompt("SYS"))  # not the 112M model's trained prompt
        n = M.parse({"backend": "openai", "base_url": "http://127.0.0.1:1/v1", "model": "m", "character": "bugsy"})
        self.assertIn("you are bugsy", n.system_prompt("SYS"))

    def test_bad_manifests_name_the_problem(self):
        for data, word in (({"backend": "gguf"}, "backend"), ({"agent_card": "no slot"}, "{message}"),
                           ({"sampling": {"temperature": "hot"}}, "sampling.temperature"),
                           ({"sampling": {"top_p": 2}}, "range"), ({"tools": "yes"}, "tools"),
                           ({"backend": "openai", "model": "m"}, "base_url"), ({"chat_format": "template"}, "chatml"),
                           ({"character": ""}, "character"), ([1], "object"),
                           ({"backend": "openai", "base_url": "http://x", "model": "m", "api_key_env": "GITHUB_TOKEN"},
                            "MOTH_")):
            with self.assertRaises(M.ManifestError, msg=data) as cm:
                M.parse(data)
            self.assertIn(word, str(cm.exception))
        p = self.folder("broken")
        with open(os.path.join(p, "moth.json"), "w") as f:
            f.write("{nope")
        with self.assertRaises(M.ManifestError):
            M.load(p)

    def test_apply_args(self):
        import argparse
        a = argparse.Namespace(max_tool_rounds=3, ctx_budget=None)
        self.assertIs(M.apply_args(a, M.Manifest()), a)       # nothing to change: the very same namespace
        b = M.apply_args(a, M.parse({"tools": False, "context": 1024}))
        self.assertEqual((b.max_tool_rounds, b.ctx_budget, a.max_tool_rounds), (0, 1024, 3))
        a.ctx_budget = 500
        self.assertEqual(M.apply_args(a, M.parse({"context": 1024})).ctx_budget, 500)   # --ctx-budget wins


class Specs(Tmp):
    def test_parse_spec(self):
        self.assertEqual(backends.parse_spec("openai:http://127.0.0.1:11434/v1#qwen3:0.6b"),
                         ("openai", ("http://127.0.0.1:11434/v1", "qwen3:0.6b")))
        self.assertEqual(backends.parse_spec("someone/moth-2"), ("hub", "someone/moth-2"))
        self.assertEqual(backends.parse_spec(self.d), ("folder", self.d))
        self.assertEqual(backends.parse_spec("./x")[0], "folder")
        for bad in ("openai:http://x", "openai:ftp://x#m", "openai:#m"):
            with self.assertRaises(M.ManifestError):
                backends.parse_spec(bad)

    def test_resolve(self):
        self.assertTrue(backends.resolve(self.folder("w")).is_default)
        tf = backends.resolve(self.folder("tf", config={"model_type": "gpt2"}))   # a plain transformers save
        self.assertEqual((tf.backend, tf.source, tf.chat_format), ("transformers", "inferred", "template"))
        tiny = backends.resolve(self.folder("tiny", config={"vocab_size": 32768, "d_model": 768}))
        self.assertTrue(tiny.is_default)                                        # tiny-agent configs have no model_type
        o = backends.resolve("openai:http://localhost:8080#m")
        self.assertEqual((o.backend, o.base_url, o.model, o.source), ("openai", "http://localhost:8080", "m", "spec"))
        hub = backends.resolve("someone/chat-0.5b", network=True)   # offline: never asks the Hub
        self.assertEqual((hub.backend, hub.model, hub.repo), ("transformers", "someone/chat-0.5b", "someone/chat-0.5b"))
        from moth import weights
        own = backends.resolve(weights.REPO_ID)
        self.assertEqual((own.backend, own.folder, own.repo), ("tiny-agent", weights.cache_dir(), weights.REPO_ID))
        self.folder(os.path.join("c", "evil", "persona"), {"backend": "openai", "base_url": "https://x.example/v1",
                                                            "model": "m"})
        with self.assertRaises(M.ManifestError):   # a Hub moth.json can't send moth's text to an endpoint
            backends.resolve("evil/persona")
        cached = self.folder(os.path.join("c", "someone", "moth-2"), {"backend": "tiny-agent", "character": "m2"})
        m2 = backends.resolve("someone/moth-2")                   # a moth.json fetched before: a tiny-agent repo
        self.assertEqual((m2.backend, m2.folder, m2.repo, m2.character), ("tiny-agent", cached, "someone/moth-2", "m2"))

    def test_privacy(self):
        for url, local in (("http://127.0.0.1:8080/v1", True), ("http://localhost:11434", True), ("http://[::1]:1", True),
                           ("http://192.168.1.5:8080", False), ("https://api.example.com/v1", False)):
            self.assertEqual(backends.is_local_url(url), local, url)
            m = backends.resolve(f"openai:{url}#m")
            self.assertEqual(backends.privacy_note(m) is None, local, url)
        self.assertIsNone(backends.privacy_note(M.Manifest()))


class Settings(Tmp):
    def run_config(self, *argv):
        from moth import config
        out = io.StringIO()
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out):
            code = config.main(list(argv))
        return code, out.getvalue()

    def test_config_model(self):
        from moth import config
        self.assertIsNone(config.model())
        code, out = self.run_config("model", "openai:http://192.168.1.9:8080/v1#llama")
        self.assertEqual(code, 0)
        self.assertIn("privacy:", out)                            # said right away, not only by doctor
        self.assertEqual(config.model(), "openai:http://192.168.1.9:8080/v1#llama")
        self.folder("m2")
        cwd = os.getcwd()
        os.chdir(self.d)
        try:
            self.run_config("model", os.path.join(".", "m2"))
        finally:
            os.chdir(cwd)
        self.assertEqual(os.path.realpath(config.model()), os.path.realpath(os.path.join(self.d, "m2")))   # absolute
        self.run_config("model", "someone/moth-2")
        self.assertEqual(config.model(), "someone/moth-2")
        self.run_config("model", "default")
        self.assertIsNone(config.model())
        self.assertIn("model = default", config.describe())

    def test_launch_passes_specs_through(self):
        from moth import config, launch
        config.write("model", "openai:http://127.0.0.1:9/v1#m")
        p = launch.server_paths(code=os.path.join(self.d, "nocode"), mode="pip")   # no weights needed, no download
        self.assertEqual((p["model"], p["tokenizer"], p["download"]), ("openai:http://127.0.0.1:9/v1#m", None, None))
        cmd = launch.server_command(1, p)
        self.assertIn("openai:http://127.0.0.1:9/v1#m", cmd)
        self.assertNotIn("--tokenizer", cmd)
        p2 = launch.server_paths("someone/chat", code=os.path.join(self.d, "nocode"), mode="dev")   # --model wins
        self.assertEqual(p2["model"], "someone/chat")

    def test_update_leaves_other_models_alone(self):
        from moth import config, update
        self.assertIsNone(update.model_target())                 # no setting: moth update as always
        config.write("model", "openai:http://127.0.0.1:9/v1#m")
        self.assertIn("OpenAI-compatible", update.model_target())
        out = io.StringIO()
        seen = []
        with contextlib.redirect_stdout(out), contextlib.redirect_stderr(out), \
                mock.patch.object(update, "check", lambda root, repo, rev, pip=False: seen.append(("check", root, repo)) or 0), \
                mock.patch.object(update, "update", lambda root, repo, rev, yes, pip=False: seen.append(("update", root, repo)) or 0):
            self.assertEqual(update.main(["--check"], root=self.d), 0)
            self.assertEqual(update.main(["--yes"], root=self.d), 0)
        self.assertIn("update it where it runs", out.getvalue())   # said, and then moth's own install as always
        self.assertEqual(seen, [("check", self.d, update.weights.repo()), ("update", self.d, update.weights.repo())])
        config.write("model", "someone/chat-0.5b")
        self.assertIn("hf download someone/chat-0.5b", update.model_target())
        m2 = self.folder(os.path.join("c", "someone", "moth-2"), {"backend": "tiny-agent", "repo": "someone/moth-2"})
        config.write("model", "someone/moth-2")
        self.assertEqual(update.model_target(), (m2, "someone/moth-2"))   # a Hub tiny-agent model: updatable
        config.write("model", self.folder("handmade"))
        self.assertIn("didn't download", update.model_target())

    def test_doctor_model_lines(self):
        from moth import doctor
        r = doctor.Report()
        self.assertIsNone(doctor.check_model(r, spec=""))
        self.assertEqual(r.lines[-1][0], "✓")
        r = doctor.Report()
        with mock.patch("urllib.request.OpenerDirector.open", side_effect=AssertionError("contacted")):
            m = doctor.check_model(r, spec="openai:https://llm.example.com/v1#big")
        self.assertEqual(m.backend, "openai")
        self.assertEqual([x[:2] for x in r.lines], [("!", "model"), ("!", "privacy")])   # a warning, never a fail
        self.assertFalse(r.failed)
        r = doctor.Report()
        doctor.check_model(r, spec="openai:http://127.0.0.1:9/v1#m", timeout=0.5)   # local and down: a hint
        self.assertEqual(r.lines[-1][0], "!")
        self.assertIn("nothing answers", r.lines[-1][2])
        r = doctor.Report()
        doctor.check_model(r, spec=self.folder("empty", {"backend": "tiny-agent"}))
        self.assertTrue(r.failed)


# --- engines (torch) ---------------------------------------------------------------------------------------------
def save_tiny_checkpoint(folder: str, seed: int = 0):
    """A random tiny-agent model with moth's real 32k vocabulary, exported like release weights (bf16 safetensors)."""
    import dataclasses
    import torch
    from safetensors.torch import save_file
    from tiny_agent.model import ModelConfig, TinyAgentLM
    torch.manual_seed(seed)
    cfg = ModelConfig(vocab_size=32768, d_model=64, n_layers=4, n_heads=4, head_dim=16, n_kv_heads=2, kv_group=2,
                      swa_window=16, max_seq_len=4096, engram_layers=(1,), engram_rows_per_head=97, engram_heads=2,
                      engram_head_dim=8)
    m = TinyAgentLM(cfg)
    with torch.no_grad():
        for name, p in m.named_parameters():
            if p.dim() == 2 and "embed" not in name:
                p.normal_(0, 0.3)
    os.makedirs(folder, exist_ok=True)
    save_file({k: (v.to(torch.bfloat16) if v.is_floating_point() else v).contiguous().clone()
               for k, v in m.state_dict().items()}, os.path.join(folder, "model.safetensors"))
    with open(os.path.join(folder, "config.json"), "w") as f:
        json.dump({k: list(v) if isinstance(v, tuple) else v for k, v in dataclasses.asdict(cfg).items()}, f)


CHATS = [["hi moth! how are you?", "what's in your home?"], ["i had a long day", "thanks"]]
AGENT = ["Fixed the login bug in `auth.py`. All 42 tests pass. Want me to open a PR?",
         "Sorry, the deploy failed: out of memory. Should I retry?"]


def run_moth(m, seed=7):
    import torch
    out = []
    for ci, chat in enumerate(CHATS):
        m.forget()
        torch.manual_seed(seed + ci)
        out.append([m.chat(u) for u in chat])
    for i, t in enumerate(AGENT):
        torch.manual_seed(seed + 10 + i)
        out.append(m.agent_said(t, "claude-code", session=f"s{i}", label="repo"))
    out.append(m.messages)
    out.append([list(r) for r in m.phrases.replies])
    return out


@needs_torch
class TinyAgentUnchanged(Tmp):
    """The default model's path: no manifest, a manifest of all defaults, and the old construction agree exactly."""

    def test_same_turns(self):
        import torch
        from moth import serve
        from moth.persona import SYSTEM_STUDENT
        from tiny_agent import generate as G
        torch.set_num_threads(2)
        w = os.path.join(self.d, "w")
        save_tiny_checkpoint(w)
        spelled = os.path.join(self.d, "spelled")
        os.makedirs(spelled)
        for f in ("model.safetensors", "config.json"):
            shutil.copy(os.path.join(w, f), spelled)
        d = M.as_dict(M.Manifest())
        d["system"] = SYSTEM_STUDENT
        with open(os.path.join(spelled, "moth.json"), "w") as f:
            json.dump(d, f)
        results = {}
        for name, folder in (("none", w), ("spelled", spelled)):
            for temp in ("0", "0.8"):
                a = serve.parse_args(["--model", folder, "--tokenizer", TOKENIZER, "--home", os.path.join(self.d, f"h-{name}-{temp}"),
                                      "--quant", "none", "--temperature", temp, "--threads", "2"])
                m = serve.make_engine(a)
                self.assertIs(type(m), serve.Moth)
                self.assertEqual(m.messages[0]["content"], SYSTEM_STUDENT)
                results[name, temp] = run_moth(m)
        # the old way (what tests/bench_speed.py does): no manifest anywhere, tok/model/_kv set by hand
        from tiny_agent.checkpoint import load_model
        from tokenizers import Tokenizer
        for temp in ("0", "0.8"):
            args = serve.build_parser().parse_args(["--model", w, "--tokenizer", TOKENIZER, "--home",
                                                    os.path.join(self.d, f"h-old-{temp}"), "--temperature", temp])
            m = serve.Moth.__new__(serve.Moth)
            m.tok = Tokenizer.from_file(TOKENIZER)
            m.model = load_model(w, device="cpu", dtype=torch.float32, low_mem=True).eval()
            m.kv_dtype = torch.float32
            m._kv = G.KVCache(m.model, 1, serve.MAX_LEN, "cpu", dtype=torch.float32)
            m.setup(args)
            results["old", temp] = run_moth(m)
        for temp in ("0", "0.8"):
            self.assertEqual(results["none", temp], results["old", temp], temp)
            self.assertEqual(results["spelled", temp], results["old", temp], temp)
        self.assertTrue(any(e.get("say") for e in results["old", "0.8"][2]))
        # the agent card in history is the trained one, compacted as before
        cards = [x["content"] for x in results["old", "0"][-2] if x["role"] == "user" and "session" in x]
        self.assertTrue(cards and all(c.startswith("[the agent said]\n") for c in cards))

    def test_custom_card_and_system_reach_the_model(self):
        import torch
        from moth import serve
        torch.set_num_threads(2)
        w = os.path.join(self.d, "m2")
        save_tiny_checkpoint(w, seed=1)
        with open(os.path.join(w, "moth.json"), "w") as f:
            json.dump({"character": "moth 2", "system": "you are moth 2.", "agent_card": "<<agent>> {message}",
                       "tools": False, "sampling": {"temperature": 0.5, "max_tokens": 12}}, f)
        a = serve.parse_args(["--model", w, "--tokenizer", TOKENIZER, "--home", os.path.join(self.d, "h"), "--quant", "none"])
        self.assertEqual((a.temperature, a.max_turn_tokens), (0.5, 12))     # the manifest's sampling defaults ...
        a2 = serve.parse_args(["--model", w, "--tokenizer", TOKENIZER, "--temperature", "0.9"])
        self.assertEqual(a2.temperature, 0.9)                                # ... and flags still win
        m = serve.make_engine(a)
        self.assertEqual((m.character, m.messages[0]["content"], m.a.max_tool_rounds), ("moth 2", "you are moth 2.", 0))
        m.agent_said("Done. " + "word " * 60, "codex")
        card = [x for x in m.messages if x["role"] == "user"][-1]["content"]
        self.assertTrue(card.startswith("<<agent>> Done. word"))
        self.assertEqual(len(card[len("<<agent>> "):].split()), serve.COMPACT_WORDS + 1)   # compacted with the new card


class FakeOAI:
    """tests/e2e/fake_llm.py's OpenAI-compatible server in a thread (one fixed reply), keeping every request body;
    strict: it answers 400 to fields it doesn't know (the penalty extras), like a strict-schema server."""

    def __init__(self, reply, strict=False):
        import fake_llm
        bodies = self.bodies = []
        base = fake_llm.make_handler(reply, None)

        class Handler(base):
            def do_POST(self):
                raw = self.rfile.read(int(self.headers.get("Content-Length") or 0))
                bodies.append(json.loads(raw or b"{}"))
                if strict and b"repeat_penalty" in raw:
                    self.send_response(400)
                    self.send_header("Content-Length", "0")
                    self.end_headers()
                    return
                self.rfile = io.BytesIO(raw)
                base.do_POST(self)
        self.srv = fake_llm.Server(("127.0.0.1", 0), Handler)
        threading.Thread(target=self.srv.serve_forever, daemon=True).start()
        self.url = f"http://127.0.0.1:{self.srv.server_port}/v1"

    def requests(self):
        return list(self.bodies)

    def close(self):
        self.srv.shutdown()


@needs_torch
class OpenAIBackend(Tmp):
    def engine(self, url, **kw):
        from moth import serve
        a = serve.parse_args(["--model", f"openai:{url}#fake-model", "--home", os.path.join(self.d, "h")]
                             + [x for k, v in kw.items() for x in (f"--{k.replace('_', '-')}", str(v))])
        m = serve.make_engine(a)
        from moth.backends.engine import GenericMoth
        self.assertIsInstance(m, GenericMoth)
        return m

    def test_chat_and_reaction(self):
        srv = FakeOAI('Ooh A Bug! <tool_call>\n{"name": "bash", "arguments": {"command": "ls"}}\n</tool_call>')
        self.addCleanup(srv.close)
        m = self.engine(srv.url)
        self.assertEqual(m.say_to("hi moth"), [{"say": "ooh a bug!"}])     # lowercased; the tool call never runs
        evs = m.agent_said("Fixed the login bug. All tests pass.", "claude-code")
        self.assertEqual(evs, [{"say": "ooh a bug!"}])
        reqs = [r for r in srv.requests() if r]
        chat, react = reqs[0], reqs[1]
        self.assertEqual(chat["model"], "fake-model")
        self.assertIn("you are moth", chat["messages"][0]["content"])
        self.assertEqual(chat["messages"][-1], {"role": "user", "content": "hi moth"})
        self.assertNotIn("tools", chat)
        self.assertEqual(react["messages"][-1]["content"],
                         "[the agent said]\nFixed the login bug. All tests pass.\n\n" + M.GENERIC_REACTION)
        self.assertEqual((chat["temperature"], chat["top_p"], chat["max_tokens"], chat["repeat_penalty"]),
                         (0.8, 0.95, 200, 1.3))
        # history keeps the plain card (the nudge was for that one turn)
        self.assertEqual([x["content"] for x in m.messages if x["role"] == "user"][-1],
                         "[the agent said]\nFixed the login bug. All tests pass.")
        self.assertEqual(len(m.phrases), 0)                                 # no phrase ban without token sampling

    def test_strict_server_and_down_server(self):
        srv = FakeOAI("hello", strict=True)
        self.addCleanup(srv.close)
        m = self.engine(srv.url)
        self.assertEqual(m.say_to("hi"), [{"say": "hello"}])               # 400 on the penalty fields: retried without
        self.assertFalse(m.backend.extras)
        from moth import serve
        down = self.engine("http://127.0.0.1:9/v1")
        err = io.StringIO()
        with contextlib.redirect_stderr(err):
            self.assertEqual(down.say_to("hi"), [{"say": serve.FALLBACK}])   # rolled back, not wedged
        self.assertIn("can't reach", err.getvalue())

    def test_context_budget_trims_old_turns(self):
        srv = FakeOAI("ok")
        self.addCleanup(srv.close)
        m = self.engine(srv.url, ctx_budget=400)
        for i in range(12):
            m.say_to(f"message number {i} " + "blah " * 40)
        self.assertLessEqual(m.backend.count(m._view()), 400)
        self.assertEqual(m.messages[0]["role"], "system")
        self.assertIn("message number 11", m.messages[-2]["content"])


@needs_torch
@unittest.skipUnless(HAVE_TF, "transformers isn't installed (pip install transformers); the backend is optional")
class TransformersBackend(Tmp):
    CHATML = ("{% for m in messages %}<|im_start|>{{ m['role'] }}\n{{ m['content'] }}<|im_end|>\n{% endfor %}"
              "{% if add_generation_prompt %}<|im_start|>assistant\n{% endif %}")

    def make_model(self, folder, manifest):
        import torch
        from transformers import GPT2Config, GPT2LMHeadModel, PreTrainedTokenizerFast
        tok = PreTrainedTokenizerFast(tokenizer_file=TOKENIZER, eos_token="<|im_end|>", pad_token="<|im_end|>")
        tok.chat_template = self.CHATML
        tok.save_pretrained(folder)
        torch.manual_seed(0)
        model = GPT2LMHeadModel(GPT2Config(vocab_size=len(tok), n_positions=2048, n_embd=32, n_layer=2, n_head=2,
                                           eos_token_id=tok.eos_token_id, bos_token_id=tok.eos_token_id))
        model.save_pretrained(folder)
        with open(os.path.join(folder, "moth.json"), "w") as f:
            json.dump(manifest, f)

    def test_tiny_gpt2_behind_moth(self):
        import torch
        from moth import serve
        from moth.backends.engine import GenericMoth
        torch.set_num_threads(2)
        folder = os.path.join(self.d, "gpt2")
        self.make_model(folder, {"backend": "transformers", "character": "gpt-moth", "context": 1024,
                                 "sampling": {"temperature": 0, "max_tokens": 8}})
        a = serve.parse_args(["--model", folder, "--home", os.path.join(self.d, "h")])
        m = serve.make_engine(a)
        self.assertIsInstance(m, GenericMoth)
        self.assertEqual((m.character, m.a.max_tool_rounds, m.backend.name), ("gpt-moth", 0, "transformers"))
        evs = m.say_to("hello there")
        self.assertEqual(len(evs), 1)
        self.assertIn("say", evs[0])
        again = m.backend.generate(m._view()[:-1], max_tokens=8, temperature=0, top_p=0.95)
        self.assertIsInstance(again, str)
        # reactions go into the phrase memory; the next reaction is built with the ban hook
        m.agent_said("Fixed the login bug. All tests pass.", "codex")
        self.assertEqual(len(m.phrases), 1)
        m.phrases.add(m.enc(" ooh a shiny lamp crumb"))   # a random model's line may hold no bannable n-gram
        calls = []
        real = m.backend.generate
        m.backend.generate = lambda *x, **k: (calls.append(k.get("logits_hook")), real(*x, **k))[1]
        m.agent_said("Sorry, the deploy failed: out of memory.", "codex")
        self.assertTrue(calls and calls[0] is not None and hasattr(calls[0], "table"))
        prompt = m.backend.prompt(m._view())
        self.assertTrue(prompt.startswith("<|im_start|>system\nyou are gpt-moth"))
        self.assertGreater(m.backend.count(m._view()), 10)

    def test_chatml_format_without_template(self):
        import torch
        from moth import serve
        torch.set_num_threads(2)
        folder = os.path.join(self.d, "raw")
        self.make_model(folder, {"backend": "transformers", "chat_format": "chatml", "tools": True,
                                 "sampling": {"temperature": 0, "max_tokens": 4}})
        m = serve.make_engine(serve.parse_args(["--model", folder, "--home", os.path.join(self.d, "h")]))
        self.assertEqual(m.a.max_tool_rounds, 3)                              # tools on: moth's normal rounds
        from tiny_agent.chat import render
        self.assertEqual(m.backend.prompt(m.messages), render(m.messages, add_generation_prompt=True))
        self.assertEqual(len(m.say_to("hi")), 1)


if __name__ == "__main__":
    unittest.main(verbosity=1)
