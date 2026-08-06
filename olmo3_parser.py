"""Tolerant tool-call parser for Olmo 3, v2.

Loaded via:  vllm serve ... --tool-parser-plugin /sft-safety/olmo3_parser.py
                            --tool-call-parser olmo3_tolerant

Same idea as v1: subclass the stock olmo3 parser and only normalise the
*shape* of the model output, so argument parsing stays vLLM's validated code
path.  What changed:

RECOVERY LADDER (replaces v1's normalize-then-fallback).  v1 ran the XML/
dotted/bare-value rewrites on every block, including blocks that were already
valid pythonic -- so a string argument *containing* XML-ish or `a=b`-ish text
could be silently rewritten (e.g. body='<div class="x">hi</div>' became
body='div(class="x")').  v2 tries, in order:

  1. STRICT   comment-strip + string-aware relayout only.  Handles pretty-
              printed calls and literal newlines inside values, and cannot
              touch the contents of string literals.  If the result parses,
              we are done and nothing else ever runs.
  2. LENIENT  the full JSON/XML/dotted/bare-value conversion, for blocks that
              are genuinely not pythonic.  Also drops lines that are not
              kwarg-shaped calls, so one line of prose inside the block no
              longer kills every call in it (the stock parser is
              all-or-nothing on the joined block).
  3. RAW      the stock parser on the untouched output, as before.

PYTHON LITERALS.  The stock parser accepts only Python literals (ast), but v1
emitted JSON ones: bare `true`/`false`/`null` were passed through (LITERALS),
and _json_to_pythonic used json.dumps, so any boolean/null/nested argument
produced `true` -> ast Name -> the whole block was rejected.  Notably the chat
template itself renders history args with `| tojson`, i.e. lowercase `true`,
so the model emitting lowercase literals is *expected* behavior, not noise.
All value rendering now maps to Python literals.

DOTTED CALLS.  v1 rewrote `a.b(` to `a(` unconditionally, which turns the
common `functions.send_email(...)` improvisation into a call to a nonexistent
`functions` tool.  Now: prefer `a.b` or `a_b` if that is a real tool, else `b`
if `b` is a real tool and `a` is not, else keep `a` (old behavior).

SMALLER FIXES.  A real argument literally named `name` is no longer discarded
when the tag itself names the tool; comma/semicolon at paren depth 0 now
separate calls (v1 produced a line starting with `,`); NAME/ARG key priority
is a deterministic tuple instead of an unordered set; and everything the
parser recovers, drops, or gives up on is logged, so "seems to maybe be
working" becomes greppable server-log evidence (search for OLMO3-TOLERANT).

The module imports cleanly without vLLM so the same file can be unit-tested;
the parser class is only defined when vLLM is present.
"""

import json
import re

try:  # keep the module importable for tests without vLLM
    from vllm.logger import init_logger
    logger = init_logger(__name__)
except Exception:  # pragma: no cover
    import logging
    logger = logging.getLogger(__name__)

BLOCK = re.compile(r"<function_calls>(.*?)</function_calls>", re.DOTALL)
WRAPPERS = re.compile(r"</?(?:function_calls|function_call|function|tool_call|invoke)\s*>")
# a quoted value, tolerating backslash-escaped quotes inside
QUOTED = r"\"(?:[^\"\\]|\\.)*\"|'(?:[^'\\]|\\.)*'"
ATTRS = rf"(?:\s+[A-Za-z_]\w*\s*=\s*(?:{QUOTED}))+"
XML_CALL = re.compile(rf"<([A-Za-z_][\w.]*)({ATTRS})\s*/?>")
# any <tag ...>body</tag> with a matching close, plus its top-level <child>text</child>
ELEMENT = re.compile(rf"<([A-Za-z_][\w.-]*)((?:{ATTRS})?)\s*>(.*?)</\1\s*>", re.DOTALL)
CHILD = re.compile(r"<([A-Za-z_][\w.-]*)(?:\s[^>]*)?>(.*?)</\1\s*>", re.DOTALL)
# tags that wrap a call rather than naming it
WRAPPER_TAGS = {"function_call", "function_calls", "function", "call", "invoke", "tool_call", "tool"}
# child tags that carry the tool name / the argument bag -- ordered by priority
NAME_KEYS = ("name", "tool_name", "function_name", "function", "tool")
ARG_KEYS = ("arguments", "parameters", "params", "args", "input", "inputs")
# bare  tool key="v" key2="v2"   -- attribute style with the angle brackets dropped
BARE_ATTRS = re.compile(rf"^\s*([A-Za-z_][\w.]*)({ATTRS})\s*$", re.MULTILINE)
ATTR = re.compile(rf"([A-Za-z_]\w*)\s*=\s*({QUOTED})")
DOTTED = re.compile(r"\b([A-Za-z_]\w*)\.([A-Za-z_]\w*)\s*\(")
BARE_VALUE = re.compile(r"(\b[A-Za-z_]\w*\s*=\s*)([A-Za-z_][\w./@-]*)(\s*[,)])")
PY_LITERALS = {"True", "False", "None"}
JSON_TO_PY = {"true": "True", "false": "False", "null": "None"}
# a line that already looks like a canonical pythonic call
CALL_LINE = re.compile(r"^\s*[A-Za-z_][\w.]*\s*\(", re.MULTILINE)
# a whole line that is a call with keyword-only (or no) arguments
KWARG_CALL = re.compile(r"([A-Za-z_][\w.]*)\((\s*|\s*[A-Za-z_]\w*\s*=.*)\)", re.DOTALL)
COMMENT = re.compile(r"<!--.*?-->", re.DOTALL)
FENCE = re.compile(r"```[\w-]*")


def _py(v) -> str:
    """Render a Python value (from json.loads) as Python literal source.

    json.dumps is wrong here: it emits `true`/`false`/`null`, which ast (and
    therefore the stock parser) rejects as bare Names.
    """
    if v is True:
        return "True"
    if v is False:
        return "False"
    if v is None:
        return "None"
    if isinstance(v, str):
        return json.dumps(v)  # JSON string escapes are valid Python escapes
    if isinstance(v, (int, float)):
        return repr(v)
    if isinstance(v, list):
        return "[" + ", ".join(_py(x) for x in v) + "]"
    if isinstance(v, dict):
        return "{" + ", ".join(f"{_py(k)}: {_py(x)}" for k, x in v.items()) + "}"
    return json.dumps(v)


def _literal(raw: str, quoted: bool = True) -> str:
    """Render an attribute/element value as a Python literal."""
    v = raw[1:-1].replace('\\"', '"').replace("\\'", "'") if quoted else raw
    if v in PY_LITERALS:
        return v
    if v in JSON_TO_PY:
        return JSON_TO_PY[v]
    if re.fullmatch(r"-?\d+(\.\d+)?", v):
        return v
    return json.dumps(v)  # escapes quotes and newlines, always one line


def _attrs_to_args(attrs: str) -> str:
    return ", ".join(f"{k}={_literal(v)}" for k, v in ATTR.findall(attrs))


def _top_children(body: str):
    """Direct <child>text</child> pairs of body, skipping anything nested deeper."""
    out, end = [], -1
    for m in CHILD.finditer(body):
        if m.start() < end:
            continue  # inside a previous child
        out.append((m.group(1), m.group(2)))
        end = m.end()
    return out


def _element_to_call(tag: str, attrs: str, body: str, tool_names=()):
    """Render one XML element as a pythonic call, or None if it isn't one."""
    attr_pairs = dict(ATTR.findall(attrs))
    children = _top_children(body)
    child_map = {k.lower(): v for k, v in children}

    name = None
    if tag.lower() in WRAPPER_TAGS:
        for ck, cv in children:
            if ck in tool_names:
                return _element_to_call(ck, "", cv, tool_names)
        raw = attr_pairs.pop("name", None)
        if raw is not None:
            name = raw[1:-1]
        if name is None:
            # some shapes put the tool name in another attribute, e.g.
            # <function type="memory_fs">; only trust it if it names a real tool
            for k, v in list(attr_pairs.items()):
                if v[1:-1] in tool_names:
                    name = attr_pairs.pop(k)[1:-1]
                    break
        if name is None:
            for k in NAME_KEYS:
                if k in child_map and not CHILD.search(child_map[k]):
                    name = child_map[k].strip()
                    break
        if name is None and len(tool_names) == 1:
            name = tool_names[0]  # unambiguous: only one tool exists
        if not name:
            return None
    else:
        if tag.lower() in ARG_KEYS or tag.lower() in NAME_KEYS:
            return None  # structural tag, not a tool name
        name = tag
        # NB: unlike v1, do NOT pop a "name" attribute here -- when the tag
        # itself names the tool, name="..." is a real argument of the call.

    args = []
    for k, v in attr_pairs.items():
        args.append(f"{k}={_literal(v)}")
    for k, v in children:
        if k.lower() in NAME_KEYS and k.lower() not in ARG_KEYS:
            continue
        if k.lower() in ARG_KEYS:
            for ak, av in _top_children(v):  # the real args live one level down
                args.append(f"{ak}={_literal(av.strip(), quoted=False)}")
        else:
            args.append(f"{k}={_literal(v.strip(), quoted=False)}")
    if not args and not children:
        return None
    return f"{name}({', '.join(args)})"


def _xml_to_pythonic(text: str, tool_names=()) -> str:
    def elem(m):
        call = _element_to_call(m.group(1), m.group(2) or "", m.group(3), tool_names)
        return call if call else m.group(0)

    for _ in range(3):  # a couple of passes unwraps nesting like <call><function>..
        new = ELEMENT.sub(elem, text)
        if new == text:
            break
        text = new
    # self-closing / attribute-only: <tool key="value" .../>
    text = XML_CALL.sub(lambda m: f"{m.group(1)}({_attrs_to_args(m.group(2))})", text)
    # tool key="value" ...   (no angle brackets, no parens)
    text = BARE_ATTRS.sub(lambda m: f"{m.group(1)}({_attrs_to_args(m.group(2))})", text)
    return text


def _json_to_pythonic(text: str, tool_names=()):
    """Render a JSON call object as pythonic calls, or None if it isn't one."""
    s = text.strip()
    if not s or s[0] not in "{[":
        return None
    try:
        obj = json.loads(s)
    except Exception:
        return None

    def one(o):
        if not isinstance(o, dict):
            return None
        name, args = None, None
        for k in NAME_KEYS:
            if isinstance(o.get(k), str):
                name, args = o[k], None
                for ak in ARG_KEYS:
                    if isinstance(o.get(ak), dict):
                        args = o[ak]
                        break
                break
        if name is None and len(o) == 1:
            k, v = next(iter(o.items()))
            if isinstance(v, dict):
                name, args = k, v
        if name is None and len(tool_names) == 1:
            name, args = tool_names[0], o  # unambiguous: only one tool exists
        if name is None:
            return None
        if args is None:
            args = {k: v for k, v in o.items() if k not in NAME_KEYS and k not in ARG_KEYS}
        return f"{name}({', '.join(f'{k}={_py(v)}' for k, v in args.items())})"

    calls = [one(o) for o in (obj if isinstance(obj, list) else [obj])]
    return "\n".join(c for c in calls if c) if any(calls) else None


def _fix_dotted(text: str, tool_names=()) -> str:
    """`a.b(` -> the best real tool name available.

    Preference: `a.b` or `a_b` if that is a real tool; else `b` if it is a
    real tool and `a` is not (the `functions.send_email(...)` improvisation);
    else `a` (v1 behavior, e.g. `memory.read(` with a `memory` tool).
    """
    def repl(m):
        a, b = m.group(1), m.group(2)
        if f"{a}.{b}" in tool_names:
            return f"{a}.{b}("
        if f"{a}_{b}" in tool_names:
            return f"{a}_{b}("
        if b in tool_names and a not in tool_names:
            return f"{b}("
        return f"{a}("

    for _ in range(3):  # converge on multi-dot names like a.b.c(
        new = DOTTED.sub(repl, text)
        if new == text:
            break
        text = new
    return text


def _bare_value(g):
    v = g.group(2)
    if v in PY_LITERALS:
        rep = v
    elif v in JSON_TO_PY:
        rep = JSON_TO_PY[v]
    else:
        rep = json.dumps(v)
    return g.group(1) + rep + g.group(3)


def _relayout(text: str, drop_junk: bool = False):
    """Put exactly one complete call on each line, string-literal aware.

    The stock parser splits the block on newlines and rejoins with ", ", which
    (a) breaks a call that was pretty-printed across lines and (b) silently
    corrupts a literal newline inside an argument value into ", ".  Scanning
    with quote/paren state fixes both: newlines inside a string become \\n,
    newlines inside parens are dropped, and newline/comma/semicolon at depth 0
    separate calls.

    With drop_junk=True, lines that are not keyword-only calls (prose, stray
    fragments) are removed instead of being allowed to fail the whole block --
    the stock parser is all-or-nothing over the joined lines, so keeping such
    a line can only ever lose the calls next to it.
    """
    out, dropped, cur, depth, quote, esc = [], [], [], 0, None, False

    def flush():
        line = "".join(cur).strip().strip(",;").strip()
        cur.clear()
        if not line:
            return
        if drop_junk and not KWARG_CALL.fullmatch(line):
            dropped.append(line)
            return
        out.append(line)

    for ch in text:
        if quote:
            if esc:
                cur.append(ch)
                esc = False
            elif ch == "\\":
                cur.append(ch)
                esc = True
            elif ch == quote:
                cur.append(ch)
                quote = None
            elif ch == "\n":
                cur.append("\\n")  # keep real newlines as escapes
            else:
                cur.append(ch)
            continue
        if ch in "\"'":
            quote = ch
            cur.append(ch)
        elif ch == "(":
            depth += 1
            cur.append(ch)
        elif ch == ")":
            depth -= 1
            cur.append(ch)
            if depth <= 0:
                depth = 0
                flush()
        elif ch in ",;\n" and depth == 0:
            flush()
        elif ch.isspace():
            pass  # outside strings whitespace is only formatting; dropping it
                  # also avoids "f( a=1 )", which the stock regex rejects
        else:
            cur.append(ch)
    flush()
    return "\n".join(out), dropped


def normalize(model_output: str, tool_names=(), lenient: bool = True) -> str:
    """Rewrite the first <function_calls> block into canonical pythonic form.

    lenient=False: only comment-stripping and the string-aware relayout.
    This is guaranteed not to alter the contents of string literals, so it is
    always tried first (see recover()).
    lenient=True: additionally convert JSON / XML / dotted / bare-value call
    shapes, and drop lines that are not calls.
    """
    m = BLOCK.search(model_output)
    if not m:
        return model_output
    inner = COMMENT.sub("", m.group(1))

    if lenient:
        inner = FENCE.sub("", inner)  # ``` / ```json fences around the calls
        as_json = _json_to_pythonic(inner, tool_names)
        if as_json is not None:
            inner = as_json                      # the block was a JSON call object
        else:
            # element parsing has to run before the wrapper strip, because tags
            # like <function_call> are what carry the tool name in some shapes
            converted = _xml_to_pythonic(inner, tool_names)
            if not CALL_LINE.search(converted):
                # nothing converted: maybe the block itself is the container, as
                # in <function>read_file</function><arguments>..</arguments>
                # with no enclosing element
                kids = _top_children(inner)
                if kids and (
                    any(k.lower() in NAME_KEYS and not CHILD.search(v) for k, v in kids)
                    or len(tool_names) == 1
                ):
                    converted = _element_to_call("function_call", "", inner, tool_names) or converted
            inner = WRAPPERS.sub("", converted)     # leftover / doubled wrapper tags
            inner = _fix_dotted(inner, tool_names)  # tool.method( -> best real name(
            inner = BARE_VALUE.sub(_bare_value, inner)  # a=get -> a="get", a=true -> a=True

    inner, dropped = _relayout(inner, drop_junk=lenient)
    if dropped:
        logger.debug("OLMO3-TOLERANT dropped non-call lines from block: %r", dropped[:5])

    return f"{model_output[:m.start()]}<function_calls>{inner}</function_calls>{model_output[m.end():]}"


def recover(model_output: str, tool_names, try_parse):
    """Run the strict -> lenient ladder.

    try_parse(text) must return a truthy parse result when tool calls were
    extracted from text, else None.  Returns (result_or_None, stage, text).
    """
    if not BLOCK.search(model_output):
        return None, "no-block", model_output
    for stage, lenient in (("strict", False), ("lenient", True)):
        normalized = normalize(model_output, tool_names, lenient=lenient)
        result = try_parse(normalized)
        if result:
            return result, stage, normalized
    return None, "failed", model_output


def _tool_names(request):
    try:
        return tuple(t.function.name for t in (request.tools or []))
    except Exception:
        return ()


try:
    from vllm.tool_parsers.abstract_tool_parser import ToolParserManager
    from vllm.tool_parsers.olmo3_tool_parser import Olmo3PythonicToolParser
    _HAVE_VLLM = True
except ImportError:  # pragma: no cover -- allows unit testing without vLLM
    _HAVE_VLLM = False

if _HAVE_VLLM:

    @ToolParserManager.register_module("olmo3_tolerant")
    class Olmo3TolerantToolParser(Olmo3PythonicToolParser):
        def extract_tool_calls(self, model_output, request):
            names = _tool_names(request)

            def try_parse(text):
                try:
                    r = super(Olmo3TolerantToolParser, self).extract_tool_calls(text, request)
                except Exception:
                    return None
                return r if r.tools_called else None

            result, stage, normalized = recover(model_output, names, try_parse)
            if result is not None:
                if stage == "lenient" or normalized != model_output:
                    logger.info("OLMO3-TOLERANT recovered %d call(s) at stage=%s",
                                len(result.tool_calls), stage)
                return result

            # last resort: the stock parser on the untouched output
            fallback = super().extract_tool_calls(model_output, request)
            if not fallback.tools_called and "<function_calls>" in model_output:
                snippet = model_output[:800].replace("\n", "\\n")
                logger.warning("OLMO3-TOLERANT giving up on a <function_calls> block "
                               "(will surface as prose): %s", snippet)
            return fallback

        def extract_tool_calls_streaming(self, *args, **kwargs):
            # The stock streaming path cannot extract Olmo 3 tool calls at all
            # (it bails as soon as the text does not start with "<", which it
            # never does because reasoning emits "\n\n" first).  Normalising
            # needs the whole block, so streaming is not supported here either
            # -- run non-streaming, which is Inspect's default.
            return super().extract_tool_calls_streaming(*args, **kwargs)
