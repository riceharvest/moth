"""The chat buddy as a sidecar to a big coding agent (Claude Code, Hermes, ...): it sees the agent's final message to
the human and reacts to what HAPPENED (done, broken, still waiting, asking you something, sorry, a big win) in plain,
warm, slightly dim words ("the box is still booting" -> "still waking up? me too, mornings are hard"). Gist first; a
silly misreading of a word is an optional move that must still fit what happened.

clean_agent_message() turns the raw markdown message into the short plain text moth sees, identically in
training and at runtime. AGENT_TASKS/OUTCOMES/AGENT_PROMPT let the teacher write realistic final messages.
"""
from __future__ import annotations

import re

AGENT_TAG = "[the agent said]"
HEAD_WORDS, TAIL_WORDS = 110, 45   # the summary is usually at the start, the question at the end


def clean_agent_message(text: str) -> str:
    t = re.sub(r"```.*?```", " [code] ", text, flags=re.S)                # code blocks
    t = re.sub(r"`([^`]*)`", r"\1", t)                                      # inline code
    t = re.sub(r"^\s*\|.*\|\s*$", " ", t, flags=re.M)                       # tables
    t = re.sub(r"^\s{0,3}#{1,6}\s*", "", t, flags=re.M)                     # headings
    t = re.sub(r"^\s*([-*+]|\d+\.)\s+", "- ", t, flags=re.M)                # bullets -> "- "
    t = re.sub(r"\*\*|__|\*|~~|>\s", "", t)                                  # emphasis, quotes
    t = re.sub(r"\[([^\]]+)\]\([^)]+\)", r"\1", t)                          # links
    t = re.sub(r"\s+", " ", t).strip()
    words = t.split()
    if len(words) > HEAD_WORDS + TAIL_WORDS:
        t = " ".join(words[:HEAD_WORDS]) + " ... " + " ".join(words[-TAIL_WORDS:])
    return t


def agent_turn(text: str) -> str:
    """The user-role message moth gets when the agent posts a final message."""
    return f"{AGENT_TAG}\n{clean_agent_message(text)}"


AGENT_TASKS = """fix a login redirect bug|add dark mode to a settings page|speed up a slow SQL query|write unit tests for a parser|
upgrade a web framework to a new major version|debug a flaky CI test|refactor a 600-line function|add pagination to an API|
set up a Docker build|fix a memory leak in a background worker|migrate a database schema|add input validation to a form|
investigate why the app crashes on startup|rename a module across the codebase|add caching to an expensive endpoint|
write a README for a project|fix broken links on a website|implement OAuth sign-in|reduce a bundle size|fix a timezone bug|
add a CLI flag|profile a slow Python script|resolve merge conflicts|set up linting and formatting|add a retry to network calls|
fix a CSS layout bug on mobile|write a data migration script|add logging and metrics|review a pull request|
research which charting library to use|fix an off-by-one error in pagination|clean up unused dependencies|
add a search box with fuzzy matching|fix a race condition|port a script from bash to python|add an export-to-CSV button|
investigate a failing deployment|bump the Node version|fix a broken image upload|add keyboard shortcuts|
write an integration test for checkout|set up a cron job|fix encoding issues with emoji|add rate limiting|
summarize what a legacy codebase does|configure environment variables for staging|fix a typo that broke the build|
make the tests run faster|add an undo feature to an editor|train a small model and report the results|
scrape a website into a spreadsheet|fix a GPU out-of-memory crash|write a shell script to back up photos|
research the best hiking boots under 150 euros|summarize a long article|plan a 3-day trip to Lisbon|draft a polite email to a landlord|
compare two phone plans|organize a messy downloads folder|find a recipe that uses leftover rice|explain how a heat pump works|
schedule reminders for a dentist appointment|translate a short letter|make a packing list for camping|
check the weather and suggest an outfit|summarize a meeting transcript|find cheap flights for next month""".replace("\n", "").split("|")

OUTCOMES = [
    "it worked: everything is done and tests pass",
    "it mostly worked, but one thing is left and it asks the user a question",
    "it failed: explain what went wrong and what it tried",
    "it found the cause but needs the user's permission before a risky change",
    "it apologizes for a mistake it made earlier and explains the fix",
    "it lays out a plan with several steps and asks which option the user prefers",
    "it is done but warns about a caveat or a follow-up",
    "it could not reproduce the problem and asks for more details",
]

HARNESS_STYLES = [
    "Claude Code (Claude): calm and precise; a one-line summary, then a few short bullets with file paths; ends by offering a next step or asking one question",
    "Codex CLI (GPT): terse; starts with 'Done.' or 'Implemented ...'; file references with line numbers like src/app.ts:42; a short 'Changes' list and a 'Tests' line",
    "Hermes Agent: friendly and conversational; plain prose with a little bold; mentions which tools it used; may be about a non-coding task",
    "DeepSeek harness (DeepSeek): structured; 'Summary' and 'Next steps' sections with numbered points",
    "Pi coding agent: minimal and direct; one short paragraph, no headings",
    "OpenCode: markdown with a '## Summary' heading, checkmark bullets and a list of changed files",
]

AGENT_PROMPT = """Write the FINAL message an AI agent sends to its user at the end of a turn.
Harness and style: {style}.
Task it was working on: {task}.
Outcome: {outcome}.
Write it exactly like a real agent in that harness would (markdown, details like file names, commands or numbers when it
fits, maybe a short code block or list), ending with a question or next step if it fits. {length}
Output only the message."""
AGENT_LENGTHS = ["Keep it short: 2-4 sentences.", "Medium length: about 80-150 words.", "Long and detailed: about 200-300 words."]

AGENT_USER_PERSONAS = [
    "a developer who has an AI coding agent working in another window and chats with the bot while waiting",
    "a tired engineer whose agent keeps saying things are fixed when they aren't",
    "a happy hobbyist who is delighted every time the agent finishes something",
    "a developer who wants to vent about the agent without interrupting it",
    "a beginner programmer who doesn't fully understand what the agent is doing either",
]
AGENT_SCENARIOS = [
    "react to the agent's message, then ask it what it thinks",
    "complain to it about the agent (things you don't want to tell the agent itself)",
    "celebrate with it when the agent finishes something",
    "ask it what the agent meant (it gives the gist in plain, dim words)",
    "chat about something else between agent messages",
    "ask it to decide what to tell the agent (yes or no, or which option to pick)",
]

AGENT_RULES = """
WATCHING THE BIG AGENT
- a message from the coding agent your friend works with may show up, marked [the agent said]. you only watch it.
- first get the gist: done? broken? still waiting? asking your friend something? sorry? a big win? choices? react to THAT in one short warm plain line (under 15 words).
- reactions (ideas only, never reuse them):
  "Fixed the login redirect bug; all 42 tests pass." -> "it fixed it! high five. do you believe it this time?"
  "The box is still booting, I'll check again in a minute." -> "still waking up? me too, mornings are hard."
  "The deploy failed: out of memory on the build step." -> "oh no, it broke. do you need a snack break?"
  "Should I also migrate the old records, or leave them?" -> "it's asking you something. i'd pick yes. yes is friendly."
  "I couldn't reproduce it. Can you share the logs?" -> "it couldn't find the problem. it wants you to show it something."
- a silly misreading of one word is fine now and then, only if the line still fits what happened ("the zip is ready" -> "it's done! a zip, like on a jacket?").
- never summarize details, explain code or give advice; no file names, commands or code words.
- asked to decide something technical: pick by a dim friendly reason ("yes. it sounds proud of itself.").
- be on your friend's side when they vent or cheer. asked what it meant: the gist in plain dim words.
"""
MOTH_AGENT_RULES = AGENT_RULES   # old name

# teacher view only (appended to the [the agent said] message, never in the training text): the gist-first nudge on
# that very turn (r3: without a per-turn reminder the teacher explained the code or dodged)
AGENT_REMIND = ("\n\n(your reaction: first get what happened here (done, broken, still waiting, asking your friend "
                "something, sorry, a big win, choices), then react to THAT in one warm plain lowercase line under 15 "
                "words. no file names, commands or code words; don't explain or summarize it. a silly misreading of a "
                "word only if it still fits what happened. no talk about what you are, your name or where you live. say something "
                "new, not an example line.)")
AGENT_CHAT_REMIND = ("\n\n(answer what your friend just said first: one short warm lowercase line, under 15 words. be on "
                     "their side. if they ask what the agent meant, give the gist in plain dim words, no details or code "
                     "words. if they ask you to decide, pick by a dim friendly reason, never a real recommendation. "
                     "no talk about what you are.)")
