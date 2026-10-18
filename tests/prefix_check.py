import asyncio, json, sys
sys.path.insert(0, ".")
from openai import AsyncOpenAI
from moth.gen_dialogs import OAI_TOOLS, NO_THINK, SIM_SWITCH, user_turn
from moth.persona import TEACHER_BIBLE, USER_PERSONAS, SCENARIOS
c = AsyncOpenAI(base_url="http://127.0.0.1:8000/v1", api_key="x")
async def main():
    r = await c.chat.completions.create(model="teacher", messages=[{"role": "system", "content": TEACHER_BIBLE}, {"role": "user", "content": "hi"}],
                                        tools=OAI_TOOLS, max_tokens=1, extra_body=NO_THINK)
    print("prefix tokens (system+tools+'hi'):", r.usage.prompt_tokens)
    msgs = [{"role": "user", "content": "hey moth what's up"}, {"role": "assistant", "content": "counting dust. i got to 7.", "tool_calls": []}]
    outs = await asyncio.gather(*[user_turn(c, "teacher", USER_PERSONAS[i], SCENARIOS[i], 4, 2, msgs) for i in range(8)])
    for o in outs: print("SIM:", o)
asyncio.run(main())
