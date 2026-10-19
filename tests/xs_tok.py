"""A small byte-level BPE with the XS chat specials at ids 0..31 (tiny_moe/data/sources.SPECIAL_TOKENS), built in
memory for tests: CI has no 48k tokenizer and needs none. Common chat words are single tokens ("Ġgreat")."""
from __future__ import annotations

import functools

SPECIALS = (["<|endoftext|>", "<|pad|>", "<|system|>", "<|user|>", "<|assistant|>", "<|agent|>", "<|end|>"]
            + [f"<|reserved_{i}|>" for i in range(25)])
WORDS = ("i i'm im the a an and or to of it it's its is in on at my me you your we be was are not no yes just so that "
         "this but like with for do don't can can't oh hi hello friend great grand grandma first bug moth cat named "
         "bo fixed tests pass agent said done sorry failed deploy login redirect parser cart total pages book crash "
         "migration birds lamp what why how name day good nice ok yay meow pip houseplant puns little buddy said")


@functools.cache
def xs_tokenizer():
    from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers
    tok = Tokenizer(models.BPE())
    tok.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tok.decoder = decoders.ByteLevel()
    trainer = trainers.BpeTrainer(vocab_size=1200, special_tokens=SPECIALS, min_frequency=1,
                                  initial_alphabet=pre_tokenizers.ByteLevel.alphabet(), show_progress=False)
    corpus = [" ".join(WORDS.split()) + "."] * 20 + [w for w in WORDS.split() for _ in range(5)] + \
             [" " + w for w in WORDS.split() for _ in range(30)]
    tok.train_from_iterator(corpus, trainer)
    assert tok.token_to_id("<|endoftext|>") == 0 and tok.token_to_id("<|end|>") == 6
    return tok
