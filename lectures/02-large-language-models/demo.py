"""Walk through Lecture 02 and print every intermediate result.

The script compares RMSNorm with LayerNorm, shows that RoPE scores depend
only on relative position, sizes the KV cache under MHA, GQA and MQA,
routes tokens through a mixture of experts, compares decoding strategies
and finally generates from a small decoder-only model.

Run it from the repository root::

    uv run python lectures/02-large-language-models/demo.py
"""

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "01-transformers"))  # reused: l01
sys.path.insert(0, str(Path(__file__).resolve().parent))

from l01.transformer import LayerNorm
from l02.attention import kv_cache_bytes, receptive_field, sliding_window_mask
from l02.decoding import beam_search, greedy_search, temperature_softmax, top_k_filter, top_p_filter
from l02.model import DecoderOnlyLM, LMConfig
from l02.moe import MixtureOfExperts, load_balancing_loss
from l02.normalization import RMSNorm
from l02.rope import apply_rope


def section(title: str) -> None:
    """Print a section header."""
    print(f"\n{'=' * 8} {title} {'=' * (60 - len(title))}")


def demo_normalization() -> None:
    """Compare LayerNorm and RMSNorm on the lecture's worked example."""
    section("1. LayerNorm vs RMSNorm")
    x = np.array([1.0, 2.0, 3.0, 6.0])
    print("x          =", x)
    print("LayerNorm  =", LayerNorm(4, eps=0.0)(x), "(centred)")
    print("RMSNorm    =", RMSNorm(4, eps=0.0)(x), "(only rescaled)")


def demo_rope() -> None:
    """Show that RoPE attention scores depend only on the distance between tokens."""
    section("2. RoPE: scores depend on relative position")
    rng = np.random.default_rng(0)
    q, k = rng.normal(size=(2, 1, 64))
    for m, n in [(5, 2), (105, 102), (1005, 1002), (5, 4)]:
        score = (apply_rope(q, np.array([m])) @ apply_rope(k, np.array([n])).T).item()
        print(f"  query at {m:4}, key at {n:4} (distance {m - n}): score {score:+.4f}")


def demo_attention() -> None:
    """Size the KV cache of a Mistral-7B-like model and show the sliding-window reach."""
    section("3. KV cache: MHA vs GQA vs MQA (32 layers, d_head 128, bf16)")
    for name, kv_heads in [("MHA", 32), ("GQA", 8), ("MQA", 1)]:
        per_token = kv_cache_bytes(32, kv_heads, 128, num_tokens=1)
        total = kv_cache_bytes(32, kv_heads, 128, num_tokens=32 * 1024)
        print(
            f"  {name} ({kv_heads:2} KV heads): {per_token // 1024:4} KiB/token, "
            f"{total / 1024**3:5.1f} GiB for 32K tokens"
        )
    print("Sliding window of 4: earliest input reachable from position 63")
    mask = sliding_window_mask(64, window=4)
    for layers in [1, 2, 4, 8]:
        earliest = receptive_field(mask, layers)[63].nonzero()[0].min()
        print(f"  after {layers} layer(s): position {earliest}")


def demo_moe() -> None:
    """Route a batch of tokens through a mixture of experts."""
    section("4. Mixture of experts (8 experts, top-2)")
    moe = MixtureOfExperts(
        d_model=32, d_ff=64, num_experts=8, top_k=2, rng=np.random.default_rng(0)
    )
    x = np.random.default_rng(1).normal(size=(64, 32))
    _, routing = moe(x)
    load = np.bincount(routing.experts.ravel(), minlength=8)
    print("tokens per expert:   ", load.tolist())
    print(
        "load-balancing loss: ",
        round(load_balancing_loss(routing.probs, routing.experts, 8), 3),
        "(1.0 = perfectly balanced, 8.0 = collapsed)",
    )
    print(f"parameters: {moe.num_parameters():,} stored, {moe.num_active_parameters():,} per token")


def demo_decoding() -> None:
    """Compare temperatures, top-k/top-p filters, and greedy vs beam search."""
    section("5. Decoding")
    logits = np.array([3.0, 2.0, 1.0, 0.0])
    for t in [0.5, 1.0, 2.0]:
        print(f"  T = {t}: {temperature_softmax(logits, t)}")
    probs = np.array([0.50, 0.25, 0.15, 0.07, 0.03])
    print("  top-k (k=2):   ", top_k_filter(probs, 2))
    print("  top-p (p=0.85):", top_p_filter(probs, 0.85))

    # Midterm II.9: P(a)=.6, P(b)=.4, P(u|a)=P(v|a)=.5, P(u|b)=.9, P(v|b)=.1
    names = ["[BOS]", "a", "b", "u", "v", "[EOS]"]
    table = {(0,): {1: 0.6, 2: 0.4}, (0, 1): {3: 0.5, 4: 0.5}, (0, 2): {3: 0.9, 4: 0.1}}

    def log_probs(prefix):
        p = np.zeros(6)
        for token, prob in table.get(tuple(prefix), {5: 1.0}).items():
            p[token] = prob
        with np.errstate(divide="ignore"):
            return np.log(p)

    for label, (tokens, lp) in [
        ("greedy", greedy_search(log_probs, 0, 5, max_len=5)),
        ("beam=2", beam_search(log_probs, 0, 5, beam_size=2, max_len=5)),
    ]:
        print(f"  {label}: {' '.join(names[t] for t in tokens)}  p = {np.exp(lp):.2f}")


def demo_model() -> None:
    """Build dense and MoE decoder-only models and generate from them."""
    section("6. A decoder-only LM (random weights)")
    dense = DecoderOnlyLM(LMConfig(vocab_size=100, d_model=32, num_layers=2, d_ff=64))
    moe = DecoderOnlyLM(
        LMConfig(vocab_size=100, d_model=32, num_layers=2, d_ff=64, num_experts=8, top_k=2)
    )
    for name, model in [("dense", dense), ("MoE", moe)]:
        print(
            f"  {name:5}: {model.num_parameters():,} parameters, "
            f"{model.num_active_parameters():,} active per token"
        )
    prompt = [1, 17, 42]
    print(
        "  next-token loss on the prompt:",
        round(dense.next_token_loss(np.array(prompt)), 3),
        f"(log V = {np.log(100):.3f})",
    )
    for t in [0.0, 1.0]:
        out = dense.generate(prompt, max_new_tokens=6, rng=np.random.default_rng(0), temperature=t)
        print(f"  generate, T = {t}: {out}")


def main() -> None:
    """Run every section of the demo."""
    np.set_printoptions(precision=3, suppress=True)
    demo_normalization()
    demo_rope()
    demo_attention()
    demo_moe()
    demo_decoding()
    demo_model()


if __name__ == "__main__":
    main()
