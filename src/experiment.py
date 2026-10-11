"""Run one neural language model end to end (Part I; reused in Part II).

train -> score the test stream -> generate from the shared prompts -> save
results/<name>.json. Every neural model goes through the same function, so the
training budget, evaluation workload, and decoding settings are identical.
"""
import matplotlib.pyplot as plt

from src.evaluate import (
    RESULTS_DIR, Timer, count_parameters, generate_samples, hardware_summary,
    neural_mean_nll, neural_next_token_probs, perplexity, save_results,
)
from src.train import train_language_model


def run_neural_model(name, model, splits, vocab, training, config=None, description=None):
    """Train, evaluate, generate, and save one model. Return a results dict.

    """
    print(f"Model: {description or name}")
    print(f"Trainable parameters: {count_parameters(model):,}")

    # Train: weights from the best validation epoch are restored at the end
    history = train_language_model(model, splits["train"], splits["valid"], **training)
    best_epoch = history["best_epoch"]
    best_valid = history["valid_loss"][best_epoch - 1]
    seconds_to_best = sum(history["epoch_seconds"][:best_epoch])
    print(f"\nBest epoch: {best_epoch} (valid loss {best_valid:.4f}, "
          f"valid perplexity {perplexity(best_valid):.2f})")
    print(f"Training time: {history['total_seconds']:.1f} s total, "
          f"{seconds_to_best:.1f} s to the best epoch")

    # Test: scored once, on the same workload as the n-gram (targets 1..N-1)
    test_ids = splits["test"]
    n_targets = len(test_ids) - 1
    with Timer() as test_timer:
        test_nll = neural_mean_nll(model, test_ids)
    throughput = n_targets / test_timer.seconds
    print(f"Test NLL: {test_nll:.4f}   Test perplexity: {perplexity(test_nll):.2f}")
    print(f"Scoring time: {test_timer.seconds:.2f} s for {n_targets:,} target tokens, "
          f"{throughput:,.0f} tokens/s")

    # Generate: same prompts, temperature, length, and per-prompt seeds as every model
    samples = generate_samples(neural_next_token_probs(model), vocab)

    results = {
        "model": description or name,
        "config": config,
        "training": training,
        "trainable_parameters": count_parameters(model), # "model size"
        "history": history, # loss curves
        "train_seconds": history["total_seconds"], # "training time"
        "seconds_to_best_epoch": seconds_to_best,
        "best_epoch": best_epoch,
        "valid_nll": best_valid,
        "valid_perplexity": perplexity(best_valid),
        "test_nll": test_nll,
        "test_perplexity": perplexity(test_nll),
        "test_seconds": test_timer.seconds,
        "test_tokens_per_second": throughput,
        "samples": samples,
        "hardware": hardware_summary(),
    }
    path = save_results(name, results)
    print(f"Saved to {path}")
    return results


def plot_history(history, title, filename=None, label_offset=(-150, 30)):
    """Train and validation loss per epoch, with the best-validation epoch marked.

    label_offset: (x, y) shift of the best-epoch label from its point, in points.
    """
    epochs = range(1, len(history["train_loss"]) + 1)
    best = history["best_epoch"]
    best_valid = history["valid_loss"][best - 1]

    fig, ax = plt.subplots(figsize=(7, 4))
    ax.plot(epochs, history["train_loss"], marker="o", markersize=4, color="#2a78d6",
            linewidth=2, label="train")
    ax.plot(epochs, history["valid_loss"], marker="s", markersize=4, color="#eb6834",
            linewidth=2, label="validation")
    ax.axvline(best, color="grey", linestyle="--", linewidth=1)
    ax.annotate(f"best epoch {best}\nvalid loss {best_valid:.3f} (ppl {perplexity(best_valid):.1f})",
                (best, best_valid), textcoords="offset points", xytext=label_offset, fontsize=9)

    ax.set_xlabel("Epoch")
    ax.set_ylabel("Mean next-token cross-entropy")
    ax.set_title(title)
    ax.legend(frameon=False)
    ax.grid(alpha=0.3)
    fig.tight_layout()
    if filename:
        RESULTS_DIR.mkdir(exist_ok=True)
        fig.savefig(RESULTS_DIR / filename, dpi=200, bbox_inches="tight")
    return fig
