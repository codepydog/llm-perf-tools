from llm_perf_tools import plot_eval_result


def main():
    inference_path = "eval_results/sglang_batch_metrics.json"
    gpu_path = "eval_results/sglang_gpu_metrics.csv"

    plots = plot_eval_result(inference_path, gpu_path)

    plots[0].savefig("inference_metrics.png")
    plots[1].savefig("gpu_metrics.png")

    print("Plots saved as PNG files")


if __name__ == "__main__":
    main()
