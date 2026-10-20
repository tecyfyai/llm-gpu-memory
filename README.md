# llm-gpu-memory

[![npm](https://img.shields.io/npm/v/llm-gpu-memory)](https://www.npmjs.com/package/llm-gpu-memory)
[![licence](https://img.shields.io/npm/l/llm-gpu-memory)](LICENSE)

How much GPU memory an open LLM needs to serve, how many of each GPU hold it, and the most tokens a second their memory allows.

## Install

Needs [Node.js](https://nodejs.org) 20 or later. Run it without installing:

```sh
npx llm-gpu-memory "granite 4.2 30b" --context 32k --users 8 --weights 8
```

Or install the command, or add the library to a project:

```sh
npm install -g llm-gpu-memory
npm install llm-gpu-memory
```

The package is [llm-gpu-memory on npm](https://www.npmjs.com/package/llm-gpu-memory).

## Example

Granite 4.2 30B with 8-bit weights, serving 8 conversations of 32K tokens at once:

```
Granite 4.2 30B (ibm-granite/granite-4.2-30b), 29.3B parameters
8 conversations of 32,768 tokens, weights 8-bit, 16-bit KV cache

  Weights      29.3 GB
  KV cache     68.7 GB
  Total          98 GB

  GPU                         Memory   GPUs  Tokens/s  Longest context
  GeForce RTX 5090             32 GB      4        73  42,192
  L4                           24 GB      8        24  70,268
  L40S                         48 GB      4        35  70,268
  RTX PRO 6000 Blackwell       96 GB      2        37  70,268
  DGX Spark                   128 GB      1       2.8  42,192
  A100 80GB                    80 GB      2        42  56,230
  H100                         80 GB      2        68  56,230
  H200                        141 GB      1        49  47,895
  B200                        180 GB      1        82  65,003
  Instinct MI300X             192 GB      1        54  70,268
  Instinct MI325X             256 GB      1        61  98,344
  Instinct MI355X             288 GB      1        82  112,382
```

The same sums run the [GPU memory calculator](https://tecyfy.com/ai-agents/gpu-memory-calculator) on tecyfy.com, which has a page for each model.

## Options

| Option | Meaning | Default |
| --- | --- | --- |
| `-c, --context <tokens>` | Tokens in each conversation, prompt and answer: `8192`, `32k`, `1m` | `8k` |
| `-u, --users <n>` | Conversations served at once | `1` |
| `-w, --weights <format>` | `released` (the format of the model's files), `16`, `8` or `4` bits | `released` |
| `--cache <bits>` | KV cache in `16` or `8` bits | `16` |
| `-g, --gpu <gpu>` | Only this GPU, such as `h100` or `"rtx 5090"` | every GPU |
| `--json` | Print the figures as JSON | |
| `-l, --list` | List the models and GPUs | |

A model is found by its id or name, so `"qwen3.5 9b"` and `qwen3-5-9b` both work.

## In code

```js
import { estimate } from 'llm-gpu-memory'

const result = estimate('granite-4-2-30b', 'h100', { context: 32768, users: 1, weights: '4' })
// { memory: { weights, cache, total }, gpus: 1, tokensPerSecond: 133.., maxContext: 131072, ... }
```

The package also exports `models`, `gpus`, `findModel`, `findGpu`, `memory`, `cacheBytes`, `weightBytes`, `fit`, `tokensPerSecond` and `contextRoom`. Sizes are in bytes. No dependencies.

## How it counts

- **Weights.** As released, the size of the model's weight files on Hugging Face. At 16 or 8 bits, 2 or 1 bytes per parameter. At 4 bits, NVFP4: 4-bit numbers with an 8-bit scale for each block of 16, so 4.5 bits.
- **KV cache.** Counted layer by layer from the model's config, because models differ: layers with full attention keep every token, sliding-window layers keep only their window, multi-head latent attention keeps one compressed vector per token, and linear attention or Mamba layers keep a fixed state per conversation. Times the number of conversations.
- **GPUs.** The fewest that hold the total, in counts a model is split across: 1, 2, 4 or 8 in a server, then whole servers of 8. 8% of each GPU's memory is kept back, as vLLM does by default (`gpu_memory_utilization=0.92`).
- **Tokens a second.** Each new token reads the weights it uses (the active experts only, for a mixture of experts) and each conversation's cache, so memory bandwidth divided by those bytes is the ceiling. GPUs in 1 server read their parts at the same time; servers in a pipeline take turns, so they add memory, not speed.
- **Longest context.** The longest conversation that still fits on those GPUs, up to the model's own limit.

## Limits

- It is an estimate for planning. Activations, CUDA graphs and the serving engine's own buffers come out of the 8% kept back, which is not exact for every setup.
- Tokens a second is a ceiling. Real servers reach part of it, less when many conversations run at once or the model is split across GPUs.
- Only open models whose attention it can count are listed. Run `--list` to see them.

## Data

The models' parameters, layers and file sizes are read from each model's Hugging Face repository; the GPUs' memory and bandwidth from each maker's spec page (`src/data.json`, with the page for each GPU).

## Licence

[MIT](LICENSE)
