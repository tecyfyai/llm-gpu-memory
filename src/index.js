// How much GPU memory an open model needs to serve: its weights and its KV cache, how many of a GPU hold it,
// and the most tokens a second the GPU's memory bandwidth allows.
import { createRequire } from 'node:module'

const data = createRequire(import.meta.url)('./data.json')

/** The open models, read from each model's Hugging Face files. */
export const models = data.models
/** The GPUs, from each maker's spec page. */
export const gpus = data.gpus
/** The day the models' files were read. */
export const dataRead = data.read

const GB = 1e9
/** Bits a weight takes in each format. 4-bit is NVFP4: 4-bit numbers with an 8-bit scale per block of 16. */
export const WEIGHT_BITS = { 16: 16, 8: 8, 4: 4.5 }
/** Bytes a cached number takes: 16-bit, or an 8-bit (FP8) KV cache. */
export const CACHE_BYTES = { 16: 2, 8: 1 }
/** The share of each GPU's memory given to the model, as vLLM's default gpu_memory_utilization. */
export const USABLE = 0.92
/** GPUs in 1 server. A model is split across the GPUs of a server first, then across servers. */
export const PER_SERVER = 8

/**
 * @typedef {object} Setup
 * @property {'released'|'16'|'8'|'4'} [weights] weight format: as released (default), 16, 8 or 4-bit
 * @property {'16'|'8'} [cache] KV cache format (default 16)
 * @property {number} [context] tokens in each conversation, prompt and answer (default 8192)
 * @property {number} [users] conversations served at once (default 1)
 */

const defaults = (setup = {}) => ({ weights: 'released', cache: '16', context: 8192, users: 1, ...setup })

/** Find a model by id or by name, ignoring case, spaces and dots ("granite 4.2 30b", "granite-4-2-30b"). */
export function findModel(query) {
  const key = (text) => String(text).toLowerCase().replace(/[^a-z0-9]/g, '')
  const wanted = key(query)
  return models.find((model) => model.id === query || key(model.id) === wanted || key(model.name) === wanted) ?? models.find((model) => key(model.name).includes(wanted))
}

/** Find a GPU by id or name ("h100", "RTX 5090"). */
export function findGpu(query) {
  const key = (text) => String(text).toLowerCase().replace(/[^a-z0-9]/g, '')
  const wanted = key(query)
  return gpus.find((gpu) => gpu.id === query || key(gpu.id) === wanted || key(gpu.name) === wanted) ?? gpus.find((gpu) => key(gpu.name).includes(wanted))
}

/** Bytes of the weights in a format. */
export function weightBytes(model, format = 'released') {
  return format === 'released' ? model.weights : (model.params * WEIGHT_BITS[format]) / 8
}

/**
 * Bytes of the KV cache of 1 conversation of `context` tokens: every token for full and latent attention
 * layers, at most the window for sliding-window layers, plus the fixed state of linear attention or Mamba layers.
 */
export function cacheBytes(model, context, format = '16') {
  const values = model.groups.reduce((sum, group) => {
    const kept = group.kind === 'window' ? Math.min(context, group.window ?? context) : context
    return sum + group.layers * group.values * kept
  }, 0)
  return values * CACHE_BYTES[format] + model.stateBytes
}

/** Weights, KV cache for all conversations, and their total, in bytes. */
export function memory(model, setup) {
  const s = defaults(setup)
  const weights = weightBytes(model, s.weights)
  const cache = cacheBytes(model, s.context, s.cache) * s.users
  return { weights, cache, total: weights + cache }
}

/** The fewest GPUs that hold `bytes`: 1, 2, 4 or 8, then whole servers of 8. */
export function fit(bytes, gpu) {
  const usable = gpu.memory * GB * USABLE
  const need = Math.max(1, Math.ceil(bytes / usable))
  const count = need <= PER_SERVER ? 2 ** Math.ceil(Math.log2(need)) : Math.ceil(need / PER_SERVER) * PER_SERVER
  return { count, servers: Math.ceil(count / PER_SERVER), filled: bytes / (usable * count) }
}

/**
 * The most tokens a second each conversation can get on `count` GPUs: memory bandwidth over the bytes read
 * for each token (the active weights once, plus each conversation's cache). Servers in a pipeline take turns,
 * so only the GPUs of 1 server add speed. Real speed is lower.
 */
export function tokensPerSecond(model, setup, gpu, count) {
  const s = defaults(setup)
  const share = model.active ? model.active / model.params : 1
  const read = weightBytes(model, s.weights) * share + cacheBytes(model, s.context, s.cache) * s.users
  return (gpu.bandwidth * GB * Math.min(count, PER_SERVER)) / read
}

/** The longest context each conversation can have on `count` GPUs, up to the model's own; 0 if the weights do not fit. */
export function contextRoom(model, setup, gpu, count) {
  const s = defaults(setup)
  const room = gpu.memory * GB * USABLE * count - weightBytes(model, s.weights)
  if (room <= 0) return 0
  const fits = (tokens) => cacheBytes(model, tokens, s.cache) * s.users <= room
  if (fits(model.context)) return model.context
  let low = 0
  let high = model.context
  while (high - low > 1) {
    const middle = Math.floor((low + high) / 2)
    if (fits(middle)) low = middle
    else high = middle
  }
  return low
}

/** Everything for 1 model and setup on 1 GPU. */
export function estimate(modelQuery, gpuQuery, setup) {
  const model = typeof modelQuery === 'string' ? findModel(modelQuery) : modelQuery
  if (!model) throw new Error(`No model matches "${modelQuery}"`)
  const gpu = typeof gpuQuery === 'string' ? findGpu(gpuQuery) : gpuQuery
  if (!gpu) throw new Error(`No GPU matches "${gpuQuery}"`)
  const s = defaults(setup)
  const need = memory(model, s)
  const place = fit(need.total, gpu)
  return {
    model: model.id,
    gpu: gpu.id,
    setup: s,
    memory: need,
    gpus: place.count,
    servers: place.servers,
    filled: place.filled,
    tokensPerSecond: tokensPerSecond(model, s, gpu, place.count),
    maxContext: contextRoom(model, s, gpu, place.count),
  }
}
