#!/usr/bin/env node
import { parseArgs } from 'node:util'
import { contextRoom, dataRead, findGpu, findModel, fit, gpus, memory, models, tokensPerSecond, USABLE } from '../src/index.js'

const HELP = `Usage: llm-gpu-memory <model> [options]
       llm-gpu-memory --list

How much GPU memory an open model needs, how many of each GPU hold it,
and the most tokens a second their memory allows.

Options
  -c, --context <tokens>   tokens in each conversation, e.g. 8k, 128k, 1m (default 8k)
  -u, --users <n>          conversations served at once (default 1)
  -w, --weights <format>   released, 16, 8 or 4 (bits; default released)
      --cache <bits>       KV cache: 16 or 8 (default 16)
  -g, --gpu <gpu>          only this GPU, e.g. h100, "rtx 5090"
      --json               print the figures as JSON
  -l, --list               list the models and GPUs
  -h, --help               show this help

Example
  npx llm-gpu-memory "granite 4.2 30b" --context 32k --users 8 --weights 8`

let args
try {
  args = parseArgs({
    allowPositionals: true,
    options: {
      context: { type: 'string', short: 'c', default: '8k' },
      users: { type: 'string', short: 'u', default: '1' },
      weights: { type: 'string', short: 'w', default: 'released' },
      cache: { type: 'string', default: '16' },
      gpu: { type: 'string', short: 'g' },
      json: { type: 'boolean' },
      list: { type: 'boolean', short: 'l' },
      help: { type: 'boolean', short: 'h' },
    },
  })
} catch (error) {
  fail(error.message)
}

function fail(message) {
  console.error(`llm-gpu-memory: ${message}\nRun llm-gpu-memory --help for usage.`)
  process.exit(1)
}

const { values, positionals } = args
const number = (n) => Math.round(n).toLocaleString('en-US')
const bytes = (n) => (n < 1e9 ? `${Number((n / 1e6).toPrecision(3))} MB` : n >= 1e11 ? `${number(n / 1e9)} GB` : `${Number((n / 1e9).toPrecision(3))} GB`)
const params = (n) => (n >= 1e12 ? `${Number((n / 1e12).toPrecision(2))}T` : n >= 1e11 ? `${Math.round(n / 1e9)}B` : `${Number((n / 1e9).toPrecision(3))}B`)
const pad = (text, width) => String(text).padEnd(width)
const lpad = (text, width) => String(text).padStart(width)

if (values.help) {
  console.log(HELP)
  process.exit(0)
}

if (values.list) {
  console.log('Models')
  for (const model of models) console.log(`  ${pad(model.id, 30)} ${pad(model.name, 30)} ${params(model.params)}${model.active ? `, ${params(model.active)} active` : ''}`)
  console.log('\nGPUs')
  for (const gpu of gpus) console.log(`  ${pad(gpu.id, 30)} ${pad(gpu.name, 30)} ${gpu.memory} GB, ${number(gpu.bandwidth)} GB/s`)
  console.log(`\nModel files read from Hugging Face on ${dataRead}.`)
  process.exit(0)
}

if (positionals.length === 0) {
  console.log(HELP)
  process.exit(1)
}

const model = findModel(positionals.join(' '))
if (!model) fail(`no model matches "${positionals.join(' ')}". Run llm-gpu-memory --list to see them.`)

const tokens = (text) => {
  const match = /^(\d+(?:\.\d+)?)\s*([km]?)$/i.exec(text.trim())
  if (!match) fail(`--context takes a number of tokens such as 8192, 32k or 1m, not "${text}"`)
  return Math.round(Number(match[1]) * { '': 1, k: 1024, m: 1024 * 1024 }[match[2].toLowerCase()])
}
const context = Math.min(tokens(values.context), model.context)
const users = Number(values.users)
if (!Number.isInteger(users) || users < 1) fail('--users takes a whole number of 1 or more')
if (!['released', '16', '8', '4'].includes(values.weights)) fail('--weights takes released, 16, 8 or 4')
if (!['16', '8'].includes(values.cache)) fail('--cache takes 16 or 8')

const setup = { weights: values.weights, cache: values.cache, context, users }
const chosen = values.gpu ? [findGpu(values.gpu) ?? fail(`no GPU matches "${values.gpu}". Run llm-gpu-memory --list to see them.`)] : gpus
const need = memory(model, setup)
const rows = chosen.map((gpu) => {
  const place = fit(need.total, gpu)
  return { gpu, ...place, speed: tokensPerSecond(model, setup, gpu, place.count), room: contextRoom(model, setup, gpu, place.count) }
})

if (values.json) {
  console.log(
    JSON.stringify(
      {
        model: model.id,
        setup,
        memory: need,
        gpus: rows.map((row) => ({ gpu: row.gpu.id, count: row.count, servers: row.servers, filled: row.filled, tokensPerSecond: row.speed, maxContext: row.room })),
      },
      null,
      2,
    ),
  )
  process.exit(0)
}

const format = values.weights === 'released' ? `${model.format}, as released` : `${values.weights}-bit`
console.log(`${model.name} (${model.repo}), ${params(model.params)} parameters${model.active ? `, ${params(model.active)} active per token` : ''}`)
console.log(`${users} conversation${users === 1 ? '' : 's'} of ${number(context)} tokens, weights ${format}, ${values.cache}-bit KV cache\n`)
console.log(`  Weights    ${lpad(bytes(need.weights), 9)}`)
console.log(`  KV cache   ${lpad(bytes(need.cache), 9)}`)
console.log(`  Total      ${lpad(bytes(need.total), 9)}\n`)
console.log(`  ${pad('GPU', 26)} ${lpad('Memory', 7)} ${lpad('GPUs', 6)} ${lpad('Tokens/s', 9)}  Longest context`)
for (const row of rows) {
  const count = row.servers > 1 ? `${row.count} (${row.servers} servers)` : String(row.count)
  const speed = row.speed >= 100 ? number(row.speed) : String(Number(row.speed.toPrecision(2)))
  console.log(`  ${pad(row.gpu.name, 26)} ${lpad(`${row.gpu.memory} GB`, 7)} ${lpad(count, 6)} ${lpad(speed, 9)}  ${number(row.room)}`)
}
console.log(`\nGPUs: the fewest that hold it, with ${Math.round((1 - USABLE) * 100)}% of each kept back. Tokens/s: the most each conversation`)
console.log(`can get from memory bandwidth; real speed is lower. Longest context: per conversation, on those GPUs.`)
