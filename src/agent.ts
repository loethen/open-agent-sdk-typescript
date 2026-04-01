/**
 * Agent - High-level API
 *
 * Provides createAgent() and query() interfaces compatible with
 * open-agent-sdk.
 *
 * Usage:
 *   import { createAgent } from 'open-agent-sdk'
 *   const agent = createAgent({ model: 'claude-sonnet-4-6' })
 *   for await (const event of agent.query('Hello')) { ... }
 */

import type {
  AgentOptions,
  QueryResult,
  SDKMessage,
  ToolDefinition,
  CanUseToolFn,
  Message,
  TokenUsage,
  PermissionMode,
} from './types.js'
import { QueryEngine } from './engine.js'
import { getAllBaseTools, filterTools } from './tools/index.js'
import { connectMCPServer, type MCPConnection } from './mcp/client.js'
import { isSdkServerConfig } from './sdk-mcp-server.js'
import { registerAgents } from './tools/agent-tool.js'
import {
  saveSession,
  loadSession,
} from './session.js'
import type Anthropic from '@anthropic-ai/sdk'

// --------------------------------------------------------------------------
// Agent class
// --------------------------------------------------------------------------

export class Agent {
  private cfg: AgentOptions
  private toolPool: ToolDefinition[]
  private modelId: string
  private apiCredentials: { key?: string; baseUrl?: string }
  private mcpLinks: MCPConnection[] = []
  private history: Anthropic.MessageParam[] = []
  private messageLog: Message[] = []
  private setupDone: Promise<void>
  private sid: string
  private abortCtrl: AbortController | null = null
  private currentEngine: QueryEngine | null = null

  constructor(options: AgentOptions = {}) {
    // Shallow copy to avoid mutating caller's object
    this.cfg = { ...options }

    // Merge credentials from options.env map, direct options, and process.env
    this.apiCredentials = this.pickCredentials()
    this.modelId = this.cfg.model ?? this.readEnv('CODEANY_MODEL') ?? 'claude-sonnet-4-6'
    this.sid = this.cfg.sessionId ?? crypto.randomUUID()

    // The underlying @anthropic-ai/sdk reads ANTHROPIC_API_KEY from process.env,
    // so we bridge our resolved credentials into it.
    if (this.apiCredentials.key) {
      process.env.ANTHROPIC_API_KEY = this.apiCredentials.key
    }
    if (this.apiCredentials.baseUrl) {
      process.env.ANTHROPIC_BASE_URL = this.apiCredentials.baseUrl
    }

    // Build tool pool from options (supports ToolDefinition[], string[], or preset)
    this.toolPool = this.buildToolPool()

    // Kick off async setup (MCP connections, agent registration, session resume)
    this.setupDone = this.setup()
  }

  /** Pick API key and base URL from options or CODEANY_* env vars. */
  private pickCredentials(): { key?: string; baseUrl?: string } {
    const envMap = this.cfg.env
    return {
      key:
        this.cfg.apiKey ??
        envMap?.CODEANY_API_KEY ??
        envMap?.CODEANY_AUTH_TOKEN ??
        this.readEnv('CODEANY_API_KEY') ??
        this.readEnv('CODEANY_AUTH_TOKEN'),
      baseUrl:
        this.cfg.baseURL ??
        envMap?.CODEANY_BASE_URL ??
        this.readEnv('CODEANY_BASE_URL'),
    }
  }

  /** Read a value from process.env (returns undefined if missing). */
  private readEnv(key: string): string | undefined {
    return process.env[key] || undefined
  }

  /** Assemble the available tool set based on options. */
  private buildToolPool(): ToolDefinition[] {
    const raw = this.cfg.tools
    let pool: ToolDefinition[]

    if (!raw || (typeof raw === 'object' && !Array.isArray(raw) && 'type' in raw)) {
      pool = getAllBaseTools()
    } else if (Array.isArray(raw) && raw.length > 0 && typeof raw[0] === 'string') {
      pool = filterTools(getAllBaseTools(), raw as string[])
    } else {
      pool = raw as ToolDefinition[]
    }

    return filterTools(pool, this.cfg.allowedTools, this.cfg.disallowedTools)
  }

  /**
   * Async initialization: connect MCP servers, register agents, resume sessions.
   */
  private async setup(): Promise<void> {
    // Register custom agent definitions
    if (this.cfg.agents) {
      registerAgents(this.cfg.agents)
    }

    // Connect MCP servers (supports stdio, SSE, HTTP, and in-process SDK servers)
    if (this.cfg.mcpServers) {
      for (const [name, config] of Object.entries(this.cfg.mcpServers)) {
        try {
          if (isSdkServerConfig(config)) {
            // In-process SDK MCP server - directly add tools
            this.toolPool = [...this.toolPool, ...config.tools]
          } else {
            // External MCP server
            const connection = await connectMCPServer(name, config)
            this.mcpLinks.push(connection)

            if (connection.status === 'connected' && connection.tools.length > 0) {
              this.toolPool = [...this.toolPool, ...connection.tools]
            }
          }
        } catch (err: any) {
          console.error(`[MCP] Failed to connect to "${name}": ${err.message}`)
        }
      }
    }

    // Resume or continue session
    if (this.cfg.resume) {
      const sessionData = await loadSession(this.cfg.resume)
      if (sessionData) {
        this.history = sessionData.messages
        this.sid = this.cfg.resume
      }
    }
  }

  /**
   * Run a query with streaming events.
   */
  async *query(
    prompt: string,
    overrides?: Partial<AgentOptions>,
  ): AsyncGenerator<SDKMessage, void> {
    await this.setupDone

    const opts = { ...this.cfg, ...overrides }
    const cwd = opts.cwd || process.cwd()

    // Create abort controller for this query
    this.abortCtrl = opts.abortController || new AbortController()
    if (opts.abortSignal) {
      opts.abortSignal.addEventListener('abort', () => this.abortCtrl?.abort(), { once: true })
    }

    // Resolve systemPrompt (handle preset object)
    let systemPrompt: string | undefined
    let appendSystemPrompt = opts.appendSystemPrompt
    if (typeof opts.systemPrompt === 'object' && opts.systemPrompt?.type === 'preset') {
      systemPrompt = undefined // Use engine default (default style)
      if (opts.systemPrompt.append) {
        appendSystemPrompt = (appendSystemPrompt || '') + '\n' + opts.systemPrompt.append
      }
    } else {
      systemPrompt = opts.systemPrompt as string | undefined
    }

    // Build canUseTool based on permission mode
    const permMode = opts.permissionMode ?? 'bypassPermissions'
    const canUseTool: CanUseToolFn = opts.canUseTool ?? (async (_tool, _input) => {
      if (permMode === 'bypassPermissions' || permMode === 'dontAsk' || permMode === 'auto') {
        return { behavior: 'allow' }
      }
      if (permMode === 'acceptEdits') {
        return { behavior: 'allow' }
      }
      return { behavior: 'allow' }
    })

    // Resolve tools with overrides
    let tools = this.toolPool
    if (overrides?.allowedTools || overrides?.disallowedTools) {
      tools = filterTools(tools, overrides.allowedTools, overrides.disallowedTools)
    }
    if (overrides?.tools) {
      const ot = overrides.tools
      if (Array.isArray(ot) && ot.length > 0 && typeof ot[0] === 'string') {
        tools = filterTools(this.toolPool, ot as string[])
      } else if (Array.isArray(ot)) {
        tools = ot as ToolDefinition[]
      }
    }

    // Create query engine with current conversation state
    const engine = new QueryEngine({
      cwd,
      model: opts.model || this.modelId,
      apiKey: this.apiCredentials.key,
      baseURL: this.apiCredentials.baseUrl,
      tools,
      systemPrompt,
      appendSystemPrompt,
      maxTurns: opts.maxTurns ?? 10,
      maxBudgetUsd: opts.maxBudgetUsd,
      maxTokens: opts.maxTokens ?? 16384,
      thinking: opts.thinking,
      jsonSchema: opts.jsonSchema,
      canUseTool,
      includePartialMessages: opts.includePartialMessages ?? false,
      abortSignal: this.abortCtrl.signal,
      agents: opts.agents,
    })
    this.currentEngine = engine

    // Inject existing conversation history
    for (const msg of this.history) {
      (engine as any).messages.push(msg)
    }

    // Run the engine
    for await (const event of engine.submitMessage(prompt)) {
      yield event

      // Track assistant messages for multi-turn persistence
      if (event.type === 'assistant') {
        const uuid = crypto.randomUUID()
        const timestamp = new Date().toISOString()
        this.messageLog.push({
          type: 'assistant',
          message: event.message,
          uuid,
          timestamp,
        })
      }
    }

    // Persist conversation state for multi-turn
    this.history = engine.getMessages()

    // Add user message to tracked messages
    const userUuid = crypto.randomUUID()
    this.messageLog.push({
      type: 'user',
      message: { role: 'user', content: prompt },
      uuid: userUuid,
      timestamp: new Date().toISOString(),
    })
  }

  /**
   * Convenience method: send a prompt and collect the final answer as a single object.
   * Internally iterates through the streaming query and aggregates the outcome.
   */
  async prompt(
    text: string,
    overrides?: Partial<AgentOptions>,
  ): Promise<QueryResult> {
    const t0 = performance.now()
    const collected = { text: '', turns: 0, tokens: { in: 0, out: 0 } }

    for await (const ev of this.query(text, overrides)) {
      switch (ev.type) {
        case 'assistant': {
          // Extract the last assistant text (multi-turn: only final answer matters)
          const fragments = ev.message.content
            .filter((c): c is Anthropic.TextBlock => c.type === 'text')
            .map((c) => c.text)
          if (fragments.length) collected.text = fragments.join('')
          break
        }
        case 'result':
          collected.turns = ev.num_turns ?? 0
          collected.tokens.in = ev.usage?.input_tokens ?? 0
          collected.tokens.out = ev.usage?.output_tokens ?? 0
          break
      }
    }

    return {
      text: collected.text,
      usage: { input_tokens: collected.tokens.in, output_tokens: collected.tokens.out },
      num_turns: collected.turns,
      duration_ms: Math.round(performance.now() - t0),
      messages: [...this.messageLog],
    }
  }

  /**
   * Get conversation messages.
   */
  getMessages(): Message[] {
    return [...this.messageLog]
  }

  /**
   * Reset conversation history.
   */
  clear(): void {
    this.history = []
    this.messageLog = []
  }

  /**
   * Interrupt the current query.
   */
  async interrupt(): Promise<void> {
    this.abortCtrl?.abort()
  }

  /**
   * Change the model during a session.
   */
  async setModel(model?: string): Promise<void> {
    if (model) {
      this.modelId = model
      this.cfg.model = model
    }
  }

  /**
   * Change the permission mode during a session.
   */
  async setPermissionMode(mode: PermissionMode): Promise<void> {
    this.cfg.permissionMode = mode
  }

  /**
   * Set maximum thinking tokens.
   */
  async setMaxThinkingTokens(maxThinkingTokens: number | null): Promise<void> {
    if (maxThinkingTokens === null) {
      this.cfg.thinking = { type: 'disabled' }
    } else {
      this.cfg.thinking = { type: 'enabled', budgetTokens: maxThinkingTokens }
    }
  }

  /**
   * Get the session ID.
   */
  getSessionId(): string {
    return this.sid
  }

  /**
   * Stop a background task.
   */
  async stopTask(taskId: string): Promise<void> {
    const { getTask } = await import('./tools/task-tools.js')
    const task = getTask(taskId)
    if (task) {
      task.status = 'cancelled'
    }
  }

  /**
   * Close MCP connections and clean up.
   * Optionally persist session to disk.
   */
  async close(): Promise<void> {
    // Persist session if enabled
    if (this.cfg.persistSession !== false && this.history.length > 0) {
      try {
        await saveSession(this.sid, this.history, {
          cwd: this.cfg.cwd || process.cwd(),
          model: this.modelId,
          summary: undefined,
        })
      } catch {
        // Session persistence is best-effort
      }
    }

    for (const conn of this.mcpLinks) {
      await conn.close()
    }
    this.mcpLinks = []
  }
}

// --------------------------------------------------------------------------
// Factory function
// --------------------------------------------------------------------------

/** Factory: shorthand for `new Agent(options)`. */
export function createAgent(options: AgentOptions = {}): Agent {
  return new Agent(options)
}

// --------------------------------------------------------------------------
// Standalone query — one-shot convenience wrapper
// --------------------------------------------------------------------------

/**
 * Execute a single agentic query without managing an Agent instance.
 * The agent is created, used, and cleaned up automatically.
 */
export async function* query(params: {
  prompt: string
  options?: AgentOptions
}): AsyncGenerator<SDKMessage, void> {
  const ephemeral = createAgent(params.options)
  try {
    yield* ephemeral.query(params.prompt)
  } finally {
    await ephemeral.close()
  }
}
