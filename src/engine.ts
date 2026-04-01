/**
 * QueryEngine - Core agentic loop
 *
 * Manages the full conversation lifecycle:
 * 1. Take user prompt
 * 2. Build system prompt with context (git status, project context, tools)
 * 3. Call LLM API with tools
 * 4. Stream response
 * 5. Execute tool calls (concurrent for read-only, serial for mutations)
 * 6. Send results back, repeat until done
 * 7. Auto-compact when context exceeds threshold
 * 8. Retry with exponential backoff on transient errors
 */

import Anthropic from '@anthropic-ai/sdk'
import type {
  SDKMessage,
  QueryEngineConfig,
  ToolDefinition,
  ToolResult,
  ToolContext,
  TokenUsage,
} from './types.js'
import { toApiTool } from './tools/types.js'
import {
  estimateMessagesTokens,
  estimateCost,
  getAutoCompactThreshold,
} from './utils/tokens.js'
import {
  shouldAutoCompact,
  compactConversation,
  microCompactMessages,
  createAutoCompactState,
  type AutoCompactState,
} from './utils/compact.js'
import {
  withRetry,
  isPromptTooLongError,
  formatApiError,
} from './utils/retry.js'
import { getSystemContext, getUserContext } from './utils/context.js'
import { normalizeMessagesForAPI } from './utils/messages.js'

// ============================================================================
// System Prompt Builder
// ============================================================================

async function buildSystemPrompt(config: QueryEngineConfig): Promise<string> {
  if (config.systemPrompt) {
    const base = config.systemPrompt
    return config.appendSystemPrompt
      ? base + '\n\n' + config.appendSystemPrompt
      : base
  }

  const parts: string[] = []

  parts.push(
    'You are an AI assistant with access to tools. Use the tools provided to help the user accomplish their tasks.',
    'You should use tools when they would help you complete the task more accurately or efficiently.',
  )

  // List available tools with descriptions
  parts.push('\n# Available Tools\n')
  for (const tool of config.tools) {
    parts.push(`- **${tool.name}**: ${tool.description}`)
  }

  // Add agent definitions
  if (config.agents && Object.keys(config.agents).length > 0) {
    parts.push('\n# Available Subagents\n')
    for (const [name, def] of Object.entries(config.agents)) {
      parts.push(`- **${name}**: ${def.description}`)
    }
  }

  // System context (git status, etc.)
  try {
    const sysCtx = await getSystemContext(config.cwd)
    if (sysCtx) {
      parts.push('\n# Environment\n')
      parts.push(sysCtx)
    }
  } catch {
    // Context is best-effort
  }

  // User context (AGENT.md, date)
  try {
    const userCtx = await getUserContext(config.cwd)
    if (userCtx) {
      parts.push('\n# Project Context\n')
      parts.push(userCtx)
    }
  } catch {
    // Context is best-effort
  }

  // Working directory
  parts.push(`\n# Working Directory\n${config.cwd}`)

  if (config.appendSystemPrompt) {
    parts.push('\n' + config.appendSystemPrompt)
  }

  return parts.join('\n')
}

// ============================================================================
// QueryEngine
// ============================================================================

export class QueryEngine {
  private config: QueryEngineConfig
  private client: Anthropic
  public messages: Anthropic.MessageParam[] = []
  private totalUsage: TokenUsage = { input_tokens: 0, output_tokens: 0 }
  private totalCost = 0
  private turnCount = 0
  private compactState: AutoCompactState
  private sessionId: string
  private apiTimeMs = 0

  constructor(config: QueryEngineConfig) {
    this.config = config
    this.client = new Anthropic({
      apiKey: config.apiKey,
      baseURL: config.baseURL,
    })
    this.compactState = createAutoCompactState()
    this.sessionId = crypto.randomUUID()
  }

  /**
   * Submit a user message and run the agentic loop.
   * Yields SDKMessage events as the agent works.
   */
  async *submitMessage(
    prompt: string | Anthropic.ContentBlockParam[],
  ): AsyncGenerator<SDKMessage> {
    // Add user message
    this.messages.push({ role: 'user', content: prompt })

    // Build tool definitions for API
    const tools = this.config.tools.map(toApiTool)

    // Build system prompt
    const systemPrompt = await buildSystemPrompt(this.config)

    // Emit init system message
    yield {
      type: 'system',
      subtype: 'init',
      session_id: this.sessionId,
      tools: this.config.tools.map(t => t.name),
      model: this.config.model,
      cwd: this.config.cwd,
      mcp_servers: [],
      permission_mode: 'bypassPermissions',
    } as SDKMessage

    // Agentic loop
    let turnsRemaining = this.config.maxTurns
    let budgetExceeded = false
    let maxOutputRecoveryAttempts = 0
    const MAX_OUTPUT_RECOVERY = 3

    while (turnsRemaining > 0) {
      if (this.config.abortSignal?.aborted) break

      // Check budget
      if (this.config.maxBudgetUsd && this.totalCost >= this.config.maxBudgetUsd) {
        budgetExceeded = true
        break
      }

      // Auto-compact if context is too large
      if (shouldAutoCompact(this.messages, this.config.model, this.compactState)) {
        try {
          const result = await compactConversation(
            this.client,
            this.config.model,
            this.messages,
            this.compactState,
          )
          this.messages = result.compactedMessages
          this.compactState = result.state
        } catch {
          // Continue with uncompacted messages
        }
      }

      // Micro-compact: truncate large tool results
      const apiMessages = microCompactMessages(
        normalizeMessagesForAPI(this.messages),
      )

      this.turnCount++
      turnsRemaining--

      // Make API call with retry
      let response: Anthropic.Message
      const apiStart = performance.now()
      try {
        response = await withRetry(
          async () => {
            const requestParams: Anthropic.MessageCreateParamsNonStreaming = {
              model: this.config.model,
              max_tokens: this.config.maxTokens,
              system: systemPrompt,
              messages: apiMessages,
              tools: tools.length > 0 ? tools : undefined,
            }

            // Add thinking if configured
            if (
              this.config.thinking?.type === 'enabled' &&
              this.config.thinking.budgetTokens
            ) {
              (requestParams as any).thinking = {
                type: 'enabled',
                budget_tokens: this.config.thinking.budgetTokens,
              }
            }

            return this.client.messages.create(requestParams)
          },
          undefined,
          this.config.abortSignal,
        )
      } catch (err: any) {
        // Handle prompt-too-long by compacting
        if (isPromptTooLongError(err) && !this.compactState.compacted) {
          try {
            const result = await compactConversation(
              this.client,
              this.config.model,
              this.messages,
              this.compactState,
            )
            this.messages = result.compactedMessages
            this.compactState = result.state
            turnsRemaining++ // Retry this turn
            this.turnCount--
            continue
          } catch {
            // Can't compact, give up
          }
        }

        yield {
          type: 'result',
          subtype: 'error',
          usage: this.totalUsage,
          num_turns: this.turnCount,
          cost: this.totalCost,
        }
        return
      }

      // Track API timing
      this.apiTimeMs += performance.now() - apiStart

      // Track usage
      if (response.usage) {
        this.totalUsage.input_tokens += response.usage.input_tokens
        this.totalUsage.output_tokens += response.usage.output_tokens
        if ('cache_creation_input_tokens' in response.usage) {
          this.totalUsage.cache_creation_input_tokens =
            (this.totalUsage.cache_creation_input_tokens || 0) +
            ((response.usage as any).cache_creation_input_tokens || 0)
        }
        if ('cache_read_input_tokens' in response.usage) {
          this.totalUsage.cache_read_input_tokens =
            (this.totalUsage.cache_read_input_tokens || 0) +
            ((response.usage as any).cache_read_input_tokens || 0)
        }
        this.totalCost += estimateCost(this.config.model, response.usage as TokenUsage)
      }

      // Add assistant message to conversation
      this.messages.push({ role: 'assistant', content: response.content })

      // Yield assistant message
      yield {
        type: 'assistant',
        message: {
          role: 'assistant',
          content: response.content,
        },
      }

      // Handle max_output_tokens recovery
      if (
        response.stop_reason === 'max_tokens' &&
        maxOutputRecoveryAttempts < MAX_OUTPUT_RECOVERY
      ) {
        maxOutputRecoveryAttempts++
        // Add continuation prompt
        this.messages.push({
          role: 'user',
          content: 'Please continue from where you left off.',
        })
        continue
      }

      // Check for tool use
      const toolUseBlocks = response.content.filter(
        (block): block is Anthropic.ToolUseBlock => block.type === 'tool_use',
      )

      if (toolUseBlocks.length === 0) {
        break // No tool calls - agent is done
      }

      // Reset max_output recovery counter on successful tool use
      maxOutputRecoveryAttempts = 0

      // Execute tools (concurrent read-only, serial mutations)
      const toolResults = await this.executeTools(toolUseBlocks)

      // Yield tool results
      for (const result of toolResults) {
        yield {
          type: 'tool_result',
          result: {
            tool_use_id: result.tool_use_id,
            tool_name: result.tool_name || '',
            output:
              typeof result.content === 'string'
                ? result.content
                : JSON.stringify(result.content),
          },
        }
      }

      // Add tool results to conversation
      this.messages.push({
        role: 'user',
        content: toolResults.map((r) => ({
          type: 'tool_result' as const,
          tool_use_id: r.tool_use_id,
          content:
            typeof r.content === 'string'
              ? r.content
              : JSON.stringify(r.content),
          is_error: r.is_error,
        })),
      })

      if (response.stop_reason === 'end_turn') break
    }

    // Yield enriched final result
    const endSubtype = budgetExceeded
      ? 'error_max_budget_usd'
      : turnsRemaining <= 0
        ? 'error_max_turns'
        : 'success'

    yield {
      type: 'result',
      subtype: endSubtype,
      session_id: this.sessionId,
      is_error: endSubtype !== 'success',
      num_turns: this.turnCount,
      total_cost_usd: this.totalCost,
      duration_api_ms: Math.round(this.apiTimeMs),
      usage: this.totalUsage,
      model_usage: { [this.config.model]: { input_tokens: this.totalUsage.input_tokens, output_tokens: this.totalUsage.output_tokens } },
      cost: this.totalCost,
    }
  }

  /**
   * Execute tool calls with concurrency control.
   *
   * Read-only tools run concurrently (up to 10 at a time).
   * Mutation tools run sequentially.
   */
  private async executeTools(
    toolUseBlocks: Anthropic.ToolUseBlock[],
  ): Promise<(ToolResult & { tool_name?: string })[]> {
    const context: ToolContext = {
      cwd: this.config.cwd,
      abortSignal: this.config.abortSignal,
    }

    const MAX_CONCURRENCY = parseInt(
      process.env.AGENT_SDK_MAX_TOOL_CONCURRENCY || '10',
    )

    // Partition into read-only (concurrent) and mutation (serial)
    const readOnly: Array<{ block: Anthropic.ToolUseBlock; tool?: ToolDefinition }> = []
    const mutations: Array<{ block: Anthropic.ToolUseBlock; tool?: ToolDefinition }> = []

    for (const block of toolUseBlocks) {
      const tool = this.config.tools.find((t) => t.name === block.name)
      if (tool?.isReadOnly?.()) {
        readOnly.push({ block, tool })
      } else {
        mutations.push({ block, tool })
      }
    }

    const results: (ToolResult & { tool_name?: string })[] = []

    // Execute read-only tools concurrently (batched by MAX_CONCURRENCY)
    for (let i = 0; i < readOnly.length; i += MAX_CONCURRENCY) {
      const batch = readOnly.slice(i, i + MAX_CONCURRENCY)
      const batchResults = await Promise.all(
        batch.map((item) =>
          this.executeSingleTool(item.block, item.tool, context),
        ),
      )
      results.push(...batchResults)
    }

    // Execute mutation tools sequentially
    for (const item of mutations) {
      const result = await this.executeSingleTool(item.block, item.tool, context)
      results.push(result)
    }

    return results
  }

  /**
   * Execute a single tool with permission checking.
   */
  private async executeSingleTool(
    block: Anthropic.ToolUseBlock,
    tool: ToolDefinition | undefined,
    context: ToolContext,
  ): Promise<ToolResult & { tool_name?: string }> {
    if (!tool) {
      return {
        type: 'tool_result',
        tool_use_id: block.id,
        content: `Error: Unknown tool "${block.name}"`,
        is_error: true,
        tool_name: block.name,
      }
    }

    // Check enabled
    if (tool.isEnabled && !tool.isEnabled()) {
      return {
        type: 'tool_result',
        tool_use_id: block.id,
        content: `Error: Tool "${block.name}" is not enabled`,
        is_error: true,
        tool_name: block.name,
      }
    }

    // Check permissions
    if (this.config.canUseTool) {
      try {
        const permission = await this.config.canUseTool(tool, block.input)
        if (permission.behavior === 'deny') {
          return {
            type: 'tool_result',
            tool_use_id: block.id,
            content: permission.message || `Permission denied for tool "${block.name}"`,
            is_error: true,
            tool_name: block.name,
          }
        }
        if (permission.updatedInput !== undefined) {
          block = { ...block, input: permission.updatedInput as Record<string, unknown> }
        }
      } catch (err: any) {
        return {
          type: 'tool_result',
          tool_use_id: block.id,
          content: `Permission check error: ${err.message}`,
          is_error: true,
          tool_name: block.name,
        }
      }
    }

    // Execute the tool
    try {
      const result = await tool.call(block.input, context)
      return { ...result, tool_use_id: block.id, tool_name: block.name }
    } catch (err: any) {
      return {
        type: 'tool_result',
        tool_use_id: block.id,
        content: `Tool execution error: ${err.message}`,
        is_error: true,
        tool_name: block.name,
      }
    }
  }

  /**
   * Get current messages for session persistence.
   */
  getMessages(): Anthropic.MessageParam[] {
    return [...this.messages]
  }

  /**
   * Get total usage across all turns.
   */
  getUsage(): TokenUsage {
    return { ...this.totalUsage }
  }

  /**
   * Get total cost.
   */
  getCost(): number {
    return this.totalCost
  }
}
