export interface ChatMessage {
  role: 'user' | 'assistant' | 'system'
  content: string
  time?: string
}

export interface ChatRequest {
  messages: ChatMessage[]
  max_tokens?: number
  temperature?: number
  top_p?: number
  repetition_penalty?: number
}

export interface ChatResponse {
  role: string
  content: string
  time: string
  status: string
  tokens_used?: number
}

export interface Role {
  id: string
  name: string
  description: string
  avatar?: string
  system_prompt?: string
}

export interface PipelineStep {
  id: string
  name: string
  status: 'idle' | 'running' | 'success' | 'error'
  message?: string
  duration?: number
}

export interface FineTuneRequest {
  url?: string
  split_mode: 'file' | 'fixed' | 'paragraph'
  target_role?: string
  chunk_size?: number
  chunk_overlap?: number
}

export interface FineTuneStatus {
  step: string
  status: 'idle' | 'running' | 'success' | 'error'
  message: string
  progress?: number
}

export interface CrawlRequest {
  url: string
  output_dir?: string
}
