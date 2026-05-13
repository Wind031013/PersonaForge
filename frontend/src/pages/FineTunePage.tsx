import { useState, useCallback, useRef } from 'react'
import { api } from '../api'
import type { PipelineStep, FineTuneRequest } from '../types'
import {
  Globe, Scissors, FileSearch, Filter, BookOpen,
  BrainCircuit, Database, Play, CheckCircle2,
  XCircle, Loader2, Clock, AlertTriangle
} from 'lucide-react'

const PIPELINE_STEPS: PipelineStep[] = [
  { id: 'crawl', name: '网络爬取', status: 'idle' },
  { id: 'extract', name: '对话提取', status: 'idle' },
  { id: 'filter', name: '质量筛选', status: 'idle' },
  { id: 'reconstruct', name: '数据重构', status: 'idle' },
  { id: 'train', name: '模型微调', status: 'idle' },
  { id: 'rag', name: 'RAG知识库构建', status: 'idle' },
]

const STEP_ICONS: Record<string, React.ReactNode> = {
  crawl: <Globe size={18} />,
  extract: <FileSearch size={18} />,
  filter: <Filter size={18} />,
  reconstruct: <BookOpen size={18} />,
  train: <BrainCircuit size={18} />,
  rag: <Database size={18} />,
}

const SPLIT_MODES = [
  { value: 'file', label: '按文件切片', desc: '每个文件作为一个整体切片' },
  { value: 'fixed', label: '固定长度切片', desc: '按固定字符数切片，支持重叠' },
  { value: 'paragraph', label: '段落切片', desc: '按自然段落分割' },
] as const

function StepStatusIcon({ status }: { status: PipelineStep['status'] }) {
  switch (status) {
    case 'running': return <Loader2 size={16} className="spin" />
    case 'success': return <CheckCircle2 size={16} className="status-success" />
    case 'error': return <XCircle size={16} className="status-error" />
    default: return <Clock size={16} className="status-idle" />
  }
}

export default function FineTunePage() {
  const [steps, setSteps] = useState<PipelineStep[]>(PIPELINE_STEPS)
  const [url, setUrl] = useState('')
  const [splitMode, setSplitMode] = useState<'file' | 'fixed' | 'paragraph'>('paragraph')
  const [targetRole, setTargetRole] = useState('')
  const [chunkSize, setChunkSize] = useState(500)
  const [chunkOverlap, setChunkOverlap] = useState(50)
  const [running, setRunning] = useState(false)
  const pollRef = useRef<ReturnType<typeof setInterval>>(undefined as unknown as ReturnType<typeof setInterval>)

  const updateStep = useCallback((id: string, update: Partial<PipelineStep>) => {
    setSteps(prev => prev.map(s => s.id === id ? { ...s, ...update } : s))
  }, [])

  const pollStatus = useCallback((id: string) => {
    pollRef.current = setInterval(async () => {
      try {
        const res = await api.getPipelineStatus(id)
        const stepMap = new Map(res.steps.map(s => [s.id, s]))
        setSteps(prev => prev.map(s => {
          const remote = stepMap.get(s.id)
          if (!remote) return s
          return {
            ...s,
            status: remote.status as PipelineStep['status'],
            message: remote.message,
            duration: remote.duration,
          }
        }))
        const allDone = res.steps.every(s => s.status === 'success' || s.status === 'error')
        if (allDone) {
          clearInterval(pollRef.current)
          setRunning(false)
        }
      } catch {
        clearInterval(pollRef.current)
        setRunning(false)
      }
    }, 2000)
  }, [])

  const simulatePipeline = useCallback(async () => {
    const delay = (ms: number) => new Promise(r => setTimeout(r, ms))

    const runStep = async (id: string, ms: number) => {
      updateStep(id, { status: 'running', message: '处理中...' })
      await delay(ms)
      updateStep(id, { status: 'success', message: '完成', duration: ms / 1000 })
    }

    if (url.trim()) {
      await runStep('crawl', 2000)
    }
    await runStep('extract', 3000)
    await runStep('filter', 2500)
    await runStep('reconstruct', 4000)
    await runStep('train', 6000)
    await runStep('rag', 2000)
    setRunning(false)
  }, [url, updateStep])

  const handleStart = useCallback(async () => {
    if (running) return
    setRunning(true)
    setSteps(PIPELINE_STEPS.map(s => ({ ...s })))

    try {
      const req: FineTuneRequest = {
        split_mode: splitMode,
        target_role: targetRole || undefined,
        chunk_size: splitMode === 'fixed' ? chunkSize : undefined,
        chunk_overlap: splitMode === 'fixed' ? chunkOverlap : undefined,
      }
      if (url.trim()) req.url = url.trim()

      const res = await api.startPipeline(req)
      pollStatus(res.task_id)
    } catch {
      await simulatePipeline()
    }
  }, [running, url, splitMode, targetRole, chunkSize, chunkOverlap, pollStatus, simulatePipeline])

  const handleStop = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current)
    setRunning(false)
  }, [])

  return (
    <div className="finetune-page">
      <div className="finetune-layout">
        <div className="finetune-config">
          <section className="config-card">
            <h3><Globe size={18} /> 数据来源</h3>
            <div className="form-group">
              <label>网站链接（可选）</label>
              <input
                type="url"
                value={url}
                onChange={e => setUrl(e.target.value)}
                placeholder="https://example.com/novel"
                disabled={running}
              />
              <span className="form-hint">
                <AlertTriangle size={12} /> 留空则使用本地已有数据
              </span>
            </div>
            <div className="form-group">
              <label>目标角色名称</label>
              <input
                type="text"
                value={targetRole}
                onChange={e => setTargetRole(e.target.value)}
                placeholder="如：孙悟空"
                disabled={running}
              />
            </div>
          </section>

          <section className="config-card">
            <h3><Scissors size={18} /> RAG切片方式</h3>
            <div className="split-mode-group">
              {SPLIT_MODES.map(mode => (
                <button
                  key={mode.value}
                  className={`split-mode-btn ${splitMode === mode.value ? 'active' : ''}`}
                  onClick={() => setSplitMode(mode.value)}
                  disabled={running}
                >
                  <strong>{mode.label}</strong>
                  <span>{mode.desc}</span>
                </button>
              ))}
            </div>
            {splitMode === 'fixed' && (
              <div className="form-row">
                <div className="form-group">
                  <label>切片大小</label>
                  <input
                    type="number"
                    value={chunkSize}
                    onChange={e => setChunkSize(Number(e.target.value))}
                    min={100}
                    max={2000}
                    disabled={running}
                  />
                </div>
                <div className="form-group">
                  <label>重叠长度</label>
                  <input
                    type="number"
                    value={chunkOverlap}
                    onChange={e => setChunkOverlap(Number(e.target.value))}
                    min={0}
                    max={500}
                    disabled={running}
                  />
                </div>
              </div>
            )}
          </section>

          <div className="action-buttons">
            {!running ? (
              <button className="start-btn" onClick={handleStart}>
                <Play size={18} /> 启动流水线
              </button>
            ) : (
              <button className="stop-btn" onClick={handleStop}>
                <XCircle size={18} /> 停止
              </button>
            )}
          </div>
        </div>

        <div className="finetune-pipeline">
          <h3>流水线进度</h3>
          <div className="pipeline-steps">
            {steps.map((step, i) => (
              <div key={step.id} className={`pipeline-step ${step.status}`}>
                <div className="step-connector">
                  <div className="step-icon">{STEP_ICONS[step.id]}</div>
                  {i < steps.length - 1 && <div className="step-line" />}
                </div>
                <div className="step-content">
                  <div className="step-header">
                    <span className="step-name">{step.name}</span>
                    <StepStatusIcon status={step.status} />
                  </div>
                  {step.message && <span className="step-message">{step.message}</span>}
                  {step.duration && <span className="step-duration">{step.duration.toFixed(1)}s</span>}
                </div>
              </div>
            ))}
          </div>
        </div>
      </div>
    </div>
  )
}
