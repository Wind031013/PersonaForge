import { useState, useRef, useEffect, useCallback } from 'react'
import { api } from '../api'
import type { ChatMessage, Role } from '../types'
import { Send, Bot, User, Loader2, ChevronDown } from 'lucide-react'

const DEFAULT_ROLES: Role[] = [
  { id: 'wukong', name: '孙悟空', description: '齐天大圣，桀骜不驯、嫉恶如仇，口头禅"俺老孙"', avatar: '🐒' },
  { id: 'bajie', name: '猪八戒', description: '天蓬元帅，贪吃好色但憨厚可爱', avatar: '🐷' },
  { id: 'shaseng', name: '沙悟净', description: '卷帘大将，忠厚老实，任劳任怨', avatar: '🧔' },
  { id: 'tangseng', name: '唐僧', description: '金蝉子转世，慈悲为怀，有时迂腐', avatar: '🧘' },
]

const ROLE_SYSTEM_PROMPTS: Record<string, string> = {
  wukong: '你是孙悟空，齐天大圣。你说话时常用"俺老孙"自称，性格桀骜不驯、嫉恶如仇、重情重义。你武艺高强，有一双火眼金睛，会七十二变。面对敌人时狂傲自信，面对师父时恭敬有加，面对师弟们则喜欢开玩笑调侃。',
  bajie: '你是猪八戒，原天蓬元帅。你贪吃好色，喜欢偷懒，遇到困难总想分行李回高老庄。但你本性善良，关键时刻也能挺身而出。你常叫孙悟空"猴哥"，叫沙僧"沙师弟"。',
  shaseng: '你是沙悟净，原卷帘大将。你性格忠厚老实，任劳任怨，是取经团队中最稳重的人。你常说的口头禅是"大师兄，师父被妖怪抓走了！"',
  tangseng: '你是唐僧，金蝉子转世。你慈悲为怀，一心向佛，有时显得迂腐。你对徒弟们有严格的要求，但内心深爱他们。你常念紧箍咒来管教孙悟空。',
}

export default function ChatPage() {
  const [roles, setRoles] = useState<Role[]>(DEFAULT_ROLES)
  const [selectedRole, setSelectedRole] = useState<Role>(DEFAULT_ROLES[0])
  const [messages, setMessages] = useState<ChatMessage[]>([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [roleMenuOpen, setRoleMenuOpen] = useState(false)
  const bottomRef = useRef<HTMLDivElement>(null)
  const inputRef = useRef<HTMLTextAreaElement>(null)

  useEffect(() => {
    bottomRef.current?.scrollIntoView({ behavior: 'smooth' })
  }, [messages])

  useEffect(() => {
    api.getRoles().then(res => {
      if (res.roles?.length) setRoles(res.roles)
    }).catch(() => {})
  }, [])

  const handleSelectRole = useCallback((role: Role) => {
    setSelectedRole(role)
    setMessages([])
    setRoleMenuOpen(false)
  }, [])

  const handleSend = useCallback(async () => {
    if (!input.trim() || loading) return

    const userMsg: ChatMessage = {
      role: 'user',
      content: input.trim(),
      time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
    }
    setMessages(prev => [...prev, userMsg])
    setInput('')
    setLoading(true)

    try {
      const systemPrompt = ROLE_SYSTEM_PROMPTS[selectedRole.id] || `你是${selectedRole.name}。${selectedRole.description}`
      const chatMessages: ChatMessage[] = [
        { role: 'system', content: systemPrompt },
        ...messages.filter(m => m.role !== 'system'),
        userMsg,
      ]

      const res = await api.chat({
        messages: chatMessages,
        temperature: 0.7,
        max_tokens: 512,
      })

      setMessages(prev => [...prev, {
        role: 'assistant',
        content: res.content,
        time: res.time,
      }])
    } catch (err) {
      setMessages(prev => [...prev, {
        role: 'assistant',
        content: `出错了: ${err instanceof Error ? err.message : '未知错误'}`,
        time: new Date().toLocaleTimeString('zh-CN', { hour: '2-digit', minute: '2-digit' }),
      }])
    } finally {
      setLoading(false)
      inputRef.current?.focus()
    }
  }, [input, loading, messages, selectedRole])

  const handleKeyDown = useCallback((e: React.KeyboardEvent) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }, [handleSend])

  return (
    <div className="chat-page">
      <header className="chat-header">
        <div className="role-selector">
          <button className="role-selector-btn" onClick={() => setRoleMenuOpen(!roleMenuOpen)}>
            <span className="role-avatar">{selectedRole.avatar || '🤖'}</span>
            <span className="role-name">{selectedRole.name}</span>
            <ChevronDown size={16} className={`chevron ${roleMenuOpen ? 'open' : ''}`} />
          </button>
          {roleMenuOpen && (
            <div className="role-dropdown">
              {roles.map(role => (
                <button
                  key={role.id}
                  className={`role-option ${role.id === selectedRole.id ? 'active' : ''}`}
                  onClick={() => handleSelectRole(role)}
                >
                  <span className="role-avatar">{role.avatar || '🤖'}</span>
                  <div className="role-info">
                    <span className="role-name">{role.name}</span>
                    <span className="role-desc">{role.description}</span>
                  </div>
                </button>
              ))}
            </div>
          )}
        </div>
        <button className="clear-btn" onClick={() => setMessages([])}>
          清空对话
        </button>
      </header>

      <div className="chat-messages">
        {messages.length === 0 && (
          <div className="chat-empty">
            <div className="empty-icon">{selectedRole.avatar || '🤖'}</div>
            <h3>与 {selectedRole.name} 开始对话</h3>
            <p>{selectedRole.description}</p>
          </div>
        )}
        {messages.map((msg, i) => (
          <div key={i} className={`message ${msg.role}`}>
            <div className="message-avatar">
              {msg.role === 'user' ? <User size={18} /> : <Bot size={18} />}
            </div>
            <div className="message-body">
              <div className="message-content">{msg.content}</div>
              <div className="message-time">{msg.time}</div>
            </div>
          </div>
        ))}
        {loading && (
          <div className="message assistant">
            <div className="message-avatar"><Bot size={18} /></div>
            <div className="message-body">
              <div className="message-content typing">
                <Loader2 size={16} className="spin" />
                <span>正在思考...</span>
              </div>
            </div>
          </div>
        )}
        <div ref={bottomRef} />
      </div>

      <div className="chat-input-area">
        <textarea
          ref={inputRef}
          value={input}
          onChange={e => setInput(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={`输入消息，与${selectedRole.name}对话...`}
          rows={1}
        />
        <button className="send-btn" onClick={handleSend} disabled={loading || !input.trim()}>
          <Send size={18} />
        </button>
      </div>
    </div>
  )
}
