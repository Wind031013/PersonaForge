import { NavLink, Outlet } from 'react-router-dom'
import { MessageSquare, Wrench } from 'lucide-react'

export default function App() {
  return (
    <div className="app-layout">
      <nav className="app-nav">
        <div className="nav-brand">
          <span className="brand-icon">🎭</span>
          <span className="brand-text">Agent驱动的角色语料工程与RAG增强的对话生成系统</span>
        </div>
        <div className="nav-links">
          <NavLink to="/" end className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <MessageSquare size={18} />
            <span>角色对话</span>
          </NavLink>
          <NavLink to="/finetune" className={({ isActive }) => `nav-link ${isActive ? 'active' : ''}`}>
            <Wrench size={18} />
            <span>模型微调</span>
          </NavLink>
        </div>
      </nav>
      <main className="app-main">
        <Outlet />
      </main>
    </div>
  )
}
