import { StrictMode } from 'react'
import { createRoot } from 'react-dom/client'
import { BrowserRouter, Routes, Route } from 'react-router-dom'
import './index.css'
import App from './App'
import ChatPage from './pages/ChatPage'
import FineTunePage from './pages/FineTunePage'

createRoot(document.getElementById('root')!).render(
  <StrictMode>
    <BrowserRouter>
      <Routes>
        <Route path="/" element={<App />}>
          <Route index element={<ChatPage />} />
          <Route path="finetune" element={<FineTunePage />} />
        </Route>
      </Routes>
    </BrowserRouter>
  </StrictMode>,
)
