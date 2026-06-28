import { useState } from 'react'
import './App.css'

function App() {
  const [messages, setMessages] = useState([])
  const [input, setInput] = useState('')
  const [loading, setLoading] = useState(false)
  const [selectedSource, setSelectedSource] = useState(null)

  const sendMessage = async () => {
    if (!input.trim()) return

    const userMessage = { role: 'user', content: input }
    setMessages((prev) => [...prev, userMessage])
    setInput('')
    setLoading(true)

    try {
      const response = await fetch('http://16.16.92.153:8000/chat', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({ query: input, top_k: 3 }),
      })
      const data = await response.json()

      const aiMessage = { role: 'assistant', content: data.answer, sources: data.sources }
      setMessages((prev) => [...prev, aiMessage])
    } catch (error) {
      const errorMessage = { role: 'assistant', content: "Erreur : impossible de contacter l'API." }
      setMessages((prev) => [...prev, errorMessage])
    } finally {
      setLoading(false)
    }
  }

  const handleKeyPress = (e) => {
    if (e.key === 'Enter') {
      sendMessage()
    }
  }

  return (
    <div className="chat-container">
      <h1>RAG Chat Assistant</h1>

      <div className="messages-area">
        {messages.map((msg, index) => (
          <div key={index} className={`message ${msg.role}`}>
            <strong>{msg.role === 'user' ? 'Vous' : 'Assistant'}</strong>
            <p>{msg.content}</p>

            {msg.sources && msg.sources.length > 0 && (
              <div className="pills-container">
                {msg.sources.map((src, i) => (
                  <button
                    key={i}
                    className="source-pill"
                    onClick={() => setSelectedSource(src)}
                  >
                    📄 {src.source.split('/').pop()} - p.{src.page}
                  </button>
                ))}
              </div>
            )}
          </div>
        ))}

        {loading && (
          <div className="message assistant">
            <strong>Assistant</strong>
            <p className="spinner">⏳ En train de réfléchir...</p>
          </div>
        )}
      </div>

      <div className="input-area">
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyPress={handleKeyPress}
          placeholder="Posez votre question..."
        />
        <button onClick={sendMessage} disabled={loading}>
          Envoyer
        </button>
      </div>

      {/* Modale d'affichage de la source */}
      {selectedSource && (
        <div className="modal-overlay" onClick={() => setSelectedSource(null)}>
          <div className="modal-content" onClick={(e) => e.stopPropagation()}>
            <button className="modal-close" onClick={() => setSelectedSource(null)}>✕</button>
            <h3>📄 {selectedSource.source.split('/').pop()}</h3>
            <p className="modal-page">Page {selectedSource.page}</p>
            <div className="modal-extract">
              {selectedSource.extract}
            </div>
          </div>
        </div>
      )}
    </div>
  )
}

export default App