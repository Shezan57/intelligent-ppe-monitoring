/**
 * ChatBot Component — PPE Violation Query Assistant
 * 
 * Provides a chat interface for site managers to ask
 * natural language questions about violations.
 */

import { useState, useRef, useEffect, useCallback } from 'react'
import axios from 'axios'

const API_BASE = import.meta.env.VITE_API_URL || 'http://localhost:8000'

// Suggested questions for quick access
const SUGGESTED_QUESTIONS = [
    "How many violations today?",
    "What's the compliance rate?",
    "Which camera has the most violations?",
    "Show me all helmet violations this week",
    "What are the most common violation types?",
]

function ChatBot() {
    const [messages, setMessages] = useState([
        {
            role: 'assistant',
            content: '👋 Hi! I\'m your PPE Safety Assistant. Ask me anything about violations, compliance rates, or detection statistics.\n\nTry: *"How many violations today?"*',
            timestamp: new Date()
        }
    ])
    const [input, setInput] = useState('')
    const [isLoading, setIsLoading] = useState(false)
    const [isAvailable, setIsAvailable] = useState(null)
    const messagesEndRef = useRef(null)
    const inputRef = useRef(null)

    // Check chatbot availability on mount
    useEffect(() => {
        axios.get(`${API_BASE}/api/chatbot/status`)
            .then(res => setIsAvailable(res.data.available))
            .catch(() => setIsAvailable(false))
    }, [])

    // Auto-scroll to bottom of messages
    useEffect(() => {
        messagesEndRef.current?.scrollIntoView({ behavior: 'smooth' })
    }, [messages])

    // Send message handler
    const handleSend = useCallback(async (questionOverride) => {
        const question = questionOverride || input.trim()
        if (!question || isLoading) return

        // Add user message
        const userMessage = { role: 'user', content: question, timestamp: new Date() }
        setMessages(prev => [...prev, userMessage])
        setInput('')
        setIsLoading(true)

        try {
            const res = await axios.post(`${API_BASE}/api/chatbot/ask`, { question })
            const data = res.data

            let content = data.answer || 'No response received.'

            // If we got tabular data, format it
            if (data.data && data.data.length > 0 && data.data.length <= 10) {
                const keys = Object.keys(data.data[0])
                content += '\n\n'
                // Simple table
                data.data.forEach((row, i) => {
                    const parts = keys.map(k => `**${k}**: ${row[k]}`)
                    content += `${i + 1}. ${parts.join(' | ')}\n`
                })
            }

            if (data.total_rows && data.total_rows > 10) {
                content += `\n\n*Showing 10 of ${data.total_rows} results.*`
            }

            setMessages(prev => [...prev, {
                role: 'assistant',
                content,
                sql: data.sql,
                timestamp: new Date()
            }])
        } catch (err) {
            setMessages(prev => [...prev, {
                role: 'assistant',
                content: '❌ Failed to reach the server. Is the backend running?',
                timestamp: new Date()
            }])
        } finally {
            setIsLoading(false)
            inputRef.current?.focus()
        }
    }, [input, isLoading])

    // Handle Enter key
    const handleKeyDown = (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault()
            handleSend()
        }
    }

    return (
        <div style={{
            display: 'flex',
            flexDirection: 'column',
            height: 'calc(100vh - 200px)',
            maxHeight: '700px',
            borderRadius: '12px',
            overflow: 'hidden',
            border: '1px solid var(--border-color)',
            background: 'var(--bg-primary, #1a1a2e)'
        }}>
            {/* Header */}
            <div style={{
                padding: '1rem 1.5rem',
                background: 'linear-gradient(135deg, #667eea 0%, #764ba2 100%)',
                color: 'white',
                display: 'flex',
                alignItems: 'center',
                gap: '0.75rem'
            }}>
                <span style={{ fontSize: '1.5rem' }}>🤖</span>
                <div>
                    <h3 style={{ margin: 0, fontSize: '1rem' }}>PPE Safety Assistant</h3>
                    <small style={{ opacity: 0.8 }}>
                        {isAvailable === null ? 'Checking...' :
                            isAvailable ? '● Online — Powered by OpenAI' :
                                '○ Offline — Set OPENAI_API_KEY'}
                    </small>
                </div>
            </div>

            {/* Messages Area */}
            <div style={{
                flex: 1,
                overflowY: 'auto',
                padding: '1rem',
                display: 'flex',
                flexDirection: 'column',
                gap: '0.75rem'
            }}>
                {messages.map((msg, i) => (
                    <div key={i} style={{
                        display: 'flex',
                        justifyContent: msg.role === 'user' ? 'flex-end' : 'flex-start',
                        alignItems: 'flex-start',
                        gap: '0.5rem'
                    }}>
                        {msg.role === 'assistant' && (
                            <span style={{
                                fontSize: '1.2rem',
                                marginTop: '4px',
                                flexShrink: 0
                            }}>🤖</span>
                        )}
                        <div style={{
                            maxWidth: '80%',
                            padding: '0.75rem 1rem',
                            borderRadius: msg.role === 'user'
                                ? '12px 12px 2px 12px'
                                : '12px 12px 12px 2px',
                            background: msg.role === 'user'
                                ? 'linear-gradient(135deg, #667eea, #764ba2)'
                                : 'var(--bg-secondary, #16213e)',
                            color: 'white',
                            fontSize: '0.9rem',
                            lineHeight: '1.5',
                            whiteSpace: 'pre-wrap',
                            wordBreak: 'break-word',
                            boxShadow: '0 1px 3px rgba(0,0,0,0.2)'
                        }}>
                            {msg.content}
                            {msg.sql && (
                                <details style={{ marginTop: '0.5rem', fontSize: '0.75rem', opacity: 0.7 }}>
                                    <summary style={{ cursor: 'pointer' }}>View SQL</summary>
                                    <code style={{
                                        display: 'block',
                                        marginTop: '0.25rem',
                                        padding: '0.5rem',
                                        background: 'rgba(0,0,0,0.3)',
                                        borderRadius: '4px',
                                        fontFamily: 'monospace',
                                        fontSize: '0.7rem'
                                    }}>{msg.sql}</code>
                                </details>
                            )}
                        </div>
                        {msg.role === 'user' && (
                            <span style={{
                                fontSize: '1.2rem',
                                marginTop: '4px',
                                flexShrink: 0
                            }}>👷</span>
                        )}
                    </div>
                ))}

                {isLoading && (
                    <div style={{
                        display: 'flex',
                        alignItems: 'center',
                        gap: '0.5rem',
                        padding: '0.75rem 1rem',
                        color: 'var(--text-muted)'
                    }}>
                        <span>🤖</span>
                        <div style={{
                            display: 'flex',
                            gap: '4px'
                        }}>
                            <span className="typing-dot" style={{ animationDelay: '0ms' }}>●</span>
                            <span className="typing-dot" style={{ animationDelay: '150ms' }}>●</span>
                            <span className="typing-dot" style={{ animationDelay: '300ms' }}>●</span>
                        </div>
                    </div>
                )}

                <div ref={messagesEndRef} />
            </div>

            {/* Suggested Questions */}
            {messages.length <= 2 && (
                <div style={{
                    padding: '0.5rem 1rem',
                    display: 'flex',
                    flexWrap: 'wrap',
                    gap: '0.5rem',
                    borderTop: '1px solid var(--border-color)',
                }}>
                    {SUGGESTED_QUESTIONS.map((q, i) => (
                        <button
                            key={i}
                            onClick={() => handleSend(q)}
                            disabled={isLoading}
                            style={{
                                padding: '0.35rem 0.75rem',
                                borderRadius: '16px',
                                border: '1px solid rgba(102,126,234,0.4)',
                                background: 'rgba(102,126,234,0.1)',
                                color: '#667eea',
                                fontSize: '0.75rem',
                                cursor: 'pointer',
                                transition: 'all 0.2s',
                            }}
                            onMouseEnter={e => {
                                e.target.style.background = 'rgba(102,126,234,0.25)'
                                e.target.style.borderColor = '#667eea'
                            }}
                            onMouseLeave={e => {
                                e.target.style.background = 'rgba(102,126,234,0.1)'
                                e.target.style.borderColor = 'rgba(102,126,234,0.4)'
                            }}
                        >
                            {q}
                        </button>
                    ))}
                </div>
            )}

            {/* Input Area */}
            <div style={{
                padding: '0.75rem 1rem',
                borderTop: '1px solid var(--border-color)',
                display: 'flex',
                gap: '0.5rem',
                alignItems: 'center'
            }}>
                <input
                    ref={inputRef}
                    type="text"
                    value={input}
                    onChange={e => setInput(e.target.value)}
                    onKeyDown={handleKeyDown}
                    placeholder="Ask about violations, compliance, stats..."
                    disabled={isLoading}
                    style={{
                        flex: 1,
                        padding: '0.65rem 1rem',
                        borderRadius: '20px',
                        border: '1px solid var(--border-color)',
                        background: 'var(--bg-secondary, #16213e)',
                        color: 'var(--text-primary, white)',
                        fontSize: '0.9rem',
                        outline: 'none',
                    }}
                />
                <button
                    onClick={() => handleSend()}
                    disabled={!input.trim() || isLoading}
                    style={{
                        width: '40px',
                        height: '40px',
                        borderRadius: '50%',
                        border: 'none',
                        background: input.trim()
                            ? 'linear-gradient(135deg, #667eea, #764ba2)'
                            : 'var(--bg-secondary, #16213e)',
                        color: 'white',
                        fontSize: '1.1rem',
                        cursor: input.trim() ? 'pointer' : 'default',
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        transition: 'all 0.2s',
                        flexShrink: 0
                    }}
                >
                    ➤
                </button>
            </div>
        </div>
    )
}

export default ChatBot
