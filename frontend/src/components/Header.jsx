/**
 * Header Component
 * Application header with logo and navigation
 */

const Header = () => {
    return (
        <header className="header">
            <div className="header__logo">
                <svg
                    className="header__logo-icon"
                    viewBox="0 0 100 100"
                    fill="none"
                    style={{ width: '40px', height: '40px' }}
                >
                    <ellipse cx="50" cy="55" rx="35" ry="25" fill="#667eea" />
                    <path d="M15 55 Q15 30 50 25 Q85 30 85 55" fill="#764ba2" />
                    <rect x="20" y="50" width="60" height="8" rx="4" fill="#667eea" />
                    <ellipse cx="50" cy="32" rx="15" ry="8" fill="#f8fafc" opacity="0.3" />
                </svg>
                <h1 className="header__title">PPE Monitor</h1>
            </div>

            <nav className="header__nav">
                <span style={{
                    fontSize: '0.75rem',
                    color: 'var(--text-muted)',
                    padding: '0.5rem 1rem',
                    background: 'var(--bg-tertiary)',
                    borderRadius: 'var(--radius-md)'
                }}>
                    🎓 Master's Thesis Project
                </span>
            </nav>
        </header>
    )
}

export default Header
