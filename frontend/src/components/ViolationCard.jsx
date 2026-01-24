/**
 * ViolationCard Component
 * Displays individual person detection with PPE status
 */

const ViolationCard = ({ person }) => {
    const isSafe = person.has_helmet && person.has_vest
    const cardClass = isSafe ? 'violation-card--safe' : ''

    // Get icon based on status
    const getIcon = () => {
        if (isSafe) return '✅'
        if (!person.has_helmet && !person.has_vest) return '🚨'
        if (!person.has_helmet) return '⛑️'
        return '🦺'
    }

    // Get status text
    const getStatus = () => {
        if (isSafe) return 'Compliant'
        const missing = []
        if (!person.has_helmet) missing.push('Helmet')
        if (!person.has_vest) missing.push('Vest')
        return `Missing: ${missing.join(', ')}`
    }

    // Get decision path badge color
    const getPathBadge = () => {
        const path = person.decision_path
        if (path.includes('Fast Safe')) return 'badge--safe'
        if (path.includes('Fast Violation')) return 'badge--violation'
        return 'badge--info'
    }

    return (
        <div className={`violation-card ${cardClass}`}>
            <div className="violation-card__icon">
                {getIcon()}
            </div>

            <div className="violation-card__content">
                <div className="violation-card__title">
                    Person #{person.person_id + 1}
                    {!isSafe && (
                        <span className="badge badge--violation" style={{ marginLeft: '0.5rem' }}>
                            {person.violation_type?.replace('_', ' ').toUpperCase()}
                        </span>
                    )}
                </div>

                <div className="violation-card__detail">
                    {getStatus()}
                </div>

                <div className="flex gap-sm mt-sm">
                    <span className={`badge ${getPathBadge()}`}>
                        {person.decision_path}
                    </span>

                    {person.sam_activated && (
                        <span className="badge badge--info">SAM Used</span>
                    )}

                    <span style={{ fontSize: '0.75rem', color: 'var(--text-muted)' }}>
                        {(person.confidence * 100).toFixed(0)}% conf
                    </span>
                </div>
            </div>

            <div style={{
                display: 'flex',
                flexDirection: 'column',
                alignItems: 'center',
                gap: '0.25rem',
                padding: '0.5rem',
                background: 'var(--bg-tertiary)',
                borderRadius: 'var(--radius-md)',
                minWidth: '60px'
            }}>
                <div style={{ display: 'flex', gap: '0.25rem' }}>
                    <span title="Helmet" style={{ fontSize: '1.25rem' }}>
                        {person.has_helmet ? '⛑️' : '❌'}
                    </span>
                    <span title="Vest" style={{ fontSize: '1.25rem' }}>
                        {person.has_vest ? '🦺' : '❌'}
                    </span>
                </div>
            </div>
        </div>
    )
}

export default ViolationCard
